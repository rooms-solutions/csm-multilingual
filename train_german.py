import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generator import load_llama3_tokenizer
from models import Model, ModelArgs, _index_causal_mask
from moshi.models import loaders

class GermanVoiceDataset(Dataset):
    def __init__(self, csv_file, root_dir, mimi_model, text_tokenizer):
        """
        Args:
            csv_file (string): Path to the CSV with annotations (clips.tsv from Common Voice)
            root_dir (string): Directory with all the audio files
            mimi_model: The Mimi codec model for audio tokenization
            text_tokenizer: The text tokenizer
        """
        self.voice_frame = pd.read_csv(csv_file, delimiter='\t')
        self.root_dir = root_dir
        self.text_tokenizer = text_tokenizer
        self.mimi = mimi_model
        
    def __len__(self):
        return len(self.voice_frame)
        
    def __getitem__(self, idx):
        # Load audio file and resample to 24kHz
        audio_path = os.path.join(self.root_dir, self.voice_frame.iloc[idx, 1])
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        if sample_rate != 24000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
        
        # Get text and speaker_id (or use a default)
        text = self.voice_frame.iloc[idx, 2]  # Assuming column 2 contains the text
        speaker_id = 0  # You might want to map speaker ids appropriately
        
        # Tokenize text
        text_tokens = self.text_tokenizer.encode(f"[{speaker_id}]{text}")
        
        # Tokenize audio using Mimi
        with torch.no_grad():
            audio_tokens = self.mimi.encode(waveform.unsqueeze(0).unsqueeze(0))[0]
        
        return {
            "text": text,
            "text_tokens": torch.tensor(text_tokens),
            "audio_tokens": audio_tokens,
            "audio_waveform": waveform,
            "speaker_id": speaker_id
        }

def train(args):
    # Set device
    device = torch.device(args.device)
    
    # Load text tokenizer
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi)
    mimi_model = loaders.get_mimi(loaders.MIMI_NAME, device=device)
    mimi_model.set_num_codebooks(32)
    
    # Initialize the model
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    
    # Load pre-trained weights if available
    if args.checkpoint:
        try:
            state_dict = torch.load(args.checkpoint)
            model.load_state_dict(state_dict)
            print(f"Loaded pre-trained model weights from {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting with fresh model weights")
    
    # Create dataset and dataloader
    train_dataset = GermanVoiceDataset(
        csv_file=args.train_csv,
        root_dir=args.audio_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.num_epochs
    )
    
    # Mixed precision training
    scaler = GradScaler() if args.use_amp else None
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare input tensors
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Reset caches
            model.reset_caches()
            
            # Setup model caches
            model.setup_caches(text_tokens.size(0))
            
            # Forward pass and loss calculation
            if args.use_amp:
                with autocast():
                    total_loss = process_batch(model, text_tokens, audio_tokens, device)
                
                # Optimize with mixed precision
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss = process_batch(model, text_tokens, audio_tokens, device)
                
                # Standard optimization
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=total_loss.item())
            
            # Save checkpoint periodically
            global_step += 1
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save checkpoint at the end of each epoch
        epoch_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Completed epoch {epoch+1}, saved checkpoint to {epoch_checkpoint_path}")

def process_batch(model, text_tokens, audio_tokens, device):
    """Process a single batch and calculate the loss"""
    # Create input format
    b, s = text_tokens.size()
    text_frame = torch.zeros(b, s, 33).long().to(device)
    text_frame[:, :, -1] = text_tokens
    text_frame_mask = torch.zeros(b, s, 33).bool().to(device)
    text_frame_mask[:, :, -1] = True
    
    # Get input positions
    input_pos = torch.arange(s, device=device).unsqueeze(0).repeat(b, 1)
    
    # Forward pass through backbone
    embeds = model._embed_tokens(text_frame)
    masked_embeds = embeds * text_frame_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    
    curr_backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)
    backbone_output = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    
    # Last hidden state for each sequence
    last_h = backbone_output[:, -1, :].unsqueeze(1)  # [B, 1, D]
    
    # Codebook predictions and loss calculation
    num_codebooks = audio_tokens.size(1)
    total_loss = 0
    
    # First codebook prediction using the backbone output
    c0_logits = model.codebook0_head(last_h.squeeze(1))
    c0_targets = audio_tokens[:, 0]
    c0_loss = nn.functional.cross_entropy(c0_logits, c0_targets)
    total_loss += c0_loss
    
    # Teacher forcing for remaining codebooks
    curr_h = model._embed_audio(0, audio_tokens[:, 0:1])
    curr_h = torch.cat([last_h, curr_h], dim=1)
    curr_pos = torch.tensor([[0, 1]], device=device).repeat(b, 1)
    
    # Process through decoder for subsequent codebooks
    for i in range(1, num_codebooks):
        # Use the decoder to predict next codebook
        curr_decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
        decoder_h = model.decoder(model.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask)
        
        # Get logits and targets
        ci_logits = torch.matmul(decoder_h[:, -1, :].unsqueeze(1), model.audio_head[i-1]).squeeze(1)
        ci_targets = audio_tokens[:, i]
        
        # Calculate loss
        ci_loss = nn.functional.cross_entropy(ci_logits, ci_targets)
        total_loss += ci_loss
        
        # For next iteration, if not the last codebook
        if i < num_codebooks - 1:
            ci_embed = model._embed_audio(i, audio_tokens[:, i:i+1])
            curr_h = ci_embed
            curr_pos = curr_pos[:, -1:] + 1
    
    return total_loss

def main():
    parser = argparse.ArgumentParser(description="Train CSM 1B on German Mozilla Common Voice")
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, required=True, help="Path to Common Voice TSV file")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
