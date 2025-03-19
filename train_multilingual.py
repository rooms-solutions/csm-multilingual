import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import multiprocessing
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Fix CUDA multiprocessing issue by setting the start method to 'spawn'
if __name__ == "__main__":
    # This must happen at the beginning before any other multiprocessing code
    multiprocessing.set_start_method('spawn', force=True)

from generator import load_llama3_tokenizer
from models import Model, ModelArgs, _index_causal_mask
from moshi.models import loaders
from multilingual_dataset import create_dataset_for_language, multilingual_collate_fn
from language_utils import LanguageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger("train_multilingual")

def process_batch(model, text_tokens, audio_tokens, device):
    """Process a single batch and calculate the loss"""
    # Debug prints to verify input shapes and types
    print(f"Debug - text_tokens shape: {text_tokens.shape}, audio_tokens shape: {audio_tokens.shape}")
    print(f"Debug - model dtype: {next(model.parameters()).dtype}")
    
    # Create input format - use clone() to avoid in-place modifications
    b, s = text_tokens.size()
    text_frame = torch.zeros(b, s, 33, dtype=torch.long, device=device)
    text_frame[:, :, -1] = text_tokens.clone()
    text_frame_mask = torch.zeros(b, s, 33, dtype=torch.bool, device=device)
    text_frame_mask[:, :, -1] = True
    
    # Get input positions - avoid in-place repeat
    input_pos = torch.arange(s, device=device).unsqueeze(0).expand(b, s)
    
    # Forward pass through backbone
    embeds = model._embed_tokens(text_frame)
    masked_embeds = embeds * text_frame_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    
    curr_backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)
    backbone_output = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    
    # Get model's dtype and ensure consistent dtype for operations
    dtype = next(model.parameters()).dtype
    backbone_output = backbone_output.to(dtype=dtype)
    
    # Last hidden state for each sequence
    last_h = backbone_output[:, -1, :].unsqueeze(1)  # [B, 1, D]
    
    # Codebook predictions and loss calculation
    num_codebooks = audio_tokens.size(1)
    total_loss = 0
    
    # First codebook prediction using the backbone output
    c0_logits = model.codebook0_head(last_h.squeeze(1).to(dtype=dtype))
    
    # Extract target as 1D tensor
    if audio_tokens.dim() == 3:  # If shape is [batch_size, num_codebooks, sequence_length]
        c0_targets = audio_tokens[:, 0, 0].clone()  # Get first token of first codebook
    else:  # If shape is [batch_size, num_codebooks]
        c0_targets = audio_tokens[:, 0].clone()  # Get first codebook
    
    # Make sure targets are 1D
    c0_targets = c0_targets.view(-1)
    c0_loss = nn.functional.cross_entropy(c0_logits, c0_targets)
    total_loss += c0_loss
    
    # Teacher forcing for remaining codebooks
    if audio_tokens.dim() == 3:
        audio_embed = model._embed_audio(0, audio_tokens[:, 0, 0].clone().view(-1, 1))
    else:
        audio_embed = model._embed_audio(0, audio_tokens[:, 0].clone().view(-1, 1))
    
    # Fix dimensions - ensure audio_embed matches last_h dimensions
    if audio_embed.dim() == 4:  # If shape is [B, 1, 1, H]
        audio_embed = audio_embed.squeeze(2)  # Remove the extra dimension
        
    # Concatenate along sequence dimension - create new tensor
    curr_h = torch.cat([last_h, audio_embed], dim=1)
    # Create new position tensor without repeat (use expand)
    curr_pos = torch.tensor([[0, 1]], device=device).expand(b, 2)
    
    # Process through decoder for subsequent codebooks
    for i in range(1, num_codebooks):
        # Use the decoder to predict next codebook
        curr_decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
        
        # Make sure input_pos is properly sized for the decoder
        # The sequence length should always be 2 (not growing with each iteration)
        if curr_pos.size(1) != 2:
            # Reset position indices to just [0, 1] for each batch
            curr_pos = torch.tensor([[0, 1]], device=device).expand(b, 2)
            
        decoder_input = model.projection(curr_h).to(dtype=dtype)
        decoder_h = model.decoder(decoder_input, input_pos=curr_pos, mask=curr_decoder_mask)
        
        # Ensure decoder_h has the correct dtype
        decoder_h = decoder_h.to(dtype=dtype)
        
        # Get logits with proper dtype consistency
        ci_logits = torch.matmul(
            decoder_h[:, -1, :].unsqueeze(1), 
            model.audio_head[i-1].to(dtype=dtype)
        ).squeeze(1)
        
        # Extract target as 1D tensor - always clone to avoid in-place issues
        if audio_tokens.dim() == 3:
            ci_targets = audio_tokens[:, i, 0].clone()
        else:
            ci_targets = audio_tokens[:, i].clone()
            
        # Make sure targets are 1D
        ci_targets = ci_targets.view(-1)
        
        # Calculate loss
        ci_loss = nn.functional.cross_entropy(ci_logits, ci_targets)
        total_loss += ci_loss
        
        # For next iteration, if not the last codebook
        if i < num_codebooks - 1:
            # Get embedding for the next codebook token - always clone inputs
            if audio_tokens.dim() == 3:
                ci_embed = model._embed_audio(i, audio_tokens[:, i, 0].clone().view(-1, 1))
            else:
                ci_embed = model._embed_audio(i, audio_tokens[:, i].clone().view(-1, 1))
            
            # Fix dimensions - ensure consistent shape
            if ci_embed.dim() == 4:
                ci_embed = ci_embed.squeeze(2)
                
            # Use a fresh position tensor each time, always keeping it at size [b, 2]
            # This ensures we always have position indices [0, 1] for each batch
            curr_h = ci_embed
            curr_pos = torch.tensor([[0, 1]], device=device).expand(b, 2)
    
    return total_loss

def evaluate(model, val_loader, device):
    """Evaluate model on validation data"""
    model.eval()
    total_val_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Prepare input tensors - ensure they're on the right device
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Reset and setup caches
            model.reset_caches()
            model.setup_caches(text_tokens.size(0))
            
            # Process batch
            val_loss = process_batch(model, text_tokens, audio_tokens, device)
            total_val_loss += val_loss.item()
            total_batches += 1
    
    return total_val_loss / total_batches if total_batches > 0 else float('inf')

def train(args):
    # Enable anomaly detection to help identify gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # Configure logging level 
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Set device
    device = torch.device(args.device)
    
    # Load text tokenizer
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi) - download weights first
    from huggingface_hub import hf_hub_download
    logger.info("Downloading Mimi codec weights...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    logger.info(f"Downloaded Mimi codec weights to {mimi_weight}")
    mimi_model = loaders.get_mimi(mimi_weight, device=device)
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
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pre-trained model weights from {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting with fresh model weights")
    
    # Determine the correct root directory - either the main data_dir or the language subdirectory
    lang_specific_dir = os.path.join(args.data_dir, args.language)
    if os.path.exists(os.path.join(lang_specific_dir, "clips")):
        root_dir = lang_specific_dir
        logger.info(f"Using language-specific directory as root: {root_dir}")
    else:
        root_dir = args.data_dir
        logger.info(f"Using main data directory as root: {root_dir}")
    
    # Create dataset for specified language
    train_dataset = create_dataset_for_language(
        language=args.language,
        csv_file=args.train_csv,
        root_dir=root_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
        max_audio_length=args.max_audio_length,
    )
    
    # Create train dataloader with safer settings for CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if args.device == "cpu" else 0,  # Use 0 workers with CUDA to avoid forking issues
        pin_memory=False,  # Disable pin_memory to avoid CUDA tensor pinning issues
        drop_last=True,
        collate_fn=multilingual_collate_fn  # Use our custom collate function
    )
    
    # Create validation dataloader if provided
    val_loader = None
    if args.val_csv:
        val_dataset = create_dataset_for_language(
            language=args.language,
            csv_file=args.val_csv,
            root_dir=root_dir,
            mimi_model=mimi_model,
            text_tokenizer=text_tokenizer,
            max_audio_length=args.max_audio_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if args.device == "cpu" else 0,  # Use 0 workers with CUDA
            pin_memory=False,  # Disable pin_memory to avoid CUDA tensor pinning issues
            collate_fn=multilingual_collate_fn  # Use our custom collate function
        )
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.num_epochs
    )
    
    # Mixed precision training
    scaler = GradScaler() if args.use_amp else None
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create output directory with language info
    output_dir = os.path.join(args.output_dir, args.language)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training config
    config = vars(args)
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    global_step = 0
    logger.info(f"Starting training for language: {args.language}")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare input tensors - text_tokens and audio_tokens need to be on the device
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Note: audio_waveform is now a list (not tensor) and not needed for training
            
            # Setup model caches (must be done before reset)
            model.setup_caches(text_tokens.size(0))
            
            # Reset caches
            model.reset_caches()
            
            # Properly set up caches with the correct batch size
            # First reset all caches completely
            model.reset_caches()
            # Then set up with the current batch size
            model.setup_caches(text_tokens.size(0))
            # Debug log to verify cache setup
            logger.debug(f"Set up caches for batch size: {text_tokens.size(0)}")
            
            # Forward pass and loss calculation
            if args.use_amp:
                with autocast():
                    total_loss = process_batch(model, text_tokens, audio_tokens, device)
                
                # Optimize with mixed precision
                optimizer.zero_grad(set_to_none=True)  # More efficient
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss = process_batch(model, text_tokens, audio_tokens, device)
                
                # Standard optimization
                optimizer.zero_grad(set_to_none=True)  # More efficient
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            scheduler.step()
            epoch_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=total_loss.item())
            
            # Save checkpoint periodically
            global_step += 1
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{args.patience}")
                
                if args.patience > 0 and patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save checkpoint at the end of each epoch
        epoch_checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        logger.info(f"Completed epoch {epoch+1}, saved checkpoint to {epoch_checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train CSM 1B for Multilingual TTS")
    
    # Data arguments
    parser.add_argument("--language", type=str, required=True, 
                        help="Language code (e.g., 'de' for German, 'en' for English)")
    parser.add_argument("--train_csv", type=str, required=True, 
                        help="Path to training data TSV file")
    parser.add_argument("--val_csv", type=str, default=None, 
                        help="Path to validation data TSV file (optional)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing all data")
    parser.add_argument("--max_audio_length", type=int, default=None,
                        help="Maximum audio length in samples (optional)")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Base output directory for checkpoints")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Save checkpoint every X steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="Gradient clipping")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device (cuda or cpu)")
    parser.add_argument("--use_amp", action="store_true", 
                        help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Patience for early stopping (0 to disable)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
