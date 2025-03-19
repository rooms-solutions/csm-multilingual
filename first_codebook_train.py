#!/usr/bin/env python3
"""
Simplified training script that only trains on the first codebook prediction
to avoid all decoder-related problems.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

# Disable KV cache system and positional embedding caching
os.environ["TORCHTUNE_DISABLE_CACHE"] = "1"
os.environ["DISABLE_ROPE_CACHE"] = "1"

from generator import load_llama3_tokenizer
from models import Model, ModelArgs
from moshi.models import loaders
from multilingual_dataset import create_dataset_for_language, multilingual_collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("first_codebook_train")

def first_codebook_batch(model, text_tokens, audio_tokens, device):
    """Process batch using only the first codebook prediction"""
    # Create input format
    b, s = text_tokens.size()
    text_frame = torch.zeros(b, s, 33, dtype=torch.long, device=device)
    text_frame[:, :, -1] = text_tokens.clone()
    text_frame_mask = torch.zeros(b, s, 33, dtype=torch.bool, device=device)
    text_frame_mask[:, :, -1] = True
    
    # Get input positions
    input_pos = torch.arange(s, device=device).unsqueeze(0).expand(b, s)
    
    # Forward pass through backbone
    embeds = model._embed_tokens(text_frame)
    masked_embeds = embeds * text_frame_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    
    # Create backbone causal mask
    seq_len = text_tokens.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    curr_backbone_mask = causal_mask.unsqueeze(0).expand(b, seq_len, seq_len)
    
    # Forward pass through backbone
    backbone_output = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    
    # Get model's dtype
    dtype = next(model.parameters()).dtype
    backbone_output = backbone_output.to(dtype=dtype)
    
    # Last hidden state for each sequence
    last_h = backbone_output[:, -1, :]
    
    # First codebook prediction only
    c0_logits = model.codebook0_head(last_h)
    
    # Extract target
    if audio_tokens.dim() == 3:
        c0_targets = audio_tokens[:, 0, 0].clone()
    else:
        c0_targets = audio_tokens[:, 0].clone()
    
    # Calculate loss for just the first codebook
    c0_targets = c0_targets.view(-1)
    loss = nn.functional.cross_entropy(c0_logits, c0_targets)
    
    return loss

def train_first_codebook(args):
    """Train only on first codebook prediction to avoid decoder issues"""
    logger.info(f"Using PyTorch {torch.__version__}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizers
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi)
    from huggingface_hub import hf_hub_download
    logger.info("Downloading Mimi codec weights...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi_model = loaders.get_mimi(mimi_weight, device=device)
    mimi_model.set_num_codebooks(32)
    
    # Initialize model
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
            logger.info(f"Loaded model weights from {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    # Create datasets
    logger.info(f"Loading datasets from {args.data_dir}")
    train_dataset = create_dataset_for_language(
        language=args.language,
        csv_file=args.train_csv,
        root_dir=args.data_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid CUDA issues
        collate_fn=multilingual_collate_fn,
    )
    
    # Create validation dataloader if provided
    val_loader = None
    if args.val_csv:
        val_dataset = create_dataset_for_language(
            language=args.language,
            csv_file=args.val_csv,
            root_dir=args.data_dir,
            mimi_model=mimi_model,
            text_tokenizer=text_tokenizer,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=multilingual_collate_fn
        )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.num_epochs
    )
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.language)
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting first-codebook-only training for {args.language}")
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move tensors to device
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Forward pass and loss calculation
            loss = first_codebook_batch(model, text_tokens, audio_tokens, device)
            
            # Optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            # Update LR scheduler
            scheduler.step()
            
            # Track statistics
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Save checkpoint periodically
            if (batch_idx + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"fc-checkpoint-{epoch+1}-{batch_idx+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    text_tokens = batch["text_tokens"].to(device)
                    audio_tokens = batch["audio_tokens"].to(device)
                    loss = first_codebook_batch(model, text_tokens, audio_tokens, device)
                    val_loss += loss.item()
                
            val_loss /= len(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(output_dir, "fc-best-model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{args.patience}")
                
                if args.patience > 0 and patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(output_dir, f"fc-model-epoch-{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved epoch checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "fc-final-model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="First-Codebook-Only CSM Training")
    
    # Data arguments
    parser.add_argument("--language", type=str, required=True, 
                        help="Language code (e.g., 'de' for German)")
    parser.add_argument("--train_csv", type=str, required=True, 
                        help="Path to training data TSV file")
    parser.add_argument("--val_csv", type=str, default=None, 
                        help="Path to validation data TSV file (optional)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing data")
    
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
    parser.add_argument("--patience", type=int, default=5, 
                        help="Patience for early stopping (0 to disable)")
    
    args = parser.parse_args()
    train_first_codebook(args)

if __name__ == "__main__":
    main()
