#!/usr/bin/env python3
"""
Simplified training script that focuses only on the first codebook prediction
to avoid decoder-related issues.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import multiprocessing
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
logger = logging.getLogger("simplified_train")

def process_batch_simple(model, text_tokens, audio_tokens, device):
    """
    Simplified batch processing that only predicts the first codebook
    to avoid decoder-related dimension issues.
    """
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

def train_simple(args):
    """Simplified training function that focuses only on first codebook prediction"""
    logger.info(f"Using PyTorch {torch.__version__}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizers
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi)
    from huggingface_hub import hf_hub_download
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
    
    # Create dataset
    train_dataset = create_dataset_for_language(
        language=args.language,
        csv_file=args.train_csv,
        root_dir=args.data_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multilingual_collate_fn,
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.language)
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting simplified training for {args.language}")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Process batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move tensors to device
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Forward pass and loss calculation (first codebook only)
            loss = process_batch_simple(model, text_tokens, audio_tokens, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Save checkpoint periodically
            if (batch_idx + 1) % 500 == 0:
                checkpoint_path = os.path.join(output_dir, f"simple-checkpoint-{epoch+1}-{batch_idx+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(output_dir, f"simple-model-epoch-{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved epoch checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "simple-final-model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Simplified CSM 1B Training")
    
    parser.add_argument("--language", type=str, required=True, 
                        help="Language code (e.g., 'de' for German)")
    parser.add_argument("--train_csv", type=str, required=True, 
                        help="Path to training data TSV file")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing all data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Base output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    train_simple(args)

if __name__ == "__main__":
    main()
