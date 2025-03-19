#!/usr/bin/env python3
"""
Utility script for testing gradient computation on a single batch
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gradient_check.log")
    ]
)
logger = logging.getLogger("gradient_check")

from generator import load_llama3_tokenizer
from models import Model, ModelArgs
from moshi.models import loaders
from multilingual_dataset import create_dataset_for_language, multilingual_collate_fn
from train_multilingual import process_batch

def check_gradients(args):
    """Test gradient computation on a single batch"""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Set device
    device = torch.device(args.device)
    
    # Load text tokenizer
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi)
    from huggingface_hub import hf_hub_download
    logger.info("Downloading Mimi codec weights...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
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
            logger.info(f"Loaded model weights from {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    # Create test dataset
    test_dataset = create_dataset_for_language(
        language=args.language,
        csv_file=args.csv_file,
        root_dir=args.data_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
        max_audio_length=None,
    )
    
    # Get a single batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simplicity
        collate_fn=multilingual_collate_fn
    )
    
    batch = next(iter(dataloader))
    text_tokens = batch["text_tokens"].to(device)
    audio_tokens = batch["audio_tokens"].to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    # Setup model caches
    model.setup_caches(text_tokens.size(0))
    model.reset_caches()
    
    # Forward and backward pass with detailed logging
    logger.info(f"Running gradient check with batch size {text_tokens.size(0)}")
    try:
        model.train()
        
        # Test with manually tracking parameters
        params_before = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Forward pass
        total_loss = process_batch(model, text_tokens, audio_tokens, device)
        logger.info(f"Forward pass successful, loss: {total_loss.item()}")
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        logger.info("Backward pass successful")
        
        # Check if gradients were properly computed
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        
        grad_norm = grad_norm ** 0.5
        logger.info(f"Gradient norm: {grad_norm}")
        
        # Update
        optimizer.step()
        logger.info("Optimizer step successful")
        
        # Verify parameters changed
        changed_params = 0
        for name, param in model.named_parameters():
            if not torch.allclose(param, params_before[name]):
                changed_params += 1
        
        logger.info(f"{changed_params}/{len(params_before)} parameters changed after update")
        
        print("✅ Gradient check completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during gradient check: {e}", exc_info=True)
        print(f"❌ Gradient check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Gradient computation check")
    
    parser.add_argument("--language", type=str, default="de",
                        help="Language code")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to dataset TSV file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing data")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    check_gradients(args)

if __name__ == "__main__":
    main()
