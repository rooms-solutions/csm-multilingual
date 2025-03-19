#!/usr/bin/env python3
"""
Simple script to test tensor dimensions and cache issues with a single batch.
"""

import os
import torch
import logging
import argparse
from pathlib import Path

# Disable cache system entirely
os.environ["TORCHTUNE_DISABLE_CACHE"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_test")

from generator import load_llama3_tokenizer
from models import Model, ModelArgs
from moshi.models import loaders
from multilingual_dataset import create_dataset_for_language, multilingual_collate_fn
from torch.utils.data import DataLoader

def test_single_batch(language="de", batch_size=4, data_dir="./data"):
    """Process a single batch through the model and print tensor shapes."""
    # Print torch version for reference
    print(f"Using PyTorch {torch.__version__}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load text tokenizer
    text_tokenizer = load_llama3_tokenizer()
    
    # Load Mimi audio codec
    from huggingface_hub import hf_hub_download
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
    
    # Create model with bfloat16 precision
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    print(f"Model created: {model.__class__.__name__}")
    
    # Configure test dataset
    csv_file = f"./data/{language}/train_{language}.tsv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Create dataset and dataloader
    test_dataset = create_dataset_for_language(
        language=language,
        csv_file=csv_file,
        root_dir=data_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
    )
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multilingual_collate_fn,
        num_workers=0
    )
    
    # Get one batch
    print(f"Getting batch from dataloader...")
    batch = next(iter(dataloader))
    
    # Extract tensors
    text_tokens = batch["text_tokens"].to(device)
    audio_tokens = batch["audio_tokens"].to(device)
    
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Audio tokens shape: {audio_tokens.shape}")
    
    # Set up model masks
    model.setup_caches(text_tokens.size(0))
    
    # Create input tensors
    b, s = text_tokens.size()
    text_frame = torch.zeros(b, s, 33, dtype=torch.long, device=device)
    text_frame[:, :, -1] = text_tokens
    text_frame_mask = torch.zeros(b, s, 33, dtype=torch.bool, device=device)
    text_frame_mask[:, :, -1] = True
    
    # Get input positions
    input_pos = torch.arange(s, device=device).unsqueeze(0).expand(b, s)
    
    # Print shapes
    print(f"Text frame shape: {text_frame.shape}")
    print(f"Text frame mask shape: {text_frame_mask.shape}")
    print(f"Input positions shape: {input_pos.shape}")
    
    # Embed tokens
    print(f"Embedding tokens...")
    embeds = model._embed_tokens(text_frame)
    print(f"Embeds shape: {embeds.shape}")
    
    masked_embeds = embeds * text_frame_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    print(f"Hidden state shape: {h.shape}")
    
    # Create a properly shaped backbone causal mask
    seq_len = text_tokens.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    curr_backbone_mask = causal_mask.unsqueeze(0).expand(b, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    print(f"Backbone mask shape: {curr_backbone_mask.shape}")
    
    # Forward pass through backbone
    print(f"Running backbone forward pass...")
    backbone_output = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    print(f"Backbone output shape: {backbone_output.shape}")
    
    # Test decoder with position [0, 1]
    print(f"Testing decoder with fixed positions...")
    decoder_positions = torch.tensor([[0, 1]], device=device).expand(b, 2)
    print(f"Decoder positions shape: {decoder_positions.shape}")
    
    # Create decoder mask
    decoder_mask = torch.tril(torch.ones(2, 2, dtype=torch.bool, device=device)).unsqueeze(0).expand(b, 2, 2)
    print(f"Decoder mask shape: {decoder_mask.shape}")
    
    # Create random decoder input
    decoder_input = torch.randn(b, 2, model.decoder.embed_dim, device=device, dtype=next(model.parameters()).dtype)
    print(f"Decoder input shape: {decoder_input.shape}")
    
    # Forward pass through decoder
    print(f"Running decoder forward pass...")
    try:
        decoder_output = model.decoder(decoder_input, input_pos=decoder_positions, mask=decoder_mask)
        print(f"✅ Decoder output shape: {decoder_output.shape}")
        print(f"Test successful! All tensor dimensions are compatible.")
        return True
    except Exception as e:
        print(f"❌ Decoder error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test cache and tensor dimension issues")
    parser.add_argument("--language", type=str, default="de", help="Language code")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    
    args = parser.parse_args()
    test_single_batch(args.language, args.batch_size, args.data_dir)

if __name__ == "__main__":
    main()
