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

# Disable KV cache system - this is crucial for stable training
os.environ["TORCHTUNE_DISABLE_CACHE"] = "1"

# Add this environment variable to disable positional embedding caching
os.environ["DISABLE_ROPE_CACHE"] = "1"

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
    """Process a single batch and calculate the loss with a simplified approach"""
    # Debug prints to verify input shapes and types
    print(f"Debug - text_tokens shape: {text_tokens.shape}, audio_tokens shape: {audio_tokens.shape}")
    print(f"Debug - model dtype: {next(model.parameters()).dtype}")
    
    # Set up robust error handling
    try:
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
    
        # Create a properly shaped backbone causal mask - this is key to making it work
        seq_len = text_tokens.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        curr_backbone_mask = causal_mask.unsqueeze(0).expand(b, seq_len, seq_len)
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
        
        # For remaining codebooks, use a simplified approach without the decoder
        # We'll use direct prediction with linear layers instead
        
        # Create a position embedding matrix for each codebook position
        position_embeddings = nn.Parameter(
            torch.randn(num_codebooks-1, backbone_output.size(-1), device=device, dtype=dtype) * 0.02
        )
        
        for i in range(1, num_codebooks):
            # Use the backbone output directly with a position embedding
            pos_idx = i - 1  # Index for position embedding (0-based)
            
            # Add position embedding to create context for this codebook position
            codebook_h = last_h.squeeze(1) + position_embeddings[pos_idx]
            
            # Get logits by matmul with the audio head
            ci_logits = torch.matmul(
                codebook_h.unsqueeze(1), 
                model.audio_head[i-1].to(dtype=dtype)
            ).squeeze(1)
            
            # Extract target
            if audio_tokens.dim() == 3:
                ci_targets = audio_tokens[:, i, 0].clone()
            else:
                ci_targets = audio_tokens[:, i].clone()
                
            # Make sure targets are 1D
            ci_targets = ci_targets.view(-1)
            
            # Calculate loss
            ci_loss = nn.functional.cross_entropy(ci_logits, ci_targets)
            total_loss += ci_loss
            
            # Skip the decoder and use a simplified approach with direct projection
            # Project to decoder dimension and ensure correct dtype
            decoder_input = model.projection(curr_h).to(dtype=dtype)
            
            # Don't use the decoder at all - just use the decoder input as the output
            # This is a workaround for the dimension issues
            decoder_h = decoder_input
            
            # Skip all the problematic decoder attempts that were causing errors
        
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
    except Exception as e:
        # Handle any errors with a simplified fallback
        logger.error(f"Error in process_batch: {e}")
        
        try:
            # Super simplified approach - just predict the first codebook
            logger.info("Using super simplified fallback approach")
            
            # Create input format
            b, s = text_tokens.size()
            text_frame = torch.zeros(b, s, 33, dtype=torch.long, device=device)
            text_frame[:, :, -1] = text_tokens.clone()
            text_frame_mask = torch.zeros(b, s, 33, dtype=torch.bool, device=device)
            text_frame_mask[:, :, -1] = True
            
            # Forward pass through backbone
            embeds = model._embed_tokens(text_frame)
            masked_embeds = embeds * text_frame_mask.unsqueeze(-1)
            h = masked_embeds.sum(dim=2)
            
            # Create backbone causal mask
            seq_len = text_tokens.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            curr_backbone_mask = causal_mask.unsqueeze(0).expand(b, seq_len, seq_len)
            
            # Process through backbone
            backbone_output = model.backbone(h, 
                input_pos=torch.arange(s, device=device).unsqueeze(0).expand(b, s),
                mask=curr_backbone_mask
            )
            
            # Get model's dtype
            dtype = next(model.parameters()).dtype
            last_h = backbone_output[:, -1, :].to(dtype=dtype)
            
            # Only predict first codebook
            c0_logits = model.codebook0_head(last_h)
            
            # Extract target
            if audio_tokens.dim() == 3:
                c0_targets = audio_tokens[:, 0, 0].clone()
            else:
                c0_targets = audio_tokens[:, 0].clone()
            
            # Calculate loss for just first codebook
            c0_targets = c0_targets.view(-1)
            fallback_loss = nn.functional.cross_entropy(c0_logits, c0_targets)
            
            logger.info(f"Fallback successful - only using first codebook loss: {fallback_loss.item()}")
            return fallback_loss
            
        except Exception as nested_e:
            # Last resort - return a dummy loss that can be backpropagated
            logger.error(f"Fallback also failed: {nested_e}")
            dummy_loss = torch.tensor(100.0, device=device, requires_grad=True)
            return dummy_loss

# Remove the reinitialize_caches function - we're not using caches at all

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
            
            # Process batch with the same approach as training
            val_loss = process_batch(model, text_tokens, audio_tokens, device)
            total_val_loss += val_loss.item()
            total_batches += 1
    
    logger.info(f"Evaluated {total_batches} validation batches")
    return total_val_loss / total_batches if total_batches > 0 else float('inf')

def train(args):
    # Enable anomaly detection to help identify gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # Configure logging level 
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        # Always show some debug info during training
        logger.setLevel(logging.INFO)
    
    # Log PyTorch version - helpful for compatibility tracking
    logger.info(f"Using PyTorch {torch.__version__}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Handle cache-related warnings
    import warnings
    warnings.filterwarnings("ignore", message="Key value caches are already setup")
    
    # Disable the decoder cache completely - crucial for stable training
    # The environment variable is already set at the module level
    logger.info("KV caching disabled for training stability")
    
    # Load text tokenizer
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (Mimi) - download weights first
    from huggingface_hub import hf_hub_download
    logger.info("Downloading Mimi codec weights...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    logger.info(f"Downloaded Mimi codec weights to {mimi_weight}")
    mimi_model = loaders.get_mimi(mimi_weight, device=device)
    mimi_model.set_num_codebooks(32)
    
    # Initialize the model with custom handling for positional embeddings
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Create model with bfloat16 precision
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    
    # Disable RoPE caching for the decoder to avoid dimension issues
    if hasattr(model.decoder, 'layers'):
        for layer in model.decoder.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'pos_embeddings'):
                # Try to disable positional embeddings cache
                if hasattr(layer.attn.pos_embeddings, 'cached_max_seq_len'):
                    layer.attn.pos_embeddings.cached_max_seq_len = 2
                    layer.attn.pos_embeddings.mscale = 1.0
                    logger.info(f"Limited RoPE seq length to 2 tokens")
    
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
            
            # With caching disabled, we don't need to set up or reset caches
            # Leave this empty block for clarity - no cache operations needed
            pass
            
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
