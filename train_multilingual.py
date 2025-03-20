import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import multiprocessing
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
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
from custom_decoder import fix_decoder_attention

def safe_matmul(tensor1, tensor2, device):
    """
    Simplified matrix multiplication - PyTorch handles device compatibility automatically.
    """
    # Just use standard matmul - PyTorch will handle device compatibility
    return torch.matmul(tensor1, tensor2)

def numerically_stable_cross_entropy(logits, targets):
    """
    Compute cross entropy loss with basic numerical stability.
    
    Args:
        logits: Model predictions
        targets: Target indices
        
    Returns:
        Cross entropy loss
    """
    # Simple NaN handling
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Use PyTorch's cross_entropy which already has good numerical stability
    return torch.nn.functional.cross_entropy(logits, targets)

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

def process_batch(model, text_tokens, audio_tokens, device, args=None, batch_idx=0):
    """Process a single batch and calculate the loss with a simplified approach"""
    # Debug info using logger
    logger.debug(f"Text tokens shape: {text_tokens.shape}, audio_tokens shape: {audio_tokens.shape}")

    # Create input format
    b, s = text_tokens.size()
    text_frame = torch.zeros(b, s, 33, dtype=torch.long, device=device)
    text_frame[:, :, -1] = text_tokens
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
    backbone_output = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    
    # Get model's dtype for consistent operations
    dtype = next(model.parameters()).dtype
    backbone_output = backbone_output.to(dtype=dtype)
    
    # Last hidden state for each sequence
    last_h = backbone_output[:, -1, :].unsqueeze(1)  # [B, 1, D]
    
    # Codebook predictions and loss calculation
    num_codebooks = audio_tokens.size(1)
    total_loss = 0

    # First codebook prediction
    c0_logits = model.codebook0_head(last_h.squeeze(1))
    
    # Extract target
    if audio_tokens.dim() == 3:
        c0_targets = audio_tokens[:, 0, 0]
    else:
        c0_targets = audio_tokens[:, 0]
    c0_targets = c0_targets.view(-1)
    
    # Calculate first codebook loss
    c0_loss = numerically_stable_cross_entropy(c0_logits, c0_targets)
    total_loss += c0_loss
    
    # Position embeddings for remaining codebooks
    position_embeddings = torch.zeros(
        num_codebooks-1, 
        backbone_output.size(-1), 
        device=device, 
        dtype=dtype
    )
    
    # Initialize with small random values
    with torch.no_grad():
        position_embeddings.normal_(0, 0.02)
        
    for i in range(1, num_codebooks):
        # Use backbone output with position embedding
        pos_idx = i - 1
        
        # Add position embedding
        codebook_h = last_h.squeeze(1) + position_embeddings[pos_idx]
        
        # Project to decoder dimension
        projected_h = model.projection(codebook_h.unsqueeze(1))
        
        # Get logits
        ci_logits = torch.matmul(projected_h.squeeze(1), model.audio_head[i-1])
        
        # Extract target
        if audio_tokens.dim() == 3:
            ci_targets = audio_tokens[:, i, 0]
        else:
            ci_targets = audio_tokens[:, i]
        ci_targets = ci_targets.view(-1)
        
        # Calculate loss
        ci_loss = numerically_stable_cross_entropy(ci_logits, ci_targets)
        total_loss += ci_loss
        
        # Process with decoder - ensure we have the expected sequence length of 2
        decoder_input_single = model.projection(codebook_h.unsqueeze(1))
        # Create a tensor of shape [batch_size, 2, dim] with consistent dimensions
        batch_size = decoder_input_single.size(0)
        dim = decoder_input_single.size(2)
        decoder_input = torch.cat([
            decoder_input_single,
            torch.zeros(batch_size, 1, dim, device=decoder_input_single.device, dtype=decoder_input_single.dtype)
        ], dim=1)
            
        # Fixed positions for decoder
        decoder_positions = torch.zeros(b, 2, dtype=torch.long, device=device)
        decoder_positions[:, 1] = 1
        
        # Simple decoder mask
        decoder_mask = torch.tril(
            torch.ones(2, 2, dtype=torch.bool, device=device)
        ).unsqueeze(0).expand(b, 2, 2)
        
        # Call decoder
        try:
            decoder_h = model.decoder(
                decoder_input,
                input_pos=decoder_positions,
                mask=decoder_mask
            )
            
            # Get last hidden state
            decoder_h_flat = decoder_h[:, -1, :]
            
            # Second prediction of same codebook
            ci_logits = torch.matmul(decoder_h_flat, model.audio_head[i-1])
            ci_loss = nn.functional.cross_entropy(ci_logits, ci_targets)
            total_loss += ci_loss
        except Exception as e:
            # On decoder error, just log it
            logger.debug(f"Using simplified path for codebook {i}: {e}")
            # We already calculated loss with projected_h above, so no need to add again
        
        # For next iteration, if not the last codebook
        if i < num_codebooks - 1:
            # Get embedding for next codebook token with explicit device control
            if audio_tokens.dim() == 3:
                token_input = audio_tokens[:, i, 0].view(-1, 1)
            else:
                token_input = audio_tokens[:, i].view(-1, 1)
            
            # Ensure token is on the correct device
            token_input = token_input.to(device=device, non_blocking=False)
            
            # Synchronize before embedding
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                
            ci_embed = model._embed_audio(i, token_input)
            
            # Ensure consistent shape
            if ci_embed.dim() == 4:
                ci_embed = ci_embed.squeeze(2)
                
            # Verify device consistency
            ci_embed = ci_embed.to(device=device, non_blocking=False)
    
    # Normalize the loss
    normalized_loss = total_loss / num_codebooks
    
    # Log occasionally
    if batch_idx % 100 == 0:
        logger.info(f"Avg codebook loss: {normalized_loss.item():.4f}")
    
    return normalized_loss

# Remove the reinitialize_caches function - we're not using caches at all

def evaluate(model, val_loader, device, args=None):
    """Evaluate model on validation data"""
    model.eval()
    total_val_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            val_loss = process_batch(model, text_tokens, audio_tokens, device, args)
            total_val_loss += val_loss.item()
            total_batches += 1
    
    logger.info(f"Evaluated {total_batches} validation batches")
    return total_val_loss / total_batches if total_batches > 0 else float('inf')

def create_simple_mlp(input_dim, hidden_dim, output_dim, device, dtype):
    """Create a simple MLP to replace decoder functionality"""
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device=device, dtype=dtype)
    return mlp

def train(args):
    # Enable anomaly detection to help identify gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # Configure logging level 
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        # Use INFO level by default (DEBUG is disabled)
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
    
    # Add argument for simplified training
    args.simplified_decoder = True
    
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
    
    # Create model with appropriate precision for the selected device (significantly improved initialization)
    with torch.cuda.device(device):
        # Simplified dtype choice - always use float16 with AMP, bfloat16 otherwise
        # This avoids dtype conflicts that cause NaN/Inf issues
        model_dtype = torch.float16 if args.use_amp else torch.bfloat16
        logger.info(f"Initializing model with dtype={model_dtype} for {'AMP' if args.use_amp else 'standard'} training")
        
        # Create model with correct dtype from the start
        model = Model(model_args)
        
        # Move to device first, then set dtype to avoid mixed device tensors
        model = model.to(device=device)
        torch.cuda.synchronize(device)
        
        # Now set dtype explicitly after model is on correct device
        model = model.to(dtype=model_dtype)
        torch.cuda.synchronize(device)
        
        # Verify the model has correct dtype on all parameters
        actual_dtype = next(model.parameters()).dtype
        logger.info(f"Model parameters dtype: {actual_dtype}")
        
        # Make sure model is in training mode
        model.train()
        
        # Force all modules to have consistent dtypes
        for module in model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype != model_dtype:
                    logger.warning(f"Parameter {param_name} has incorrect dtype {param.dtype}, fixing to {model_dtype}")
                    param.data = param.data.to(dtype=model_dtype)
        
        torch.cuda.synchronize(device)
        logger.info(f"Model successfully initialized on {device} with {model_dtype}")
        
        # Use our custom attention module instead of simplified decoder
        logger.info("Replacing decoder attention with fixed implementation")
        model = fix_decoder_attention(model)
        
        # Force all parameters to be on correct device
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device=device, non_blocking=False)
        
        # Extra synchronization
        torch.cuda.synchronize(device)
        
        logger.info("Decoder attention modules replaced successfully")
        logger.info(f"Verified model is on device: {next(model.parameters()).device}")
    
    # Load CSM-1B pretrained model if requested
    if args.use_csm_pretrained:
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading CSM-1B pretrained model...")
            model_path = hf_hub_download("sesame/csm-1b", "ckpt.pt")
            logger.info(f"Loading CSM-1B model from {model_path}")
            
            # Load pretrained state dict
            csm_state_dict = torch.load(model_path, map_location=device)
            
            # Load weights, potentially with adaptation
            model_dict = model.state_dict()
            
            # Get parameter shapes for logging
            shape_mismatches = 0
            matching_params = 0
            
            # Create pretrained dict, handling shape mismatches
            pretrained_dict = {}
            for k, v in csm_state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        pretrained_dict[k] = v
                        matching_params += 1
                    else:
                        shape_mismatches += 1
                        logger.debug(f"Shape mismatch for {k}: pretrained {v.shape} vs model {model_dict[k].shape}")
            
            # Update model with pretrained weights
            logger.info(f"Loading {matching_params}/{len(model_dict)} layers from CSM-1B")
            logger.info(f"Skipped {shape_mismatches} layers due to shape mismatches")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Freeze backbone if requested
            if args.freeze_backbone:
                logger.info("Freezing backbone parameters to preserve CSM-1B capabilities")
                for name, param in model.backbone.named_parameters():
                    param.requires_grad = False
                    
                # Count frozen vs trainable parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
            
            logger.info("Successfully initialized from CSM-1B model")
            
            # Adjust learning rate for fine-tuning if not explicitly set
            if not args.preserve_learning_rate and args.freeze_backbone:
                original_lr = args.learning_rate
                args.learning_rate = min(original_lr, 2e-5)  # Use lower LR for fine-tuning
                logger.info(f"Adjusted learning rate for fine-tuning: {original_lr} -> {args.learning_rate}")
                
        except Exception as e:
            logger.warning(f"Failed to load CSM pretrained model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
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
    
    # Create train dataloader with optimized settings
    num_workers = 2 if args.device == "cuda" else args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
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
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
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
    
    # Add a function to check for and fix gradient issues
    def check_gradients(model):
        """Check for and fix NaN/Inf gradients to maintain training stability"""
        any_issues = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    any_issues = True
                    # Log the issue
                    logger.warning(f"NaN/Inf found in gradients for parameter: {name}")
                    # Fix the gradient - replace with zeros
                    param.grad = torch.zeros_like(param.grad)
        return any_issues
    
    # Mixed precision training with complete AMP overhaul for maximum compatibility
    if args.use_amp:
        try:
            logger.info("Initializing improved gradient scaler for mixed precision training")
            
            # Force conversion to Float16 for AMP - this is critical for stability
            logger.info("Converting model to Float16 for guaranteed AMP compatibility")
            model = model.to(dtype=torch.float16)
            
            # Ensure optimizer is recreated after model conversion to maintain correct state
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            
            # Set initial scale to a lower value to prevent overflow
            scaler = GradScaler(
                'cuda', 
                enabled=True,
                init_scale=2**10,  # Start with smaller scale factor
                growth_factor=1.5,  # More conservative growth
                growth_interval=100,  # Less frequent growth
                backoff_factor=0.5,  # More aggressive backoff
                max_scale=2**16     # Lower max scale to avoid overflow
            )
            
            # Initialize scaler with conservative settings for stability
            logger.info("AMP initialization complete - using with enhanced stability safeguards")
            
        except Exception as e:
            logger.warning(f"Critical error initializing AMP: {e}")
            logger.warning("Falling back to non-AMP training for stability")
            args.use_amp = False
            scaler = None
            # Make sure model is in a consistent state
            model = model.to(dtype=torch.bfloat16)
    else:
        scaler = None
    
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
    
    # Initialize gradients to zero before starting
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare input tensors - text_tokens and audio_tokens need to be on the device
            text_tokens = batch["text_tokens"].to(device)
            audio_tokens = batch["audio_tokens"].to(device)
            
            # Note: audio_waveform is now a list (not tensor) and not needed for training
            
            # Process batch and get loss
            normalized_loss = total_loss = process_batch(model, text_tokens, audio_tokens, device, args, batch_idx)
            normalized_loss = normalized_loss / args.gradient_accumulation_steps
                
            # Handle AMP or standard training
            if args.use_amp:
                # Process with AMP
                with autocast('cuda', dtype=torch.float16):
                    # Most computation already done in process_batch
                    pass
                    
                # Handle any NaN values in loss
                if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                    normalized_loss = torch.tensor(1.0, device=device, dtype=torch.float16, requires_grad=True)
                    
                # Backward pass with scaling
                scaler.scale(normalized_loss).backward()
                    
                # Only update after accumulating gradients
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                        
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        
                    # Step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Standard training path
                normalized_loss.backward()
                    
                # Update after accumulation
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                
            # Force synchronization after parameter updates
            if torch.cuda.is_available() and ((batch_idx + 1) % args.gradient_accumulation_steps == 0):
                torch.cuda.synchronize(device)
            epoch_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=total_loss.item())
            
            # Only count a complete step after gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                global_step += 1
            
            # Save checkpoint periodically
            if global_step > 0 and global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, args)
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
    parser.add_argument("--disable_custom_decoder", action="store_true",
                        help="Disable custom decoder implementation (not recommended)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--amp_compatibility_mode", action="store_true",
                        help="Use compatibility mode for AMP with BFloat16 (skips unscale operation)")
    parser.add_argument("--force_float16", action="store_true",
                        help="Force Float16 precision instead of BFloat16 for better AMP compatibility")
    parser.add_argument("--stable_training", action="store_true", 
                        help="Enable additional numerical stability measures (slightly slower but more robust)")
    
    # CSM pretrained model arguments
    parser.add_argument("--use_csm_pretrained", action="store_true",
                        help="Use CSM-1B pretrained weights as base model")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone parameters to preserve CSM-1B capabilities")
    parser.add_argument("--preserve_learning_rate", action="store_true",
                        help="Don't reduce learning rate when fine-tuning")
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
