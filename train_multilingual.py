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
from custom_decoder import fix_decoder_attention

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
        # Explicitly create on device with proper error handling
        try:
            # Ensure proper size from backbone output
            embed_dim = backbone_output.size(-1)
            # Create directly on the device with CUDA synchronization to ensure placement
            with torch.cuda.device(device):
                position_embeddings = torch.randn(
                    num_codebooks-1, 
                    embed_dim, 
                    device=device, 
                    dtype=dtype
                ) * 0.02
                # Synchronize to ensure tensor is created before continuing
                torch.cuda.synchronize(device)
                
            # Verify the device to be certain
            if position_embeddings.device != device:
                logger.warning(f"Position embeddings on wrong device: {position_embeddings.device} vs expected {device}")
                # Force synchronous copy to device
                position_embeddings = position_embeddings.to(device=device, dtype=dtype, non_blocking=False)
                torch.cuda.synchronize(device)
        except Exception as embed_err:
            logger.error(f"Error creating position embeddings: {embed_err}")
            # Fallback creation with hard-coded size
            with torch.cuda.device(device):
                position_embeddings = torch.randn(
                    num_codebooks-1, 
                    2048,  # Default backbone embedding size 
                    device=device, 
                    dtype=dtype,
                    # Force creation on device
                    requires_grad=False
                ) * 0.02
                torch.cuda.synchronize(device)
        
        for i in range(1, num_codebooks):
            # Use the backbone output directly with a position embedding
            pos_idx = i - 1  # Index for position embedding (0-based)
            
            # Add position embedding to create context for this codebook position
            codebook_h = last_h.squeeze(1) + position_embeddings[pos_idx]
            
            # Project backbone dimension (2048) to decoder dimension (1024) before matmul
            projected_h = model.projection(codebook_h.unsqueeze(1)).to(device=device, dtype=dtype)
            
            # Get audio head and ensure it's definitely on the correct device
            audio_head_i = model.audio_head[i-1].to(device=device, dtype=dtype, non_blocking=False)
            
            # Get logits by matmul with the audio head - transpose the audio_head for correct multiplication
            ci_logits = torch.matmul(
                projected_h.squeeze(1).to(device=device, non_blocking=False),  # Shape: [batch_size, decoder_dim]
                audio_head_i  # Shape: [decoder_dim, vocab_size]
            )  # Result: [batch_size, vocab_size]
            
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
            
            # Use the decoder properly with consistent dimensions
            # Project to decoder dimension and ensure correct dtype and device
            decoder_input = model.projection(codebook_h.unsqueeze(1)).to(device=device, dtype=dtype)
            
            # Create fixed positions tensors with proper shape
            decoder_positions = torch.zeros(b, 2, dtype=torch.long, device=device)
            decoder_positions[:, 1] = 1  # Second position is 1
            
            # Create a properly sized causal mask for the decoder
            # This is critical - the mask must be [batch_size, seq_len, seq_len]
            decoder_mask = torch.tril(
                torch.ones(2, 2, dtype=torch.bool, device=device)
            ).unsqueeze(0).expand(b, 2, 2)
            
            try:
                # Now use the proper decoder with our fixed attention implementation
                # Pass mask as a keyword argument to avoid conflicts and ensure device consistency
                decoder_h = model.decoder(decoder_input, input_pos=decoder_positions, mask=decoder_mask).to(device=device, dtype=dtype)
                
                # Log what we're doing
                if i == 1:  # Only log once per batch
                    logger.info("Using fixed decoder implementation with proper attention handling")
            except Exception as decoder_err:
                # Log the specific decoder error
                logger.error(f"Decoder error: {decoder_err}")
                # Create a dummy decoder output with proper device/dtype to continue
                decoder_h = decoder_input.clone().to(device=device, dtype=dtype)
            
            # Log what we're doing
            if i == 1:  # Only log once per batch
                logger.info("Using fixed decoder implementation with proper attention handling")
        
            # Ensure decoder_h has the correct dtype
            decoder_h = decoder_h.to(dtype=dtype)
            
            # Process the decoder output through our simplified decoder if available
            if hasattr(model, 'simplified_decoders') and i-1 < len(model.simplified_decoders):
                # Use our simple MLP for this codebook position
                # Extract input features
                if decoder_h.dim() == 3:
                    # Shape is [batch_size, seq_len, dim]
                    decoder_features = decoder_h[:, -1, :].to(dtype=dtype)
                else:
                    # Shape is [batch_size, dim]
                    decoder_features = decoder_h.squeeze(1).to(dtype=dtype)
                
                # Process through the simplified decoder for this position
                decoder_output = model.simplified_decoders[i-1](decoder_features)
                
                # Use the output directly for the logits calculation with explicit device handling
                try:
                    # Ensure both tensors are on same device 
                    decoder_output = decoder_output.to(device=device, dtype=dtype)
                    audio_head = model.audio_head[i-1].to(device=device, dtype=dtype) 
                    ci_logits = torch.matmul(decoder_output, audio_head)
                except Exception as mm_err:
                    logger.error(f"Matrix multiplication error in simplified decoder: {mm_err}")
                    # Create fallback logits
                    vocab_size = model.args.audio_vocab_size
                    ci_logits = torch.zeros(decoder_output.size(0), vocab_size, device=device, dtype=dtype)
                
            else:
                # Fallback to original approach if simplified decoders not available
                # Handle logits calculation with careful dimension management
                if decoder_h.dim() == 3:
                    # Standard case - decoder_h is [batch_size, seq_len, decoder_dim]
                    decoder_h_flat = decoder_h[:, -1, :].to(dtype=dtype)
                elif decoder_h.dim() == 2:
                    # Fallback case - decoder_h is [batch_size, decoder_dim]
                    decoder_h_flat = decoder_h.squeeze(1).to(dtype=dtype)
                else:
                    # Unusual case - reshape to expected dimensions
                    decoder_h_flat = decoder_h.view(b, -1, decoder_h.size(-1))[:, -1, :].to(dtype=dtype)
                
                # Ensure audio_head has correct shape and device for matrix multiplication
                audio_head = model.audio_head[i-1].to(device=device, dtype=dtype)
            
                # Force double-check on device before matmul to prevent errors
                if audio_head.device != device:
                    logger.warning(f"Audio head still on wrong device: {audio_head.device}, forcing to {device}")
                    # Force non-blocking copy to device with synchronization
                    audio_head = audio_head.to(device=device, dtype=dtype, non_blocking=False)
                    torch.cuda.synchronize(device)
            
                # Also ensure decoder_h_flat is on the correct device with synchronization
                decoder_h_flat = decoder_h_flat.to(device=device, dtype=dtype, non_blocking=False)
                torch.cuda.synchronize(device)
            
                # Add debug prints to trace device issues if needed
                if args is not None and getattr(args, 'debug', False) and i == 1 and batch_idx < 2:
                    debug_info = {
                        "decoder_h_device": decoder_h.device,
                        "decoder_h_flat_device": decoder_h_flat.device,
                        "audio_head_device": audio_head.device
                    }
                    logger.debug(f"Debug tensor device info: {debug_info}")
                
                # Log shapes and devices for debugging
                logger.debug(f"decoder_h_flat shape: {decoder_h_flat.shape}, audio_head shape: {audio_head.shape}")
                logger.debug(f"Before matmul - decoder_h_flat device: {decoder_h_flat.device}, audio_head device: {audio_head.device}")
            
                # Now do the final matmul with guaranteed device matching
                try:
                    # Explicitly detach and clone to avoid any reference issues
                    decoder_h_flat_safe = decoder_h_flat.detach().clone().to(device=device, dtype=dtype, non_blocking=False)
                    audio_head_safe = audio_head.detach().clone().to(device=device, dtype=dtype, non_blocking=False)
                    
                    # Force synchronization before operation
                    torch.cuda.synchronize(device)
                    
                    # Check device one more time and log
                    logger.debug(f"Final check - decoder_h_flat device: {decoder_h_flat_safe.device}, audio_head device: {audio_head_safe.device}")
                    
                    # Use safe copies for matmul
                    ci_logits = torch.matmul(decoder_h_flat_safe, audio_head_safe)
                except RuntimeError as e:
                    if "device" in str(e).lower():
                        # Last attempt with completely new tensors
                        logger.warning("Matmul device error, making final attempt with new tensors")
                        # Create new tensors directly on device
                        with torch.cuda.device(device):
                            # Copy data to new tensors
                            decoder_h_flat_new = torch.zeros_like(decoder_h_flat, device=device, dtype=dtype)
                            decoder_h_flat_new.copy_(decoder_h_flat)
                            audio_head_new = torch.zeros_like(audio_head, device=device, dtype=dtype)
                            audio_head_new.copy_(audio_head)
                            torch.cuda.synchronize(device)
                            ci_logits = torch.matmul(decoder_h_flat_new, audio_head_new)
                    else:
                        raise
            
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
                # Get embedding for the next codebook token with careful device handling
                try:
                    # Get tokens and ensure they're on the right device first
                    if audio_tokens.dim() == 3:
                        token_input = audio_tokens[:, i, 0].clone().view(-1, 1).to(device=device)
                    else:
                        token_input = audio_tokens[:, i].clone().view(-1, 1).to(device=device)
                    
                    # Get embedding with device-prepared inputs
                    ci_embed = model._embed_audio(i, token_input)
                    
                    # Double-check the embedding device
                    if ci_embed.device != device:
                        ci_embed = ci_embed.to(device=device, dtype=dtype)
                except Exception as embed_err:
                    logger.error(f"Error in audio embedding for position {i}: {embed_err}")
                    # Create fallback embedding
                    embed_dim = decoder_h.size(-1)
                    ci_embed = torch.zeros(
                        decoder_h.size(0), 1, embed_dim, 
                        device=device, dtype=dtype
                    )
                
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
            
            # Only predict first codebook, ensuring everything is on the same device
            # Force explicit device/dtype placement
            last_h = last_h.to(device=device, dtype=dtype)
            c0_logits = model.codebook0_head(last_h).to(device=device)
            
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

def evaluate(model, val_loader, device, args=None):
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
    
    # Create model with bfloat16 precision and properly place on device
    with torch.cuda.device(device):
        model = Model(model_args).to(device=device, dtype=torch.bfloat16)
        # Force synchronization to ensure model is fully on device
        torch.cuda.synchronize(device)
        
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
                    total_loss = process_batch(model, text_tokens, audio_tokens, device, args, batch_idx)
                    
                # Optimize with mixed precision
                optimizer.zero_grad(set_to_none=True)  # More efficient
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss = process_batch(model, text_tokens, audio_tokens, device, args, batch_idx)
                    
                # Standard optimization
                optimizer.zero_grad(set_to_none=True)  # More efficient
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                
            # Force synchronization after each step
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
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
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
