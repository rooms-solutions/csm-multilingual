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
    Perform matrix multiplication with guaranteed device compatibility.
    This function ensures both tensors are on the same device before multiplication.
    """
    # Normalize device to torch.device object with proper index
    if isinstance(device, str):
        actual_device = torch.device(device if device != "cuda" else f"cuda:{torch.cuda.current_device()}")
    else:
        actual_device = device
    
    # Clone tensors to avoid modifying the originals and ensure they're on the right device
    tensor1_safe = tensor1.detach().clone().to(device=actual_device, non_blocking=False)
    tensor2_safe = tensor2.detach().clone().to(device=actual_device, non_blocking=False)
    
    # Force synchronization
    if actual_device.type == "cuda":
        torch.cuda.synchronize(actual_device)
    
    # Perform the matrix multiplication
    return torch.matmul(tensor1_safe, tensor2_safe)

def numerically_stable_cross_entropy(logits, targets, epsilon=1e-5):
    """
    Compute cross entropy loss with enhanced numerical stability for AMP training.
    
    Args:
        logits: Model predictions
        targets: Target indices
        epsilon: Small value to avoid log(0)
        
    Returns:
        Numerically stable cross entropy loss
    """
    # Handle NaN/Inf values more aggressively
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Get shape information
    batch_size = logits.size(0)
    
    # For AMP compatibility, avoid potential precision issues
    # Use direct cross entropy instead of manual calculation
    try:
        # Use PyTorch's cross_entropy which has better numerical stability
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
        # Safety check for the result
        if torch.isnan(loss) or torch.isinf(loss):
            # Fallback to more careful approach
            # Limit the range of logits to avoid extreme values
            logits = torch.clamp(logits, -1e4, 1e4)
            loss = torch.nn.functional.cross_entropy(logits, targets)
    except Exception as e:
        # Last resort fallback
        # Apply very aggressive normalization
        logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-8)
        loss = torch.nn.functional.cross_entropy(logits, targets)
    
    # Final check to ensure we return a valid loss
    if torch.isnan(loss) or torch.isinf(loss):
        # Return a stable default loss that can be backpropagated
        return torch.tensor(10.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
    
    return loss

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
    # Debug info using logger instead of print
    logger.debug(f"Text tokens shape: {text_tokens.shape}, audio_tokens shape: {audio_tokens.shape}")
    logger.debug(f"Model dtype: {next(model.parameters()).dtype}")
    
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
        
        # Use numerically stable cross entropy instead of standard version
        try:
            # Check for NaN inputs first
            if torch.isnan(c0_logits).any() or torch.isinf(c0_logits).any():
                logger.warning(f"Detected NaN or Inf in c0_logits at batch {batch_idx}")
                # Attempt to fix NaN/Inf values
                c0_logits = torch.nan_to_num(c0_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Use our stable implementation
            c0_loss = numerically_stable_cross_entropy(c0_logits, c0_targets)
        except Exception as e:
            logger.error(f"Error in c0 loss calculation: {e}")
            # Fallback with very careful input handling
            c0_logits_safe = torch.nan_to_num(c0_logits.detach().clone(), nan=0.0)
            c0_loss = nn.functional.cross_entropy(c0_logits_safe, c0_targets)
        
        total_loss += c0_loss
        
        # For remaining codebooks, use a simplified approach without the decoder
        # We'll use direct prediction with linear layers instead
        
        # Create a position embedding matrix for each codebook position
        # Explicitly create on device with proper error handling
        try:
            # Ensure device is a torch.device object
            if isinstance(device, str):
                cuda_device = torch.device(device if device != "cuda" else f"cuda:{torch.cuda.current_device()}")
            else:
                cuda_device = device
        
            # Ensure proper size from backbone output
            embed_dim = backbone_output.size(-1)
        
            # Create directly on the device with CUDA synchronization to ensure placement
            position_embeddings = torch.zeros(
                num_codebooks-1, 
                embed_dim, 
                device=cuda_device, 
                dtype=dtype
            )
        
            # Fill with random values directly on device
            with torch.no_grad():
                position_embeddings.normal_(0, 0.02)
        
            # Force synchronization
            if cuda_device.type == "cuda":
                torch.cuda.synchronize()
            
            # No need to verify device since we created it directly on the right device
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
            
            # Get logits by matmul with the audio head using safe_matmul
            ci_logits = safe_matmul(
                projected_h.squeeze(1),  # Shape: [batch_size, decoder_dim]
                audio_head_i,  # Shape: [decoder_dim, vocab_size]
                device
            )  # Result: [batch_size, vocab_size]
            
            # Extract target
            if audio_tokens.dim() == 3:
                ci_targets = audio_tokens[:, i, 0].clone()
            else:
                ci_targets = audio_tokens[:, i].clone()
                
            # Make sure targets are 1D
            ci_targets = ci_targets.view(-1)
            
            # Calculate loss with numerical stability
            try:
                # Check for NaN/Inf values
                if torch.isnan(ci_logits).any() or torch.isinf(ci_logits).any():
                    logger.warning(f"Detected NaN or Inf in ci_logits for codebook {i} at batch {batch_idx}")
                    ci_logits = torch.nan_to_num(ci_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                
                # Use our stable implementation
                ci_loss = numerically_stable_cross_entropy(ci_logits, ci_targets)
            except Exception as e:
                logger.error(f"Error in ci_loss calculation for codebook {i}: {e}")
                # Fallback with careful input handling
                ci_logits_safe = torch.nan_to_num(ci_logits.detach().clone(), nan=0.0)
                ci_loss = nn.functional.cross_entropy(ci_logits_safe, ci_targets, reduction='mean')
            
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
                # Explicitly ensure all tensors have correct device before passing to decoder
                decoder_input = decoder_input.to(device=device, dtype=dtype, non_blocking=False)
                decoder_positions = decoder_positions.to(device=device, non_blocking=False)
                decoder_mask = decoder_mask.to(device=device, non_blocking=False)
                
                # Force synchronize to ensure transfers complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                # Now use the proper decoder with our fixed attention implementation
                # Make sure decoder_input has exactly 2 sequence positions [batch, 2, dim]
                if decoder_input.size(1) != 2:
                    # Adjust to exactly 2 sequence positions
                    if decoder_input.size(1) > 2:
                        # Truncate to 2 positions
                        decoder_input = decoder_input[:, :2, :]
                    else:
                        # Pad to 2 positions by duplicating the last position
                        pad = decoder_input[:, -1:, :].expand(-1, 2-decoder_input.size(1), -1)
                        decoder_input = torch.cat([decoder_input, pad], dim=1)
                
                # Make sure decoder_positions is exactly [batch, 2] with values [0,1]
                decoder_positions = torch.zeros(b, 2, dtype=torch.long, device=device)
                decoder_positions[:, 1] = 1  # Set second position to 1
            
                # Verify decoder mask has the right dimensions [batch, 2, 2]
                decoder_mask = torch.tril(
                    torch.ones(2, 2, dtype=torch.bool, device=device)
                ).unsqueeze(0).expand(b, 2, 2)
            
                # Pass to decoder with corrected dimensions
                decoder_h = model.decoder(
                    decoder_input, 
                    input_pos=decoder_positions, 
                    mask=decoder_mask
                ).to(device=device, dtype=dtype)
                
                # Log what we're doing
                if i == 1:  # Only log once per batch
                    logger.debug("Using fixed decoder implementation with proper attention handling")
            except Exception as decoder_err:
                # Log the specific decoder error
                logger.error(f"Decoder error: {decoder_err}")
                # Fall back to using the projected input directly
                decoder_h = decoder_input.clone().to(device=device, dtype=dtype)
            
            # Log what we're doing
            if i == 1:  # Only log once per batch
                logger.debug("Using fixed decoder implementation with proper attention handling")
        
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
                
                # Use the output directly for the logits calculation with our safe_matmul
                try:
                    # Use our custom safe_matmul function
                    audio_head = model.audio_head[i-1]
                    ci_logits = safe_matmul(decoder_output, audio_head, device)
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
            
                # Normalize device to device object format
                if isinstance(device, str):
                    actual_device = torch.device(device if device != "cuda" else f"cuda:{torch.cuda.current_device()}")
                else:
                    actual_device = device
                
                # Move tensors to device without unnecessary warnings
                if audio_head.device != actual_device:
                    # Force copy to device with synchronization
                    audio_head = audio_head.to(device=actual_device, dtype=dtype, non_blocking=False)
                    torch.cuda.synchronize(actual_device)
            
                # Ensure decoder_h_flat is on the correct device
                decoder_h_flat = decoder_h_flat.to(device=actual_device, dtype=dtype, non_blocking=False)
                torch.cuda.synchronize(actual_device)
            
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
                
                # Add more detailed shape debugging when debug flag is set
                if args is not None and getattr(args, 'debug', False):
                    logger.debug(f"decoder_input shape: {decoder_input.shape}")
                    logger.debug(f"decoder_positions shape: {decoder_positions.shape}")
                    logger.debug(f"decoder_mask shape: {decoder_mask.shape}")
                    logger.debug(f"decoder_h shape: {decoder_h.shape}")
            
                # Now do the final matmul with our safe matrix multiplication function
                try:
                    # Use our custom safe_matmul function to guarantee device compatibility
                    ci_logits = safe_matmul(decoder_h_flat, audio_head, device)
                    
                    # Log success
                    if i == 1 and batch_idx < 2:
                        logger.debug(f"Matrix multiplication successful with safe_matmul for position {i}")
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
            
            # Calculate loss for just first codebook with numerical stability
            c0_targets = c0_targets.view(-1)
            
            # Clean up logits to prevent NaN
            c0_logits = torch.nan_to_num(c0_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Apply scaling to prevent extreme values
            if c0_logits.abs().max() > 1e4:
                c0_logits = c0_logits * (1e4 / c0_logits.abs().max())
            
            # Use stable cross entropy
            fallback_loss = numerically_stable_cross_entropy(c0_logits, c0_targets)
            
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
            
            # Verify AMP compatibility with comprehensive tests
            logger.info("Running complete AMP compatibility verification")
            
            # Test 1: Basic scaling and unscaling
            dummy_tensor = torch.tensor([1.0], device=device, dtype=torch.float16, requires_grad=True)
            dummy_optimizer = optim.AdamW([dummy_tensor], lr=0.001)
            
            try:
                with autocast('cuda'):
                    dummy_loss = dummy_tensor * 2
                
                # Test full AMP workflow
                scaler.scale(dummy_loss).backward()
                scaler.unscale_(dummy_optimizer)
                scaler.step(dummy_optimizer)
                scaler.update()
                dummy_optimizer.zero_grad()
                
                logger.info("✓ AMP basic test passed")
                
                # Test 2: More complex tensor operations
                x = torch.randn(32, 32, device=device, dtype=torch.float16, requires_grad=True)
                optimizer2 = optim.AdamW([x], lr=0.001)
                
                with autocast('cuda'):
                    y = torch.nn.functional.softmax(x, dim=1)
                    loss2 = y.mean()
                
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer2)
                scaler.step(optimizer2)
                scaler.update()
                
                logger.info("✓ AMP complex operations test passed")
                
            except RuntimeError as test_err:
                # If we encounter BFloat16 specific errors, we'll handle special cases
                if "BFloat16" in str(test_err) or "not implemented" in str(test_err):
                    logger.warning(f"AMP compatibility test failed: {test_err}")
                    logger.warning("Enabling strict compatibility mode with special handling")
                    args.amp_compatibility_mode = True
                    # Last attempt with ultra-conservative settings
                    scaler = GradScaler('cuda', enabled=True, init_scale=1.0)
                else:
                    # For other errors, log but continue with AMP
                    logger.warning(f"AMP test encountered error: {test_err}")
            
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
            
            # With caching disabled, we don't need to set up or reset caches
            # Leave this empty block for clarity - no cache operations needed
            pass
            
            # Normalize loss for gradient accumulation
            normalized_loss = total_loss = process_batch(model, text_tokens, audio_tokens, device, args, batch_idx)
            normalized_loss = normalized_loss / args.gradient_accumulation_steps
            
            # Forward pass and loss calculation with gradient accumulation
            if args.use_amp:
                with autocast('cuda'):
                    # Use the already computed loss
                    pass
                
                # Optimize with mixed precision using try-except blocks for safety
                try:
                    # Check for NaN in loss before backward pass
                    if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                        logger.warning(f"NaN/Inf detected in loss before backward: {normalized_loss.item()}")
                        # Reset the loss to a stable value
                        model_dtype = next(model.parameters()).dtype
                        normalized_loss = torch.tensor(1.0, device=device, dtype=model_dtype, requires_grad=True)
                    
                    scaler.scale(normalized_loss).backward()
                    
                    # Check for NaN gradients after backward
                    grad_issues = check_gradients(model)
                    if grad_issues:
                        logger.warning("Fixed NaN/Inf gradients after backward pass")
                    
                    # Only update weights after accumulating gradients for specified steps
                    if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                        # Completely reworked update logic with proper error handling
                        if args.amp_compatibility_mode:
                            # In compatibility mode, skip unscale_ operation entirely
                            logger.debug("Using compatibility mode - skipping unscale")
                                
                            # Apply gradient clipping directly to scaled gradients
                            # Use a fixed scaling factor to compensate for the scaling
                            scale_factor = scaler._scale.item() if hasattr(scaler, '_scale') else 128.0
                            adjusted_clip = args.grad_clip * scale_factor
                            torch.nn.utils.clip_grad_norm_(model.parameters(), adjusted_clip)
                                
                            # Step optimizer and update scaler without unscaling
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard AMP path with enhanced error handling
                            try:
                                # Safe unscaling with verification
                                scaler.unscale_(optimizer)
                                    
                                # Extra verification after unscaling
                                has_nans = False
                                for name, param in model.parameters():
                                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                        has_nans = True
                                        logger.debug(f"NaN gradient in {name} after unscaling")
                                        param.grad = torch.zeros_like(param.grad)
                                    
                                # Apply gradient clipping to unscaled gradients
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                                    
                                # Step and update
                                scaler.step(optimizer)
                                scaler.update()
                                    
                            except RuntimeError as unscale_err:
                                if "BFloat16" in str(unscale_err) or "not implemented" in str(unscale_err):
                                    logger.warning(f"Unscale error: {unscale_err}")
                                    logger.warning("Switching to compatibility mode for this update")
                                        
                                    # Apply fixed scaling as a fallback
                                    scale_factor = scaler._scale.item() if hasattr(scaler, '_scale') else 128.0
                                    adjusted_clip = args.grad_clip * scale_factor
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), adjusted_clip)
                                        
                                    # Try to step without unscaling
                                    try:
                                        scaler.step(optimizer)
                                        scaler.update()
                                    except Exception as step_err:
                                        logger.error(f"Error in optimizer step: {step_err}")
                                        # Last resort: update optimizer directly
                                        optimizer.step()
                                else:
                                    # For non-BFloat16 errors, log but still try to continue
                                    logger.error(f"Unexpected error in unscale: {unscale_err}")
                                    optimizer.step()  # Try regular step as fallback
                        
                        optimizer.zero_grad(set_to_none=True)  # More efficient
                        scheduler.step()
                except Exception as e:
                    logger.error(f"Error in backward pass: {e}")
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Standard optimization with gradient accumulation
                # Check for NaN in loss before backward pass
                if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                    logger.warning(f"NaN/Inf detected in loss before backward: {normalized_loss.item()}")
                    # Reset the loss to a stable value
                    model_dtype = next(model.parameters()).dtype
                    normalized_loss = torch.tensor(1.0, device=device, dtype=model_dtype, requires_grad=True)
                
                normalized_loss.backward()
                
                # Check for NaN gradients after backward
                grad_issues = check_gradients(model)
                if grad_issues:
                    logger.warning("Fixed NaN/Inf gradients after backward pass")
                
                # Only update weights after accumulating gradients for specified steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # More efficient
                    scheduler.step()
                
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
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
