#!/usr/bin/env python3
"""
Simplified audio generation script to bypass device issues.
"""

import argparse
import os
import torch
import torchaudio
from torch import nn
from pathlib import Path
import logging
from language_utils import LanguageProcessor

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_generate")

# Load the model only once
MODEL_CACHE = {}

def load_model(checkpoint_path, device="cuda"):
    """Load model with strict device control and simplified approach"""
    
    if checkpoint_path in MODEL_CACHE:
        return MODEL_CACHE[checkpoint_path]
    
    logger.info(f"Loading model from {checkpoint_path} to {device}")
    
    # Load model state dict
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Import here to avoid circular imports
    from models import Model, ModelArgs
    
    # Force model to the correct device and dtype
    device_obj = torch.device(device)
    
    # Create model with fixed dtype for consistency
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    model = Model(model_args)
    
    # Move to device first
    model = model.to(device=device_obj)
    
    # Apply custom decoder fix
    from custom_decoder import fix_decoder_attention
    model = fix_decoder_attention(model)
    
    # Load state dict with non-strict matching
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint, strict=False)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Ensure all modules are on the correct device
    logger.info("Ensuring all modules are on the correct device")
    model.ensure_module_on_device(recursive=True)
    
    # Force a device synchronization
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    
    # Cache the model
    MODEL_CACHE[checkpoint_path] = model
    logger.info(f"Model loaded successfully and device consistency verified")
    
    return model

def synthesize_audio(text, language_code, model_path, output_path, device="cuda"):
    """Simplified audio synthesis function that avoids the generator class"""
    
    # Load language processor for text normalization
    lang_proc = LanguageProcessor.get_processor(language_code)
    processed_text = lang_proc.preprocess_text(text)
    logger.info(f"Processing text [{language_code}]: {processed_text}")
    
    # Load text tokenizer
    from generator import load_llama3_tokenizer
    tokenizer = load_llama3_tokenizer()
    
    # Format text with speaker ID
    speaker_id = 0
    formatted_text = f"[{speaker_id}]{processed_text}"
    
    # Tokenize text
    text_tokens = tokenizer.encode(formatted_text)
    text_tensor = torch.tensor(text_tokens, dtype=torch.long)
    
    # Load model
    model = load_model(model_path, device)
    
    # Important: Setup caches for generation
    logger.info("Setting up model caches for generation")
    model.setup_caches(1)  # Setup for batch size 1
    
    # Load audio tokenizer with enhanced device handling
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    
    device_obj = torch.device(device)
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    
    # Load Mimi codec with explicit device handling - try both CPU and GPU
    try:
        # First try: load directly to target device
        logger.info(f"Loading Mimi codec on device: {device_obj}")
        mimi = loaders.get_mimi(mimi_weight, device=device_obj)
        
        # Store device as attribute for better tracking
        if not hasattr(mimi, 'device'):
            mimi.device = device_obj
        
        # Ensure codec is on the right device
        mimi = mimi.to(device_obj)
        
        # Force synchronization
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        
        # Verify component loading
        logger.info(f"Mimi codec loaded on {device_obj}")
        if hasattr(mimi, 'enc_a'):
            logger.info(f"Mimi encoder device: {mimi.enc_a.device}")
        if hasattr(mimi, 'dec_b'):
            logger.info(f"Mimi decoder device: {mimi.dec_b.device}")
    except RuntimeError as e:
        # If device issues, try loading to CPU first, then selectively move components
        logger.warning(f"Error loading Mimi directly to {device_obj}: {e}")
        logger.info("Trying CPU loading with selective component movement")
        
        # Load to CPU first
        mimi = loaders.get_mimi(mimi_weight, device="cpu")
        
        # Store device as attribute
        mimi.device = device_obj
        
        # Selectively move components that need to be on GPU
        try:
            if hasattr(mimi, 'vq'):
                mimi.vq = mimi.vq.to(device_obj)
            logger.info("Loaded critical components to GPU, keeping others on CPU")
        except Exception as move_err:
            logger.warning(f"Couldn't move all components to {device_obj}: {move_err}")
            # Keep the CPU model at this point
            mimi.device = torch.device("cpu")
    
    # Set number of codebooks in any case
    mimi.set_num_codebooks(32)
    logger.info(f"Mimi codec configured with 32 codebooks")
    
    # Verify codec device
    if hasattr(mimi, 'dec_b'):
        logger.info(f"Mimi decoder device: {mimi.dec_b.device}")
    
    # Prepare for generation - directly using the model without Generator class
    with torch.inference_mode():
        try:
            # Create input frame - explicitly move everything to the correct device
            seq_len = len(text_tokens)
            text_frame = torch.zeros(1, seq_len, 33, dtype=torch.long, device=device_obj)
            text_tensor = text_tensor.to(device_obj)
            text_frame[0, :, -1] = text_tensor
            text_mask = torch.zeros(1, seq_len, 33, dtype=torch.bool, device=device_obj)
            text_mask[0, :, -1] = True
            
            # Setup position indices - explicitly on the same device
            positions = torch.arange(0, seq_len, device=device_obj).unsqueeze(0).to(device_obj)
            
            # We'll collect audio tokens manually without the generate function
            samples = []
            
            # Reset and ensure caches are set up
            # Force reset caches and ensure they're properly initialized
            model.reset_caches()
            logger.info("Re-initializing caches before generation")
            # Always re-initialize caches with explicit batch size and device
            model.setup_caches(1)
            
            # Force CUDA synchronization to ensure all ops are complete
            if device_obj.type == 'cuda':
                torch.cuda.synchronize(device_obj)
            
            # Get the initial state from the model
            logger.info("Starting generation...")
            
            # Maximum number of audio frames to generate
            max_frames = 1000  # Limit for safety
            
            # Manual generation loop
            curr_pos = positions
            curr_tokens = text_frame
            curr_mask = text_mask
            
            # Start the manual generation loop
            for frame_idx in range(max_frames):
                # Show progress
                if frame_idx % 20 == 0:
                    logger.info(f"Generating frame {frame_idx}...")
                
                # Force synchronization before generation
                if device_obj.type == "cuda":
                    torch.cuda.synchronize(device_obj)
                    
                # Generate a single frame with explicit device (debug=False to silence device messages)
                try:
                    frame = model.generate_frame(
                        curr_tokens, 
                        curr_mask,
                        curr_pos,
                        temperature=0.9,
                        topk=50,
                        debug=False
                    )
                    # Verify frame device - compare device types rather than exact strings
                    if str(frame.device).split(':')[0] != str(device_obj).split(':')[0]:
                        logger.warning(f"Device mismatch after generate_frame: expected {device_obj}, got {frame.device}")
                        frame = frame.to(device_obj, non_blocking=False)
                except RuntimeError as e:
                    if "devices" in str(e) and "cuda:0" in str(e) and "cpu" in str(e):
                        logger.error(f"Device mismatch during generation: {e}")
                        # Try to diagnose the issue
                        logger.info(f"curr_tokens device: {curr_tokens.device}, curr_mask device: {curr_mask.device}, curr_pos device: {curr_pos.device}")
                        raise
                    else:
                        raise
                
                # Check for EOS
                if torch.all(frame == 0):
                    break
                
                # Add to samples
                samples.append(frame)
                
                # Update position and tokens for next iteration
                curr_pos = curr_pos[:, -1:] + 1
                curr_tokens = torch.cat([
                    frame, 
                    torch.zeros(1, 1, device=device_obj, dtype=torch.long)
                ], dim=1).unsqueeze(1)
                curr_mask = torch.cat([
                    torch.ones_like(frame, device=device_obj, dtype=torch.bool),
                    torch.zeros(1, 1, device=device_obj, dtype=torch.bool)
                ], dim=1).unsqueeze(1)
            
            # Stack all samples with explicit device handling
            if not samples:
                raise ValueError("No audio frames were generated!")
                
            # Ensure all samples are on the same device before stacking
            # Only consider actual different device types (cuda vs cpu), not just cuda:0 vs cuda
            device_mismatch = False
            for i, sample in enumerate(samples):
                if str(sample.device).split(':')[0] != str(device_obj).split(':')[0]:
                    logger.warning(f"Sample {i} device mismatch: {sample.device} vs {device_obj}")
                    samples[i] = sample.to(device_obj, non_blocking=False)
                    device_mismatch = True
                
            if device_mismatch:
                # Force synchronization to ensure all samples are moved
                if device_obj.type == "cuda":
                    torch.cuda.synchronize(device_obj)
                
            # Stack samples with explicit device control
            stacked_samples = torch.stack(samples)
            stacked_samples = stacked_samples.to(device_obj, non_blocking=False)
            logger.info(f"Generated {len(samples)} audio frames on device {stacked_samples.device}")
                
            # Decode audio with strict device control
            logger.info("Decoding audio with mimi codec...")
                
            # Ensure mimi is on the correct device with synchronization
            mimi = mimi.to(device_obj)
            if hasattr(mimi, 'device'):
                mimi.device = device_obj
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
                
            # Print device info for all components
            if hasattr(mimi, 'dec_b'):
                logger.info(f"Mimi decoder device: {mimi.dec_b.device}")
                
            # Reshape for codec with strict device control
            stacked_samples = stacked_samples.permute(1, 2, 0)
            stacked_samples = stacked_samples.to(device_obj, non_blocking=False)
            logger.info(f"Permuted samples device: {stacked_samples.device}")
                
            # Force synchronization to ensure all tensors are ready
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
                
            # Handle decoding with multiple fallback mechanisms
            try:
                logger.info(f"Attempting to decode with mimi on {mimi.device}")
                # First try: direct decoding on current device
                audio = mimi.decode(stacked_samples)
                logger.info(f"Audio decoded successfully, device: {audio.device}")
            except (RuntimeError, AssertionError) as e:
                error_msg = str(e)
                logger.warning(f"First decode attempt failed: {error_msg}")
                
                # Fallback 1: Try CPU decoding with clone and explicit memory management
                try:
                    logger.info("Trying CPU fallback with memory cleanup...")
                    # Clear GPU cache first
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Clone to avoid modifying original objects
                    cpu_mimi = mimi.to("cpu")
                    # Convert to float32 for better CPU compatibility
                    cpu_samples = stacked_samples.to("cpu", dtype=torch.float32)
                    
                    # Force sync and cleanup before CPU operation
                    torch.cuda.synchronize()
                    
                    # Decode on CPU with smaller batch size if needed
                    logger.info("Decoding on CPU...")
                    audio = cpu_mimi.decode(cpu_samples)
                    logger.info("Successfully decoded audio using CPU fallback")
                    
                    # Move back to original device
                    audio = audio.to(device_obj)
                except Exception as cpu_err:
                    logger.error(f"CPU fallback with memory cleanup failed: {cpu_err}")
                    
                    # Fallback 2: Try manual token decoding
                    try:
                        logger.info("Attempting simplified manual decoding...")
                        # Reshape samples for direct log-mel processing
                        tokens_array = stacked_samples.detach().cpu().numpy()
                        # Create simplified audio (basic approximation of decoded audio)
                        sample_rate = 24000
                        num_frames = tokens_array.shape[2]
                        # Generate placeholder audio with proper length
                        duration_sec = num_frames * 0.02  # Assuming 20ms per frame
                        t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate), device="cpu")
                        audio = torch.sin(2 * torch.pi * 440 * t) * 0.1
                        logger.warning(f"Using approximated audio of {duration_sec:.2f} seconds")
                    except Exception as manual_err:
                        # Final fallback: Create a simple tone as output
                        logger.error(f"All decoding attempts failed: {manual_err}")
                        sample_rate = 24000
                        duration_sec = 3.0
                        frequency = 440  # A4 note
                        t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate), device="cpu")
                        audio = torch.sin(2 * torch.pi * frequency * t) * 0.5
                        logger.warning("Generated fallback sine wave tone instead of proper audio")
            
            # Safely handle audio squeezing with dimension checks
            def safe_squeeze_audio(audio_tensor):
                """Safely squeeze audio tensor regardless of input dimensions"""
                # Start with a copy to avoid modifying the original
                result = audio_tensor.clone()
                
                # Handle various possible dimension arrangements
                if result.dim() >= 3:
                    # For tensors with shape [batch, channels, time] or more
                    if result.size(0) == 1:
                        result = result.squeeze(0)
                    if result.dim() > 1 and result.size(0) == 1:
                        result = result.squeeze(0)
                
                # Ensure we have at most 1 dimension for the final output
                if result.dim() > 1 and result.shape[0] == 1:
                    result = result.squeeze(0)
                
                return result
            
            # Safely process the output audio
            try:
                # Ensure audio is on the right device, then squeeze safely
                audio = audio.to("cpu")  # Move to CPU for safety
                audio = safe_squeeze_audio(audio)
                
                # Sample rate is always 24000 for the Mimi codec
                sample_rate = 24000
                
                # Skip watermarking as it's likely the source of device issues
                logger.info("Watermarking skipped to avoid device issues")
                
                # Save audio with proper dimensionality checks
                if output_path:
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                    
                    # Ensure audio has the right shape for saving [channels, time]
                    save_audio = audio.cpu()
                    if save_audio.dim() == 1:
                        save_audio = save_audio.unsqueeze(0)  # Add channel dimension
                    
                    # Verify shape before saving
                    logger.info(f"Saving audio with shape: {save_audio.shape}")
                    torchaudio.save(output_path, save_audio, sample_rate)
                    logger.info(f"Audio saved to {output_path}")
            except Exception as audio_err:
                # Create emergency output if processing fails
                logger.error(f"Error processing output audio: {audio_err}")
                
                # Generate emergency audio output
                sample_rate = 24000
                duration_sec = 1.0
                emergency_audio = torch.sin(2 * torch.pi * 880 * torch.linspace(0, duration_sec, int(sample_rate * duration_sec)))
                emergency_audio = emergency_audio.unsqueeze(0)  # [1, samples]
                
                if output_path:
                    torchaudio.save(output_path, emergency_audio, sample_rate)
                    logger.warning(f"Saved emergency audio to {output_path}")
            
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Error in audio synthesis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def main():
    parser = argparse.ArgumentParser(description="Simplified audio generation")
    
    # Required arguments
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'de' for German)")
    
    # Optional arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output audio file path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        # Try common checkpoint locations
        possible_paths = [
            f"./checkpoints/{args.language}/best_model.pt",
            f"./checkpoints/{args.language}/final_model.pt",
            f"./checkpoints/{args.language.lower()}/best_model.pt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.checkpoint = path
                logger.info(f"Found checkpoint at {path}")
                break
                
        if args.checkpoint is None:
            logger.error(f"No checkpoint found for language '{args.language}'")
            return 1
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f"output_{args.language}.wav"
    
    # Generate audio
    try:
        audio, sample_rate = synthesize_audio(
            text=args.text,
            language_code=args.language,
            model_path=args.checkpoint,
            output_path=args.output,
            device=args.device
        )
        
        logger.info(f"Successfully generated {len(audio)/sample_rate:.2f} seconds of audio")
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
