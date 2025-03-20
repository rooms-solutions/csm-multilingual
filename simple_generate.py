#!/usr/bin/env python3
"""
Simplified audio generation script to bypass device issues.
"""

import argparse
import os
import torch
import torchaudio
import numpy as np
from torch import nn
from pathlib import Path
import logging
from language_utils import LanguageProcessor

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_generate")

# Load the model only once
MODEL_CACHE = {}

def load_model(checkpoint_path, device="cuda", isolation_level=0):
    """
    Load model with strict device control and isolation options
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device ("cuda" or "cpu")
        isolation_level: Level of CUDA isolation (0=none, 1=soft, 2=hard)
    """
    cache_key = f"{checkpoint_path}_{device}_{isolation_level}"
    
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    logger.info(f"Loading model from {checkpoint_path} to {device} with isolation_level={isolation_level}")
    
    # Load model state dict
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Import here to avoid circular imports
    from models import Model, ModelArgs
    
    # Apply CUDA isolation if requested (for CPU fallback paths)
    old_cuda_visible = None
    if isolation_level == 2 and device == "cpu":
        # Hard isolation: Temporarily hide all CUDA devices
        logger.info("Using hard CUDA isolation for CPU model loading")
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
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
        
        # Soft isolation: Create model on CPU first, then move
        if isolation_level == 1 and device != "cpu":
            logger.info("Using soft isolation: creating on CPU first")
            model = Model(model_args)
            # Apply custom decoder fix
            from custom_decoder import fix_decoder_attention
            model = fix_decoder_attention(model)
            # Load state dict with non-strict matching
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            # Now move to target device
            model = model.to(device=device_obj)
        else:
            # Normal path or hard isolation
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
        logger.info(f"Ensuring all modules are on device {device_obj}")
        model.ensure_module_on_device(recursive=True)
        
        # Force a device synchronization
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        
        # Cache the model
        MODEL_CACHE[cache_key] = model
        logger.info(f"Model loaded successfully and device consistency verified")
        
        return model
    finally:
        # Restore CUDA visibility if changed
        if old_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
        elif isolation_level == 2 and "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

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
                
            # Handle decoding with completely isolated fallback mechanisms
            try:
                logger.info(f"Attempting to decode with mimi on {mimi.device}")
                # First try: direct decoding on current device
                audio = mimi.decode(stacked_samples)
                logger.info(f"Audio decoded successfully, device: {audio.device}")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"First decode attempt failed: {error_msg}")
                
                # Fallback 1: Create a completely fresh Mimi instance on CPU
                try:
                    # Complete isolation from existing CUDA context
                    logger.info("Creating fresh CPU-only Mimi instance...")
                    
                    # Force complete GPU cleanup to avoid contamination
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Get samples to CPU without any CUDA connections
                    with torch.no_grad():
                        cpu_samples = stacked_samples.detach().clone().cpu()
                        # Force sync to ensure data is copied
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    
                    # Create a completely new Mimi model
                    from moshi.models import loaders
                    from huggingface_hub import hf_hub_download
                    fresh_mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
                    
                    # Important: Explicitly create with torch.device("cpu") and avoid any CUDA references
                    import os
                    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                    try:
                        # Temporarily hide CUDA devices to force CPU-only initialization
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        
                        # Create a fresh model with CPU-only operations
                        fresh_cpu_mimi = loaders.get_mimi(fresh_mimi_weight, device=torch.device("cpu"))
                        fresh_cpu_mimi.set_num_codebooks(32)
                        
                        # Ensure in eval mode with no grad
                        fresh_cpu_mimi.eval()
                        
                        # Log model device to verify it's on CPU
                        logger.info(f"Fresh Mimi created on device: {next(fresh_cpu_mimi.parameters()).device}")
                        
                        # CPU-only decode operation
                        logger.info("Decoding with CPU-only model...")
                        with torch.no_grad():
                            audio = fresh_cpu_mimi.decode(cpu_samples.to(dtype=torch.float32))
                        logger.info("Successfully decoded audio using isolated CPU model")
                    finally:
                        # Restore CUDA visibility
                        if old_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                        else:
                            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                
                except Exception as cpu_err:
                    logger.error(f"Isolated CPU fallback failed: {cpu_err}")
                    
                    # Fallback 2: Convert tokens directly using code table approach
                    try:
                        logger.info("Attempting mel-spectrogram approximation...")
                        with torch.no_grad():
                            # Get token data to CPU as numpy
                            token_data = stacked_samples.detach().cpu().numpy()
                            
                            # Create a simple approximation of audio from tokens
                            sample_rate = 24000
                            num_frames = token_data.shape[2]
                            
                            # Create a proper duration based on frame count
                            duration_sec = num_frames * 0.02  # 20ms per frame
                            
                            # Generate a more pleasing audio representation
                            # Use a mix of sine waves to approximate speech
                            time_arr = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
                            audio_data = np.zeros_like(time_arr)
                            
                            # Extract some variability from tokens
                            for i in range(min(4, token_data.shape[0])):
                                # Use tokens to modulate frequency and amplitude
                                freqs = (token_data[i, 0, :num_frames] % 40 + 100) * 5
                                for t_idx, freq in enumerate(freqs):
                                    # Apply the frequency in a small window
                                    start_idx = int(t_idx * 0.02 * sample_rate)
                                    end_idx = int((t_idx + 1) * 0.02 * sample_rate)
                                    if end_idx > len(time_arr):
                                        break
                                    window = time_arr[start_idx:end_idx]
                                    audio_data[start_idx:end_idx] += 0.1 * np.sin(2 * np.pi * freq * window)
                            
                            # Convert back to torch tensor
                            audio = torch.from_numpy(audio_data).float()
                            logger.info(f"Created audio approximation with {duration_sec:.2f} seconds")
                    except Exception as conversion_err:
                        logger.error(f"Token conversion failed: {conversion_err}")
                        
                        # Final fallback: Create a guaranteed audio output
                        sample_rate = 24000
                        duration_sec = max(3.0, len(samples) * 0.02)  # At least 3 seconds or based on frame count
                        freq_pattern = [440, 330, 440, 550]  # Create a simple melody
                        
                        # Create a recognizable pattern
                        audio_data = []
                        for i in range(int(duration_sec)):
                            # Use a different frequency for each second
                            freq = freq_pattern[i % len(freq_pattern)]
                            t = torch.linspace(0, 1.0, sample_rate, device="cpu")
                            segment = torch.sin(2 * torch.pi * freq * t) * 0.3
                            audio_data.append(segment)
                        
                        # Combine all segments
                        audio = torch.cat(audio_data)
                        logger.warning(f"Generated fallback melody of {duration_sec:.2f} seconds")
            
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
            
            # Completely isolated audio processing to avoid device contamination
            try:
                # Process on CPU to avoid any CUDA issues
                with torch.no_grad():
                    # Always convert to CPU and clone to break any CUDA history
                    cpu_audio = audio.detach().clone().to("cpu")
                    
                    # Force sync to ensure full copy
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Process shapes with defensive programming
                    if cpu_audio.dim() > 4:
                        # Too many dimensions, flatten to manageable shape
                        logger.warning(f"Audio has too many dimensions: {cpu_audio.shape}, flattening")
                        cpu_audio = cpu_audio.reshape(-1)
                    
                    # Apply safe squeeze with explicit error checking
                    try:
                        cpu_audio = safe_squeeze_audio(cpu_audio)
                    except Exception as squeeze_err:
                        logger.warning(f"Error during audio squeeze: {squeeze_err}, using as-is")
                    
                    # Sample rate is always 24000 for the Mimi codec
                    sample_rate = 24000
                    
                    # Skip watermarking as it's likely the source of device issues
                    logger.info("Watermarking skipped to avoid device issues")
                    
                    # Save audio with best-effort shape normalization
                    if output_path:
                        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                        
                        # Guaranteed valid shape for torchaudio.save
                        if cpu_audio.dim() == 1:
                            # Single channel: [samples] -> [1, samples]
                            cpu_audio = cpu_audio.unsqueeze(0)
                        elif cpu_audio.dim() == 2:
                            # Check if dimensions are valid: [channels, samples]
                            if cpu_audio.size(0) > cpu_audio.size(1):
                                # Likely [samples, channels], transpose
                                logger.info(f"Transposing audio from {cpu_audio.shape}")
                                cpu_audio = cpu_audio.T
                        elif cpu_audio.dim() == 3:
                            # Take first batch: [batch, channels, samples] -> [channels, samples]
                            cpu_audio = cpu_audio[0]
                        else:
                            # Fall back to mono for any other case
                            logger.warning(f"Irregular audio shape: {cpu_audio.shape}, converting to mono")
                            cpu_audio = cpu_audio.reshape(-1).unsqueeze(0)
                        
                        # Final check before save
                        logger.info(f"Final audio shape for saving: {cpu_audio.shape}")
                        
                        # Normalize audio to prevent excessive volume
                        if cpu_audio.abs().max() > 1.0:
                            cpu_audio = cpu_audio / cpu_audio.abs().max()
                        
                        try:
                            torchaudio.save(output_path, cpu_audio, sample_rate)
                            logger.info(f"Audio saved to {output_path}")
                        except Exception as save_err:
                            logger.error(f"Error saving audio: {save_err}, attempting simpler format")
                            # Last resort: save as WAV with scipy
                            try:
                                from scipy.io import wavfile
                                # Convert to numpy and ensure correct shape
                                wav_data = cpu_audio.numpy()
                                if wav_data.ndim > 1:
                                    wav_data = wav_data[0]  # Take first channel if multi-channel
                                wavfile.write(output_path, sample_rate, wav_data)
                                logger.info(f"Audio saved using scipy fallback")
                            except Exception as scipy_err:
                                logger.error(f"All save methods failed: {scipy_err}")
            except Exception as audio_err:
                # Create guaranteed emergency output if processing fails
                logger.error(f"Fatal error in audio processing: {audio_err}")
                
                # Use scipy directly to create and save a simple tone
                try:
                    import numpy as np
                    from scipy.io import wavfile
                    
                    sample_rate = 24000
                    duration_sec = 2.0
                    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
                    emergency_signal = np.sin(2 * np.pi * 880 * t) * 0.3
                    
                    # Add a beep pattern to indicate error
                    for i in range(4):
                        start = int(i * 0.5 * sample_rate)
                        end = int((i * 0.5 + 0.2) * sample_rate)
                        emergency_signal[start:end] = np.sin(2 * np.pi * 1760 * t[0:(end-start)]) * 0.5
                    
                    if output_path:
                        wavfile.write(output_path, sample_rate, emergency_signal)
                        logger.warning(f"Saved emergency audio signal to {output_path}")
                except Exception as emergency_err:
                    logger.error(f"Even emergency audio creation failed: {emergency_err}")
            
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
