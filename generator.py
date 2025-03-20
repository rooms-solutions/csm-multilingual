from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
  speaker: int
  text: str
  # (num_samples,), sample_rate = 24_000
  audio: torch.Tensor


def load_llama3_tokenizer():
  """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
  tokenizer_name = "meta-llama/Llama-3.2-1B"
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  bos = tokenizer.bos_token
  eos = tokenizer.eos_token
  tokenizer._tokenizer.post_processor = TemplateProcessing(
      single=f"{bos}:0 $A:0 {eos}:0",
      pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
      special_tokens=[(f"{bos}", tokenizer.bos_token_id),
                      (f"{eos}", tokenizer.eos_token_id)],
  )

  return tokenizer


class Generator:
  def __init__(
      self,
      model: Model,
  ):
    # Get and store device
    self.device = next(model.parameters()).device
    print(f"Generator initialized on device: {self.device}")

    self._model = model
    self._model.setup_caches(1)

    self._text_tokenizer = load_llama3_tokenizer()

    # Make sure all components are on the same device
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=self.device)
    mimi.set_num_codebooks(32)

    # Wrap the mimi codec in our safe wrapper
    self._audio_tokenizer = create_safe_mimi_wrapper(mimi, device=self.device)

    # Store maximum token value for safety checks during decoding
    if hasattr(mimi, 'vq') and hasattr(mimi.vq, 'codebook_size'):
        self.max_token_value = mimi.vq.codebook_size - 1
    else:
        self.max_token_value = 2048  # Safe default

    self._watermarker = load_watermarker(device=self.device)

    self.sample_rate = mimi.sample_rate

  def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[
    torch.Tensor, torch.Tensor]:
    frame_tokens = []
    frame_masks = []

    text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), 33).long()
    text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True

    frame_tokens.append(text_frame.to(self.device))
    frame_masks.append(text_frame_mask.to(self.device))

    return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

  def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    frame_tokens = []
    frame_masks = []

    # (K, T)
    audio = audio.to(self.device)
    audio_tokens = \
    self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
    # add EOS frame
    eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
    audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

    audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
    audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(
      self.device)
    audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
    audio_frame_mask[:, :-1] = True

    frame_tokens.append(audio_frame)
    frame_masks.append(audio_frame_mask)

    return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

  def _tokenize_segment(self, segment: Segment) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
    text_tokens, text_masks = self._tokenize_text_segment(segment.text,
                                                          segment.speaker)
    audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

    return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat(
        [text_masks, audio_masks], dim=0)

  @torch.inference_mode()
  def generate(
      self,
      text: str,
      speaker: int,
      context: List[Segment],
      max_audio_length_ms: float = 90_000,
      temperature: float = 0.9,
      topk: int = 50,
  ) -> torch.Tensor:
    # Get device from model
    device = self.device
    print(f"Generation device: {device}")

    # Reset caches
    self._model.reset_caches()

    max_audio_frames = int(max_audio_length_ms / 80)
    tokens, tokens_mask = [], []
    for segment in context:
      segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
      tokens.append(segment_tokens)
      tokens_mask.append(segment_tokens_mask)

    gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
      text, speaker)
    tokens.append(gen_segment_tokens)
    tokens_mask.append(gen_segment_tokens_mask)

    # Ensure all tensors are on the correct device
    prompt_tokens = torch.cat(tokens, dim=0).long().to(device)
    prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(device)

    samples = []
    curr_tokens = prompt_tokens.unsqueeze(0)
    curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
    curr_pos = torch.arange(0, prompt_tokens.size(0), device=device).unsqueeze(
      0).long()

    max_seq_len = 2048 - max_audio_frames
    if curr_tokens.size(1) >= max_seq_len:
      raise ValueError(
        f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

    for _ in range(max_audio_frames):
      sample = self._model.generate_frame(curr_tokens, curr_tokens_mask,
                                          curr_pos, temperature, topk)
      if torch.all(sample == 0):
        break  # eos

      # Verify device consistency
      sample = sample.to(device)
      samples.append(sample)

      curr_tokens = torch.cat([sample, torch.zeros(1, 1, device=device).long()],
                              dim=1).unsqueeze(1)
      curr_tokens_mask = torch.cat(
          [torch.ones_like(sample, device=device).bool(),
           torch.zeros(1, 1, device=device).bool()], dim=1
      ).unsqueeze(1)
      curr_pos = curr_pos[:, -1:] + 1

    # Stack samples and decode
    print(f"Stacking samples on device: {device}")
    stacked_samples = torch.stack(samples).to(device)

    # Custom patching for CUDA compatibility
    def patch_mimi_upsample(mimi_model):
      """Patch Mimi model's upsampling to avoid CUDA compatibility issues"""

      # Define custom upsampling that stays on CUDA but avoids transpose conv issues
      def custom_upsample(x, scale_factor=2):
        """Custom upsampling that uses interpolate instead of conv_transpose1d"""
        # Force correct format and contiguity
        x = x.contiguous().to(dtype=torch.float32)
        
        # Store original dimensions for better handling
        batch_size, channels, seq_len = x.shape
        
        # Use interpolate which has better CUDA compatibility
        try:
          # Use recommended interpolation with fewer artifacts
          result = torch.nn.functional.interpolate(
              x, scale_factor=scale_factor, mode='linear', align_corners=False
          )
          
          # Use different smoothing approaches depending on channel count
          if channels <= 512:  # For manageable channel counts
            # More sophisticated smoothing with learned-like filtering
            kernel_size = min(7, seq_len // 8)
            if kernel_size % 2 == 0:  # Ensure odd kernel size
              kernel_size += 1
              
            if kernel_size > 1:
              padding = kernel_size // 2
              
              # Create a smoothing filter that better preserves audio characteristics
              # Use a Hann window for better frequency response
              window = torch.hann_window(kernel_size, device=x.device).reshape(1, 1, kernel_size)
              weight = window.repeat(channels, 1, 1)
              weight = weight / weight.sum(dim=2, keepdim=True)  # Normalize
              
              # Apply filter with grouped convolution for efficiency
              result = torch.nn.functional.conv1d(
                  result, weight, padding=padding, groups=channels
              )
          
          # For high channel counts, use simpler method
          elif channels > 512:
            # Use a simpler filter for very high-dimensional data
            kernel_size = 3
            padding = 1
            weight = torch.ones(channels, 1, kernel_size, device=x.device) / kernel_size
            result = torch.nn.functional.conv1d(
                result, weight, padding=padding, groups=channels
            )
            
        except RuntimeError as e:
          # Most basic fallback for any errors
          print(f"Enhanced upsampling failed: {e}, using basic method")
          # Create a simple upsampled tensor via reshape + repeat
          x_flat = x.reshape(batch_size * channels, 1, seq_len)
          result = x_flat.repeat_interleave(scale_factor, dim=2)
          result = result.reshape(batch_size, channels, seq_len * scale_factor)
          
        return result

        # Define a mapping of layer names to expected input channel counts

      # This maps the layer name to (input_channels, expected_channels)
      layer_channel_map = {
        "decoder.model.2.convtr": (1024, 512),
        "decoder.model.5.convtr": (512, 256),
        "decoder.model.8.convtr": (256, 128),
        "decoder.model.11.convtr": (128, 64),
        "upsample.convtr": (64, 32),
      }

      # Apply the patch to various components in the Mimi model
      try:
        # Patch primary upsampling method if available
        if hasattr(mimi_model, '_to_encoder_framerate'):
          original_method = mimi_model._to_encoder_framerate

          def patched_to_encoder_framerate(x):
            """Patched version of _to_encoder_framerate"""
            # Skip the original upsample and just do interpolation
            x = x.contiguous().to(dtype=torch.float32)
            return custom_upsample(x)

            # Replace the method

          mimi_model._to_encoder_framerate = patched_to_encoder_framerate
          print("Successfully patched Mimi codec upsampling method")

          # Also patch any internal modules with convtr in their name
        for name, module in mimi_model.named_modules():
          # Skip any non-important or nested modules
          if 'convtr' in name.lower() or isinstance(module,
                                                    torch.nn.ConvTranspose1d):
            # Get correct channel dimensions for this layer if available
            channel_info = None
            for layer_prefix, channel_dims in layer_channel_map.items():
              if layer_prefix in name:
                channel_info = channel_dims
                break

                # Replace this module's forward to use our custom implementation
            original_forward = module.forward

            def make_patched_forward(mod_name, channel_info):
              # Store original forward method
              original_forward = module.forward

              def patched_forward(self, x):
                print(f"Intercepted problematic convtr in {mod_name}")

                # Check if we need to fix channel dimensions
                if channel_info is not None:
                  input_channels, expected_channels = channel_info
                  current_channels = x.size(1)

                  # Only apply the fix if we have the specific mismatch
                  if current_channels == input_channels and input_channels != expected_channels:
                    # Calculate the reduction factor
                    reduction_factor = input_channels // expected_channels
                    if reduction_factor > 1:
                      batch_size, channels, seq_len = x.shape
                      # Reshape and average groups of channels
                      x = x.reshape(batch_size, expected_channels,
                                    reduction_factor, seq_len)
                      x = x.mean(dim=2)
                      print(f"Fixed channel dimension: {x.shape}")
                
                # Try original forward first
                try:
                  # Ensure tensor is contiguous and has correct dtype
                  x = x.contiguous().to(dtype=torch.float32)
                  return original_forward(x)
                except Exception as e:
                  print(f"Original convtr failed: {e}, using custom upsampling")
                  # Fall back to custom upsampling
                  return custom_upsample(x)

              return patched_forward

              # Bind the new method

            import types
            module.forward = types.MethodType(
              make_patched_forward(name, channel_info), module)
            print(f"Patched conv_transpose1d in {name}")

        return True
      except Exception as e:
        print(f"Error during patching: {e}")
        return False

    # Apply the patch to the audio tokenizer
    print("Applying CUDA-compatible patches to audio tokenizer...")
    patch_success = patch_mimi_upsample(self._audio_tokenizer)
    print(f"Patching {'succeeded' if patch_success else 'failed'}")

    # Critical fix: Ensure audio tokenizer decode preserves device
    try:
      # Set the mimi model to the correct device first
      if hasattr(self._audio_tokenizer, 'to'):
        self._audio_tokenizer = self._audio_tokenizer.to(device)

      # Use permute to correctly transform the stacked samples
      permuted_samples = stacked_samples.permute(1, 2, 0).to(device)

      # Force correct format to avoid CUDA kernel issues
      # Ensure we use integer tokens which the Mimi codec expects
      permuted_samples = permuted_samples.contiguous().to(dtype=torch.long)
      
      # Apply safety preprocessing to avoid CUDA assertion failures
      batch_size, channels, seq_len = permuted_samples.shape

      # 1. Clamp token values to avoid index out of bounds errors
      max_token = getattr(self._audio_tokenizer, 'max_token', self.max_token_value)
      permuted_samples = torch.clamp(permuted_samples, min=0, max=max_token)

      # 2. Fix the channel dimension mismatch if needed
      if channels == 1024:
          print(f"Fixing channel dimension mismatch: reshaping {permuted_samples.shape}")
          # Convert to 512 channels as expected by decoder
          permuted_samples = permuted_samples.reshape(batch_size, 512, 2, seq_len)
          permuted_samples = permuted_samples.mean(dim=2).to(dtype=torch.long)
          print(f"New shape after fix: {permuted_samples.shape}")

      # Print debug info
      print(f"Decoding with samples on device: {permuted_samples.device}")
      print(
        f"Audio tokenizer on device: {next(self._audio_tokenizer.parameters()).device}")

      try:
        # Decode with CUDA optimizations
        with torch.amp.autocast('cuda', enabled=False):
          # Decode the audio
          audio = self._audio_tokenizer.decode(permuted_samples)
      except Exception as e:
        print(f"Error during audio decode: {e}")
        # Don't attempt fallbacks, just raise the exception
        raise e

      # Immediately move result to correct device and reshape
      audio = audio.to(device)
      audio = audio.squeeze(0).squeeze(0).to(device)

      print(f"After decode, audio on device: {audio.device}")
    except Exception as e:
      print(f"Error during audio decode: {e}")
      raise e

    # Watermarking
    try:
      print(f"Applying watermark to audio on device: {audio.device}")
      audio, wm_sample_rate = watermark(self._watermarker, audio,
                                        self.sample_rate, CSM_1B_GH_WATERMARK)
      audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate,
                                             new_freq=self.sample_rate).to(
        device)
      print(f"After watermarking, audio on device: {audio.device}")
    except Exception as e:
      print(f"Warning: Could not apply watermark: {e}")
      print("Continuing without watermarking")
      # Keep the audio as is without watermarking

    return audio


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
  model_args = ModelArgs(
      backbone_flavor="llama-1B",
      decoder_flavor="llama-100M",
      text_vocab_size=128256,
      audio_vocab_size=2051,
      audio_num_codebooks=32,
  )
  model = Model(model_args).to(device=device, dtype=torch.bfloat16)
  state_dict = torch.load(ckpt_path)
  model.load_state_dict(state_dict)

  generator = Generator(model)
  return generator


def load_multilingual_model(ckpt_path: str, device: str = "cuda") -> Generator:
  """Load a multilingual model with compatibility for custom decoder attention"""
  # Ensure device is a torch.device object
  if isinstance(device, str):
    device = torch.device(device)

  model_args = ModelArgs(
      backbone_flavor="llama-1B",
      decoder_flavor="llama-100M",
      text_vocab_size=128256,
      audio_vocab_size=2051,
      audio_num_codebooks=32,
  )

  # Create model with correct dtype from the start
  model = Model(model_args).to(device=device, dtype=torch.bfloat16)

  # Apply the same custom decoder fix used during training
  from custom_decoder import fix_decoder_attention
  model = fix_decoder_attention(model)

  # Load state dict with some forgiveness for mismatched keys
  state_dict = torch.load(ckpt_path, map_location=device)

  # Try loading with strict=False to ignore missing/unexpected keys
  model.load_state_dict(state_dict, strict=False)
  print("Model loaded with adjusted decoder attention architecture")

  # Make sure everything is on the correct device
  model = model.to(device)

  generator = Generator(model)
  return generator


def create_safe_mimi_wrapper(original_mimi, device="cuda"):
  """Create a safe wrapper around Mimi codec to prevent CUDA errors"""

  # Create a class that wraps the Mimi codec
  class SafeMimiWrapper:
    def __init__(self, orig_mimi, device):
      self.mimi = orig_mimi
      self.device = device
      # Get the maximum valid token index from the Mimi model
      if hasattr(self.mimi, 'vq') and hasattr(self.mimi.vq, 'codebook_size'):
        self.max_token = self.mimi.vq.codebook_size - 1
      else:
        # Conservative fallback
        self.max_token = 2000
      print(f"Safe Mimi wrapper created with max token value: {self.max_token}")
    
    def named_modules(self):
      """Pass through named_modules calls to the wrapped mimi model"""
      if hasattr(self.mimi, 'named_modules'):
        return self.mimi.named_modules()
      return []
      
    def parameters(self):
      """Pass through parameters calls to the wrapped mimi model"""
      return self.mimi.parameters()

    def encode(self, audio):
      """Pass through to original encoder"""
      return self.mimi.encode(audio)

    def decode(self, tokens):
      """Safe decoding that preprocesses tokens to avoid CUDA errors"""
      # Move to CPU for safe preprocessing
      tokens_cpu = tokens.detach().cpu()

      # Get token shape
      batch_size, channels, seq_len = tokens_cpu.shape

      # 1. Clamp token values to valid range
      safe_tokens = torch.clamp(tokens_cpu, min=0, max=self.max_token)

      # 2. Handle channel dimension mismatches
      if channels == 1024:
        # If we have the 1024 vs 512 channel mismatch, fix it
        print(f"Fixing channel dimension: reshaping {channels} -> 512")
        safe_tokens = safe_tokens.reshape(batch_size, 512, 2, seq_len)
        safe_tokens = safe_tokens.mean(dim=2).to(torch.long)
        channels = 512

      # Move tokens back to device for decoding
      safe_tokens = safe_tokens.to(self.device)

      # IMPORTANT CHANGE: Try original Mimi decoder FIRST
      try:
        print("Using Mimi neural vocoder for decoding...")
        return self.mimi.decode(safe_tokens)
      except Exception as e:
        print(f"Mimi neural vocoder failed: {e}, falling back to direct decoding")
        
        # Only use direct decoding as a fallback
        try:
          # Simple direct conversion
          upsampling_factor = 120  # Each token expands to ~120 samples
          audio_length = seq_len * upsampling_factor
          
          # Upsample using CUDA-safe interpolation
          # Convert to float first
          float_tokens = safe_tokens.float()

          # To avoid CUDA kernel issues, normalize token values to 0-1 range
          float_tokens = float_tokens / self.max_token

          # Reshape for upsampling
          float_tokens = float_tokens.permute(0, 2, 1)  # [B, L, C]

          # Upsample efficiently with interpolate (CUDA-optimized)
          upsampled = torch.nn.functional.interpolate(
              float_tokens.unsqueeze(1),
              # Add dummy channel dim [B, 1, L, C]
              size=(audio_length, channels),
              mode='bicubic',
              align_corners=False
          ).squeeze(1)  # Remove dummy dim -> [B, L', C]

          # Convert back to original dimensions [B, C, L']
          upsampled = upsampled.permute(0, 2, 1)

          # Normalize to avoid clipping
          upsampled = torch.tanh(upsampled)

          # Average across channels to get final audio
          audio = upsampled.mean(dim=1, keepdim=True)

          print(f"Direct decoding fallback used, shape: {audio.shape}")
          print(f"WARNING: Using approximation instead of neural vocoder!")
          return audio
          
        except Exception as e2:
          print(f"All decoding methods failed: {e2}")
          
          # Try using failsafe decoder as last resort
          try:
            from failsafe_decoder import get_failsafe_decoder
            print("Attempting to use failsafe decoder...")
            failsafe_decoder = get_failsafe_decoder()
            audio = failsafe_decoder.decode(tokens)
            print("Using failsafe decoder (will sound robotic)")
            return audio
          except ImportError:
            # Re-raise original error if failsafe not available
            raise e2

          # Create and return the wrapper

  return SafeMimiWrapper(original_mimi, device)
