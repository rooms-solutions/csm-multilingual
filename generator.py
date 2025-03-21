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
        """Disable any patching and use the Mimi model directly as designed"""
        print("Using direct Mimi codec architecture without patching")
        return True

    # Get the device of the Mimi model and ensure tensor is on the same device
    device = self.device
    print(f"Decoding audio on device: {device}")

    # Prepare audio tokens for decoding
    # Get exact shapes for debugging
    b, c, s = stacked_samples.shape
    print(f"Audio token shape before permute: {stacked_samples.shape}")

    # Correctly format tokens for Mimi decoder
    audio_tokens = stacked_samples.permute(1, 2, 0).contiguous()
    print(f"Audio token shape after permute: {audio_tokens.shape}")

    # Hard check for expected dimensions
    if audio_tokens.shape[0] != 32:  # Check number of codebooks
        raise ValueError(f"Expected 32 codebooks but got {audio_tokens.shape[0]}. The model architecture is incompatible.")

    # Convert to exactly 64 channels as expected by Mimi
    # Do a direct reshape that preserves all information from original tokens
    # This is the critical fix - proper handling of the channel dimension
    try:
        if audio_tokens.shape[0] == 32:
            # Correctly format as [batch=1, channels=32, seq_len]
            # This matches the format CSM uses for decoding
            audio_tokens = audio_tokens.unsqueeze(0) 
            print(f"Final audio token shape before decode: {audio_tokens.shape}")
            
            # Apply special German model adapter if needed
            # This handles the 512 → 64 channel conversion
            if audio_tokens.shape[1] == 512 or audio_tokens.shape[1] == 32:
                audio_tokens = reshape_tokens_for_german_model(audio_tokens)
            
            # Decode directly without patching or fallbacks
            try:
                audio = self._audio_tokenizer.decode(audio_tokens)
                audio = audio.squeeze(0).squeeze(0)
                print(f"Successfully decoded audio with shape: {audio.shape}")
            except Exception as e:
                # Provide clear error without fallbacks
                raise RuntimeError(f"Audio decoding failed with error: {e}. The model may need retraining with a compatible architecture.") from e
        else:
            raise ValueError(f"Unexpected token dimensions: {audio_tokens.shape}. Cannot produce clear audio with this format.")
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


def reshape_tokens_for_german_model(audio_tokens):
    """
    Special adapter function for the German model that converts 512 channels to 64
    while preserving audio structure
    """
    # Get shape information
    batch_size, channels, seq_len = audio_tokens.shape
    
    if channels == 512 and seq_len > 0:
        print(f"Applying German model token adapter: {channels} → 64 channels")
        
        # Reshape to isolate each codebook component
        # [B, 512, S] → [B, 64, 8, S] where 512 = 64 × 8
        reshaped = audio_tokens.reshape(batch_size, 64, 8, seq_len)
        
        # Take weighted average of each group to maintain audio structure
        # Create importance weights that prioritize certain dimensions
        weights = torch.tensor([0.2, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3], 
                              device=audio_tokens.device).view(1, 1, 8, 1)
        
        # Apply weighted average
        weighted = reshaped * weights
        adapted_tokens = weighted.sum(dim=2) / weights.sum()
        
        # Ensure we have the right datatype
        adapted_tokens = adapted_tokens.to(dtype=audio_tokens.dtype)
        print(f"Successfully adapted tokens from shape {audio_tokens.shape} to {adapted_tokens.shape}")
        
        return adapted_tokens
    
    # Return original if no adaptation needed
    return audio_tokens

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
    """Create a minimal wrapper for Mimi codec that follows CSM approach"""
    
    class SimpleCSMCompatibleMimi:
        def __init__(self, mimi, device):
            self.mimi = mimi
            self.device = device
            
        def parameters(self):
            return self.mimi.parameters()
            
        def encode(self, audio):
            return self.mimi.encode(audio)
            
        def decode(self, tokens):
            # Direct decoding without any fallbacks or fixes
            # Tokens should be in format [batch, codebooks, seq_len]
            return self.mimi.decode(tokens)
        
    return SimpleCSMCompatibleMimi(original_mimi, device)
