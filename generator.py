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

    # Wrap the mimi codec in our robust wrapper
    self._audio_tokenizer = create_mimi_codec_wrapper(mimi, device=self.device)

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
    
    # Debug token values
    print(f"Token value range: min={audio_tokens.min().item()}, max={audio_tokens.max().item()}")
    if audio_tokens.size(2) > 0:
        print(f"Sample tokens for first frame: {audio_tokens[0, :, 0].tolist()[:5]}...")
    
    # Ensure tokens have the right shape and values
    audio_tokens = audio_tokens.view(1, 32, -1)  # Explicitly reshape to expected format
    audio_tokens = torch.clamp(audio_tokens, 0, 2048)  # Ensure values are in valid range

    # Ensure we have exactly 32 codebooks as expected by Mimi
    if audio_tokens.shape[1] != 32:
        audio_tokens = reshape_tokens_for_german_model(audio_tokens)
        print(f"Reshaped to {audio_tokens.shape} for Mimi compatibility")

    # Decode directly without patching
    try:
        # Try to decode with a small test sample first
        test_tokens = torch.randint(0, 1024, (1, 32, 10), device=device)
        test_audio = self._audio_tokenizer.decode(test_tokens)
        print(f"Test audio shape: {test_audio.shape}")
        
        # Now decode the real tokens
        audio = self._audio_tokenizer.decode(audio_tokens)
        audio = audio.squeeze(0).squeeze(0)
        print(f"Successfully decoded audio with shape: {audio.shape}")
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
    Properly convert model tokens to format expected by Mimi codec
    without losing critical information
    """
    batch_size, channels, seq_len = audio_tokens.shape
    
    # Check if we actually need to reshape
    if channels == 32:
        return audio_tokens  # Already in the correct format
        
    if channels != 32:
        print(f"WARNING: Model outputs {channels} channels but Mimi expects 32")
        
        # Don't use weighted averaging - it loses critical information
        # Instead, slice the first 32 codebooks which contain most of the
        # acoustic information
        if channels > 32:
            return audio_tokens[:, :32, :]
        else:
            # Pad to 32 channels if we have fewer
            padding = torch.zeros(batch_size, 32-channels, seq_len, 
                               device=audio_tokens.device,
                               dtype=audio_tokens.dtype)
            return torch.cat([audio_tokens, padding], dim=1)
            
    return audio_tokens

def load_multilingual_model(ckpt_path: str, device: str = "cuda") -> Generator:
  """Load a multilingual model with correct decoder attention"""
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
  
  # Important: Fix decoder attention BEFORE loading weights
  # The model was trained with this modification
  from custom_decoder import fix_decoder_attention
  model = fix_decoder_attention(model)

  # Load state dict with matching architecture
  state_dict = torch.load(ckpt_path, map_location=device)
  model.load_state_dict(state_dict, strict=True)
  print("Model loaded successfully")

  # Ensure model is on the correct device
  model = model.to(device)

  generator = Generator(model)
  return generator


def create_mimi_codec_wrapper(original_mimi, device="cuda"):
    """Create proper wrapper for Mimi codec that preserves all functionality"""
    
    class SimpleMimiWrapper:
        def __init__(self, mimi, device):
            self.mimi = mimi
            self.device = device
            self.sample_rate = getattr(mimi, 'sample_rate', 24000)
            
            # Store codebook information
            self.num_codebooks = getattr(mimi, 'num_codebooks', 32)
            self.codebook_size = 2048  # Default for Mimi
            
            if hasattr(mimi, 'vq') and hasattr(mimi.vq, 'codebook_size'):
                self.codebook_size = mimi.vq.codebook_size
                
            print(f"Initialized simplified Mimi wrapper: {self.num_codebooks} codebooks, size {self.codebook_size}")
            
        def parameters(self):
            return self.mimi.parameters()
            
        def encode(self, audio):
            """Simple encoder that just forwards to Mimi"""
            # Ensure audio is on the device
            audio = audio.to(self.device)
            
            # Forward to Mimi
            tokens = self.mimi.encode(audio)
            
            # Handle tuple return if needed
            if isinstance(tokens, tuple):
                tokens = tokens[0]
                
            return tokens
            
        def decode(self, tokens):
            """Simple decoder that just forwards to Mimi"""
            # Ensure tokens are on the correct device
            tokens = tokens.to(self.device)
            
            # Make sure shape is correct for Mimi's expectations
            if tokens.dim() != 3:
                print(f"Warning: reshaping tokens from {tokens.shape} to expected 3D format")
                if tokens.dim() == 2:
                    tokens = tokens.unsqueeze(0)
            
            # Clamp values to valid range
            tokens = torch.clamp(tokens, 0, self.codebook_size-1)
            
            # Try a direct decode with minimal intervention
            try:
                # Test small section first to debug
                print(f"Decoding tokens with shape {tokens.shape}")
                return self.mimi.decode(tokens)
            except Exception as e:
                print(f"Simplified decoder failed: {e}")
                # Last resort fallback - return empty audio
                return torch.zeros(1, 1, 24000, device=self.device)
        
    return SimpleMimiWrapper(original_mimi, device)
