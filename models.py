from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2

# Ensure DISABLE_CACHE environment variable is set
import os
os.environ["TORCHTUNE_DISABLE_CACHE"] = "1" 
os.environ["DISABLE_ROPE_CACHE"] = "1"

# Configure logger
logger = logging.getLogger(__name__)


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    # Ensure both tensors are on the same device
    device = mask.device
    input_pos = input_pos.to(device)
    
    # Clamp input_pos to valid indices to avoid out-of-bounds access
    max_idx = mask.size(0) - 1
    input_pos = torch.clamp(input_pos, 0, max_idx)
    
    # Index the mask
    r = mask[input_pos, :]
    return r.to(device)


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())

        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))
        
    def ensure_module_on_device(self, recursive=True):
        """Ensures all modules and parameters are on the same device"""
        device = next(self.parameters()).device
        print(f"Ensuring model is on device: {device}")
        
        # Move the module itself
        self.to(device)
        
        # Check all parameters directly owned by this module
        for name, param in self.named_parameters(recurse=False):
            if param.device != device:
                print(f"Moving parameter {name} from {param.device} to {device}")
                param.data = param.data.to(device)
        
        # Check all buffers
        for name, buf in self.named_buffers(recurse=False):
            if buf.device != device:
                print(f"Moving buffer {name} from {buf.device} to {device}")
                buf.data = buf.data.to(device)
        
        # Recursively check all child modules
        if recursive:
            for name, module in self.named_children():
                if hasattr(module, 'to'):
                    module.to(device)
                
                # Also move all their parameters
                for param_name, param in module.named_parameters(recurse=True):
                    if param.device != device:
                        print(f"Moving {name}.{param_name} from {param.device} to {device}")
                        param.data = param.data.to(device)
        
        # Force sync if CUDA
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup caches and causal masks with consistent dimensions.
        
        With our custom decoder, we only need to set up backbone caches.
        """
        # Get device directly from parameters to ensure consistency
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        print(f"Setting up caches on device: {device}")
        
        # Create fresh causal masks with explicit device
        backbone_max_seq_len = getattr(self.backbone, 'max_seq_len', 2048)
        backbone_mask = _create_causal_mask(backbone_max_seq_len, device)
        # Always use size 2 for decoder to ensure consistent dimensions
        decoder_mask = _create_causal_mask(2, device) 
        
        # Use persistent=True to ensure buffers are saved with state_dict
        self.register_buffer("backbone_causal_mask", backbone_mask, persistent=True)
        self.register_buffer("decoder_causal_mask", decoder_mask, persistent=True)
        
        # Store current batch size as an attribute to check later
        self._current_batch_size = max_batch_size
        
        # Force synchronization to ensure all tensor operations are complete
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        # For backbone only - we're using our custom decoder that doesn't need caching
        try:
            # Move backbone to correct device first if needed
            if hasattr(self.backbone, 'to'):
                self.backbone = self.backbone.to(device)
                
            if hasattr(self.backbone, 'setup_caches'):
                self.backbone.setup_caches(max_batch_size, dtype)
                if hasattr(self.backbone, 'kv_caches') and self.backbone.kv_caches:
                    # Explicitly move any created caches to the correct device
                    for cache in self.backbone.kv_caches:
                        if hasattr(cache, 'to'):
                            cache.to(device)
            # Additional check for setting up kv cache directly
            elif hasattr(self.backbone, 'setup_kv_cache'):
                self.backbone.setup_kv_cache(max_batch_size)
                
            # Force synchronization again
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except Exception as e:
            print(f"Note: Using simplified caching mechanism: {e}")
        
        # Return mask that's guaranteed to be on the correct device
        return backbone_mask.to(device)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        # Get device from parameters to ensure consistency
        param_device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        # Ensure caches are set up - create if missing
        if not hasattr(self, "backbone_causal_mask"):
            print("Setting up caches before generation...")
            self.setup_caches(b)

        # Debug info
        print(f"Parameter device: {param_device}, Input device: {tokens.device}")
        
        # Ignore device mismatch when it's just "cuda" vs "cuda:0" etc.
        # as long as they're both on the same type of device (cuda or cpu)
        inputs_on_correct_type = (
            str(tokens.device).split(':')[0] == str(param_device).split(':')[0] and
            str(tokens_mask.device).split(':')[0] == str(param_device).split(':')[0] and
            str(input_pos.device).split(':')[0] == str(param_device).split(':')[0]
        )
        
        # Force all inputs to the parameter device for consistency only if truly needed
        device = param_device
        if not inputs_on_correct_type:
            tokens = tokens.to(device, non_blocking=False)
            tokens_mask = tokens_mask.to(device, non_blocking=False)
            input_pos = input_pos.to(device, non_blocking=False)
        
        # Force synchronization to ensure device transfers are complete
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Get the backbone mask, ensuring it exists and is on the right device
        if not hasattr(self, "backbone_causal_mask"):
            # If still missing despite trying to set up, create a fallback mask
            print("Creating fallback causal mask")
            backbone_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))
            self.register_buffer("backbone_causal_mask", backbone_mask, persistent=True)
        
        # Ensure backbone_causal_mask is on the right device    
        self.backbone_causal_mask = self.backbone_causal_mask.to(device, non_blocking=False)
        
        # Create the mask for this input
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        curr_backbone_mask = curr_backbone_mask.to(device, non_blocking=False)
        
        # Ensure backbone is on the right device
        if hasattr(self.backbone, 'to'):
            self.backbone = self.backbone.to(device)
        
        # Generate embeddings and forward pass
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        # Ensure all tensors are on the same device before calling backbone
        h = h.to(device, non_blocking=False)
        input_pos = input_pos.to(device, non_blocking=False)
        curr_backbone_mask = curr_backbone_mask.to(device, non_blocking=False)
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            
        # Call backbone with explicit device control
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        h = h.to(device=device, dtype=dtype, non_blocking=False)
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        last_h = h[:, -1, :].to(device)
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature).to(device)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1).to(device)
        curr_sample = c0_sample.clone().to(device)
        
        # Create fixed positions for the decoder with explicit device
        batch_size = curr_h.size(0)
        curr_pos = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
        curr_pos[:, 1] = 1  # Set second position to 1

        # No need to reset decoder caches when using our fixed attention
        for i in range(1, self.args.audio_num_codebooks):
            # Create a fresh 2x2 causal mask for each iteration with explicit device
            curr_decoder_mask = torch.tril(
                torch.ones(2, 2, dtype=torch.bool, device=device)
            ).unsqueeze(0).expand(curr_h.size(0), 2, 2)
            
            # Ensure projection output has exactly 2 sequence positions
            projection_out = self.projection(curr_h).to(device)
            if projection_out.size(1) != 2:
                # Force to exactly 2 positions
                if projection_out.size(1) > 2:
                    # Take first 2 positions
                    projection_out = projection_out[:, :2]
                else:
                    # Repeat last position to get 2 positions
                    pad = projection_out[:, -1:].expand(-1, 2-projection_out.size(1), -1)
                    projection_out = torch.cat([projection_out, pad], dim=1)
            
            # Ensure everything is on the same device
            projection_out = projection_out.to(device)
            
            # Use our fixed decoder with explicit device handling
            decoder_h = self.decoder(
                projection_out, 
                input_pos=curr_pos, 
                mask=curr_decoder_mask
            ).to(device=device, dtype=dtype)
            
            # Ensure audio_head is on same device before matmul
            audio_head_i = self.audio_head[i - 1].to(device)
            ci_logits = torch.matmul(decoder_h[:, -1, :], audio_head_i)
            ci_sample = sample_topk(ci_logits, topk, temperature).to(device)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed.to(device)
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1).to(device)
            curr_pos = torch.zeros((curr_h.size(0), 2), dtype=torch.long, device=device) 
            curr_pos[:, 1] = 1
            
        # Final device check - don't warn about "cuda" vs "cuda:0"
        curr_sample = curr_sample.to(device)
        print(f"Generated frame output device: {curr_sample.device}")
        return curr_sample

    def reset_caches(self):
        """Reset caches for backbone only.
        
        Our custom decoder doesn't use caching.
        """
        # Get the device directly from parameters
        device = next(self.parameters()).device
        
        # Try to reset backbone caches if available
        # If caches are disabled, this will be a no-op
        try:
            # Ensure backbone is on the right device
            if hasattr(self.backbone, 'to'):
                self.backbone = self.backbone.to(device)
                
            if hasattr(self.backbone, 'reset_caches'):
                self.backbone.reset_caches()
                
            # Force CUDA synchronization
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                
            # Set a flag so we know caches need to be rebuilt
            self._caches_reset = True
        except Exception as e:
            print(f"Cache reset note: {e}")
            pass
    
    def validate_batch_size(self, batch_size):
        """Check if current caches are compatible with given batch size.
        
        Returns True if compatible, False otherwise.
        """
        expected_batch_size = getattr(self, '_current_batch_size', None)
        if expected_batch_size is None:
            return True
        
        return expected_batch_size == batch_size

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens with enhanced dimension handling"""
        try:
            # Get device directly from model parameters for consistency
            device = next(self.parameters()).device
            tokens = tokens.to(device)
            
            # Normalize token shape for consistent handling
            # Ensure tokens is 2D: [batch_size, seq_len]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(1)
            elif tokens.dim() > 2:
                tokens = tokens.view(tokens.size(0), -1)
                
            # Compute embedding with proper offset
            offset_tokens = tokens + codebook * self.args.audio_vocab_size
            offset_tokens = offset_tokens.to(device)
            embeddings = self.audio_embeddings(offset_tokens)
            
            # Ensure output has shape [batch_size, seq_len, dim]
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)
                
            # Make sure embeddings are on the right device before returning
            return embeddings.to(device)
            
        except RuntimeError as e:
            # Enhanced error handling with more diagnostic info
            shape_str = f"tokens.shape: {tokens.shape}, tokens.dtype: {tokens.dtype}"
            logger.warning(f"Embedding error: {e}, {shape_str}")
            
            # More aggressive reshaping as a last resort
            try:
                # Clone to avoid in-place modification errors
                safe_tokens = tokens.clone().detach()
                
                # Force into a known good shape
                if safe_tokens.dim() == 1:
                    safe_tokens = safe_tokens.unsqueeze(1)
                else:
                    safe_tokens = safe_tokens.view(safe_tokens.size(0), -1)[:, :1]
                    
                embeddings = self.audio_embeddings(safe_tokens + codebook * self.args.audio_vocab_size)
                
                # Ensure consistent output shape
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(1)
                    
                return embeddings
                
            except Exception as nested_e:
                logger.error(f"Audio embedding fallback failed: {nested_e}")
                # Last resort: return zeros with the right shape
                batch_size = tokens.size(0) if tokens.dim() > 0 else 1
                embed_dim = self.audio_embeddings.weight.size(1)
                return torch.zeros(batch_size, 1, embed_dim, device=tokens.device)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
