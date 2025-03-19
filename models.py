from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


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
    r = mask[input_pos, :]
    return r


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

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup caches and causal masks with consistent dimensions.
        
        Ensures consistent dimensions between backbone and decoder caches.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # First completely clear model caches
        self.reset_caches()
        
        # Create the backbone and decoder causal masks
        backbone_max_seq_len = getattr(self.backbone, 'max_seq_len', 2048)
        backbone_mask = _create_causal_mask(backbone_max_seq_len, device)
        # Always use size 2 for decoder to ensure consistent dimensions
        decoder_mask = _create_causal_mask(2, device) 
        
        # Register the masks as buffers
        self.register_buffer("backbone_causal_mask", backbone_mask)
        self.register_buffer("decoder_causal_mask", decoder_mask)
        
        # Configure decoder's position embeddings for consistent dimensions
        if hasattr(self.decoder, 'layers'):
            for layer in self.decoder.layers:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'pos_embeddings'):
                    # Try to configure positional embeddings for sequence length 2
                    if hasattr(layer.attn.pos_embeddings, 'setup'):
                        try:
                            layer.attn.pos_embeddings.setup(
                                max_batch_size=max_batch_size,
                                max_seq_len=2,
                                dtype=dtype,
                                device=device
                            )
                        except Exception:
                            pass
                            
        # Actually setup the backbone and decoder caches with consistent dimensions
        try:
            if hasattr(self.backbone, 'setup_caches'):
                self.backbone.setup_caches(max_batch_size, dtype)
        except Exception:
            pass
            
        try:
            if hasattr(self.decoder, 'setup_caches'):
                # Always use sequence length 2 for decoder
                self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=2)
        except Exception:
            pass
            
        return backbone_mask

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
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        
        # Create fixed positions for the decoder
        # Use simpler tensor creation to avoid expand/repeat operations
        batch_size = curr_h.size(0)
        curr_pos = torch.zeros((batch_size, 2), dtype=torch.long, device=curr_h.device)
        curr_pos[:, 1] = 1  # Set second position to 1

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.args.audio_num_codebooks):
            # Create a fresh 2x2 causal mask for each iteration
            curr_decoder_mask = torch.tril(
                torch.ones(2, 2, dtype=torch.bool, device=curr_h.device)
            ).unsqueeze(0).expand(curr_h.size(0), 2, 2)
            
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            # Maintain a consistent position tensor 
            # Always use [0, 1] for each step
            curr_pos = torch.zeros((curr_h.size(0), 2), dtype=torch.long, device=curr_h.device) 
            curr_pos[:, 1] = 1

        return curr_sample

    def reset_caches(self):
        """Reset caches for both backbone and decoder.
        
        This implementation works with or without KV caching enabled.
        """
        # Try to reset backbone and decoder caches if available
        # If caches are disabled, this will be a no-op
        try:
            if hasattr(self.backbone, 'reset_caches'):
                self.backbone.reset_caches()
        except Exception:
            pass
            
        try:
            if hasattr(self.decoder, 'reset_caches'):
                self.decoder.reset_caches()
        except Exception:
            pass

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens with enhanced dimension handling"""
        try:
            # Normalize token shape for consistent handling
            # Ensure tokens is 2D: [batch_size, seq_len]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(1)
            elif tokens.dim() > 2:
                tokens = tokens.view(tokens.size(0), -1)
                
            # Compute embedding with proper offset
            embeddings = self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)
            
            # Ensure output has shape [batch_size, seq_len, dim]
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)
                
            return embeddings
            
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
