import math
import torch
import torch.nn as nn

class FixedPositionRoPE(nn.Module):
    """Custom RoPE implementation for fixed positions [0,1]"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Precompute sin/cos for positions 0 and 1 only
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Only compute for positions 0 and 1
        self.register_buffer("cos_cached", torch.cos(torch.outer(torch.tensor([0.0, 1.0]), inv_freq)))
        self.register_buffer("sin_cached", torch.sin(torch.outer(torch.tensor([0.0, 1.0]), inv_freq)))
    
    def forward(self, x, position_ids):
        # x: [batch, seq_len, heads, head_dim]
        batch, seq_len, heads, dim = x.shape
        # Clamp positions to 0 or 1
        position_ids = torch.clamp(position_ids, max=1)
        
        # Select the precomputed sin/cos for positions 0 and 1
        cos = self.cos_cached[position_ids].view(batch, seq_len, 1, dim//2, 1)  # [batch, seq, 1, dim//2, 1]
        sin = self.sin_cached[position_ids].view(batch, seq_len, 1, dim//2, 1)
        
        # Apply RoPE rotation
        x_reshape = x.view(batch, seq_len, heads, dim//2, 2)
        x_out = torch.empty_like(x_reshape)
        x_out[..., 0] = x_reshape[..., 0] * cos - x_reshape[..., 1] * sin
        x_out[..., 1] = x_reshape[..., 1] * cos + x_reshape[..., 0] * sin
        
        return x_out.view(batch, seq_len, heads, dim)


class SimpleDecoderAttention(nn.Module):
    """Simplified decoder attention that doesn't rely on caching"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Use our custom RoPE implementation
        self.pos_embed = FixedPositionRoPE(self.head_dim)
        
        # Add cache-related attributes for compatibility
        self.cache_enabled = False
    
    def forward(self, x, *args, **kwargs):
        # Extract parameters with precedence to named arguments
        mask = kwargs.get('mask', None)
        position_ids = kwargs.get('position_ids', None)
        input_pos = kwargs.get('input_pos', None)
        
        # Backward compatibility for positional args
        if len(args) >= 1 and mask is None:
            mask = args[0]
        
        batch_size, seq_len, _ = x.shape
        
        # Default positions if not provided
        if position_ids is None:
            if input_pos is not None:
                # Use input_pos if provided (compatibility with original interface)
                position_ids = input_pos
            else:
                # Default to [0, 1] positions
                position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Ensure input is on the correct device and has the right dtype
        device = x.device
        dtype = x.dtype
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Ensure tensors are on the expected device
        if q.device != device:
            q = q.to(device=device, dtype=dtype, non_blocking=False)
        if k.device != device:
            k = k.to(device=device, dtype=dtype, non_blocking=False)
        if v.device != device:
            v = v.to(device=device, dtype=dtype, non_blocking=False)
        
        # Apply our custom RoPE
        q = self.pos_embed(q, position_ids)
        k = self.pos_embed(k, position_ids)
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask - ensure same device first
        if mask is not None:
            # Ensure mask is on the same device as attn_weights
            if mask.device != attn_weights.device:
                mask = mask.to(device=attn_weights.device)
                
            # Handle different mask formats
            if mask.dim() == 3:  # [batch, seq, seq]
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), float('-inf'))
            elif mask.dim() == 2:  # [seq, seq]
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
    
    def caches_are_enabled(self):
        """Return whether caching is enabled (always False for our implementation)"""
        return False
        
    def reset_caches(self):
        """No-op since we don't use caches"""
        pass


def fix_decoder_attention(model):
    """Replace problematic attention modules with our fixed implementation"""
    if hasattr(model.decoder, 'layers'):
        for i, layer in enumerate(model.decoder.layers):
            if hasattr(layer, 'attn'):
                # Get dimensions from existing layer
                embed_dim = layer.attn.q_proj.out_features if hasattr(layer.attn, 'q_proj') else 1024
                num_heads = layer.attn.num_heads if hasattr(layer.attn, 'num_heads') else 8
                
                # Replace with our custom implementation
                layer.attn = SimpleDecoderAttention(embed_dim, num_heads)
                print(f"Replaced attention in decoder layer {i}")
    return model
