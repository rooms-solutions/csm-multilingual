import math
import torch
import torch.nn as nn

class FixedPositionRoPE(nn.Module):
    """Custom RoPE implementation for fixed positions [0,1]"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        
        # Precompute sin/cos for positions 0 and 1 only
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float) / half_dim))
        
        # Only compute for positions 0 and 1
        self.register_buffer("cos_cached", torch.cos(torch.outer(torch.tensor([0.0, 1.0]), inv_freq)))
        self.register_buffer("sin_cached", torch.sin(torch.outer(torch.tensor([0.0, 1.0]), inv_freq)))
    
    def forward(self, x, position_ids):
        """Apply RoPE with careful dimension handling."""
        batch, seq_len, heads, dim = x.shape
        half_dim = dim // 2
        
        # Handle position IDs to ensure correct format
        if position_ids.dim() == 1:
            # Expand to batch size
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        
        # Ensure right sequence length and clamp to valid values
        if position_ids.size(1) != seq_len:
            # Create position IDs for sequence length
            new_pos_ids = torch.zeros(batch, seq_len, dtype=torch.long, device=x.device)
            # Copy existing IDs
            copy_len = min(seq_len, position_ids.size(1))
            new_pos_ids[:, :copy_len] = position_ids[:, :copy_len]
            # Default remaining positions to 1
            if copy_len < seq_len:
                new_pos_ids[:, copy_len:] = 1
            position_ids = new_pos_ids
        
        # Clamp to valid values (0 or 1)
        position_ids = position_ids.clamp(max=1).long()
        
        # Split input into real and imaginary parts
        x_real = x[..., :half_dim]  # First half
        x_imag = x[..., half_dim:]  # Second half
        
        # Initialize output tensors
        out_real = torch.zeros_like(x_real)
        out_imag = torch.zeros_like(x_imag)
        
        # Apply rotations for each position without complex reshaping
        for b in range(batch):
            for s in range(seq_len):
                pos = position_ids[b, s].item()
                
                # Get rotation factors for this position
                cos = self.cos_cached[pos]  # [half_dim]
                sin = self.sin_cached[pos]  # [half_dim]
                
                # Perform rotation (complex multiplication)
                out_real[b, s] = x_real[b, s] * cos.unsqueeze(0) - x_imag[b, s] * sin.unsqueeze(0)
                out_imag[b, s] = x_imag[b, s] * cos.unsqueeze(0) + x_real[b, s] * sin.unsqueeze(0)
        
        # Combine real and imaginary parts
        return torch.cat([out_real, out_imag], dim=-1)


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
        
        # Register a device to help with tracking
        self.register_buffer("_device_tracker", torch.zeros(1))
    
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
                # For our fixed 2-position case, explicitly create [0, 1] positions
                position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
                if seq_len > 1:
                    position_ids[:, 1:] = 1  # Position 0 for first token, 1 for all others
        
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
        
        # Force tensors to match input device before reshaping, with stronger guarantees
        device, dtype = x.device, x.dtype
        
        # Ensure we're using the correct device name format (use actual device object, not string)
        if isinstance(device, str) and device == "cuda":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        # Force synchronous copy to device to ensure completion
        q = q.to(device=device, dtype=dtype, non_blocking=False)
        k = k.to(device=device, dtype=dtype, non_blocking=False)
        v = v.to(device=device, dtype=dtype, non_blocking=False)
        
        # Verify device placement was successful
        if q.device != device or k.device != device or v.device != device:
            # Try again with explicit device index if still mismatched
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device = torch.device(f"cuda:{current_device}")
                q = q.to(device=device, dtype=dtype, non_blocking=False)
                k = k.to(device=device, dtype=dtype, non_blocking=False)
                v = v.to(device=device, dtype=dtype, non_blocking=False)
        
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
        # Get device from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        for i, layer in enumerate(model.decoder.layers):
            if hasattr(layer, 'attn'):
                # Get dimensions from existing layer
                embed_dim = layer.attn.q_proj.out_features if hasattr(layer.attn, 'q_proj') else 1024
                num_heads = layer.attn.num_heads if hasattr(layer.attn, 'num_heads') else 8
                
                # Replace with our custom implementation and ensure it's on the same device
                new_attn = SimpleDecoderAttention(embed_dim, num_heads)
                new_attn = new_attn.to(device=device, dtype=dtype)
                
                # Force synchronization to ensure module is on the correct device
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                # Replace the attention module
                layer.attn = new_attn
                print(f"Replaced attention in decoder layer {i} - device: {new_attn._device_tracker.device}")
    
    return model
