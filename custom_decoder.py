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
        
        # Ensure position_ids always has the right dimensions [batch_size, seq_len]
        if position_ids.dim() == 1:
            # Expand to batch size
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        
        # Very important: Ensure position_ids has exactly seq_len positions
        if position_ids.size(1) != seq_len:
            # Always resize to exact sequence length
            if position_ids.size(1) > seq_len:
                # Truncate
                position_ids = position_ids[:, :seq_len]
            else:
                # Pad with position 1 (we need exactly 2 positions for our case)
                pad_len = seq_len - position_ids.size(1)
                pad = torch.ones(batch, pad_len, dtype=torch.long, device=position_ids.device)
                position_ids = torch.cat([position_ids, pad], dim=1)
        
        # Clamp to valid values (0 or 1)
        position_ids = position_ids.clamp(max=1).long()
        
        # Simpler, more efficient implementation using broadcasting
        # Get embeddings for all positions in the batch at once
        cos_pos = self.cos_cached[position_ids]  # [batch, seq_len, half_dim]
        sin_pos = self.sin_cached[position_ids]  # [batch, seq_len, half_dim]
        
        # Reshape for broadcasting across heads
        cos_pos = cos_pos.unsqueeze(2).expand(-1, -1, heads, -1)  # [batch, seq_len, heads, half_dim]
        sin_pos = sin_pos.unsqueeze(2).expand(-1, -1, heads, -1)  # [batch, seq_len, heads, half_dim]
        
        # Split x into real and imaginary parts
        x_real, x_imag = x[..., :half_dim], x[..., half_dim:]
        
        # Apply rotary embeddings
        out_real = x_real * cos_pos - x_imag * sin_pos
        out_imag = x_imag * cos_pos + x_real * sin_pos
        
        # Combine real and imaginary parts
        return torch.cat([out_real, out_imag], dim=-1)


class EnhancedDecoderAttention(nn.Module):
    """Proper decoder attention that matches original CSM architecture"""
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
        
        # Standard rotary embeddings with proper sequence length support
        self.max_seq_len = 2048
        self.pos_embed = self._create_proper_rope()
        
        # KV caching for efficient generation
        self.k_cache = None
        self.v_cache = None
        self.cache_size = 0
        self.cache_enabled = True
        
    def _create_proper_rope(self):
        """Create proper rotary positional embeddings"""
        from torch import nn
        import math
        
        class ProperRotaryEmbedding(nn.Module):
            def __init__(self, dim, max_pos=2048, base=10000.0):
                super().__init__()
                self.dim = dim
                self.max_pos = max_pos
                self.base = base
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", inv_freq)
                
            def forward(self, x, position_ids):
                """Apply RoPE to tensor x based on position_ids"""
                batch, seq_len, heads, dim = x.shape
                half_dim = dim // 2
                
                # Get embeddings for positions
                inv_freq = self.inv_freq
                positions = position_ids.float()
                
                # Compute sin and cos for positions
                freqs = torch.einsum('bi,j->bij', positions, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().unsqueeze(2)
                sin = emb.sin().unsqueeze(2)
                
                # Apply rotation
                x_real, x_imag = x[..., :half_dim], x[..., half_dim:]
                out_real = x_real * cos - x_imag * sin
                out_imag = x_imag * cos + x_real * sin
                
                return torch.cat([out_real, out_imag], dim=-1)
                
        return ProperRotaryEmbedding(self.head_dim)
        
    def _setup_cache(self, batch_size, seq_len, device):
        """Set up KV cache for efficient generation"""
        if not self.cache_enabled:
            return
            
        self.k_cache = torch.zeros(
            (batch_size, seq_len, self.num_heads, self.head_dim),
            device=device, dtype=self.q_proj.weight.dtype
        )
        self.v_cache = torch.zeros(
            (batch_size, seq_len, self.num_heads, self.head_dim),
            device=device, dtype=self.q_proj.weight.dtype
        )
        self.cache_size = 0
    
    def forward(self, x, *args, **kwargs):
        """Forward pass supporting both training and generation modes with flexible arg handling"""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Extract parameters with precedence to named arguments
        mask = kwargs.get('mask', None)
        position_ids = kwargs.get('position_ids', None)
        input_pos = kwargs.get('input_pos', None)
        
        # Backward compatibility for positional args
        if len(args) >= 1 and mask is None:
            mask = args[0]
        if len(args) >= 2 and position_ids is None:
            position_ids = args[1]
        
        # Set up position IDs if not provided
        if position_ids is None:
            if input_pos is not None:
                position_ids = input_pos
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings properly
        q = self.pos_embed(q, position_ids)
        k = self.pos_embed(k, position_ids)
        
        # Handle KV caching for generation
        if self.cache_enabled and hasattr(self, 'k_cache') and self.k_cache is not None:
            if self.k_cache.size(0) != batch_size:
                self._setup_cache(batch_size, self.max_seq_len, device)
                
            if seq_len == 1 and self.cache_size > 0:  # Autoregressive generation
                # Update cache with new KV
                self.k_cache[:, self.cache_size:self.cache_size+1] = k
                self.v_cache[:, self.cache_size:self.cache_size+1] = v
                
                # Use full cached context
                k_full = self.k_cache[:, :self.cache_size+1]
                v_full = self.v_cache[:, :self.cache_size+1]
                
                # Update cache size
                self.cache_size += 1
                
                # Reshape for attention
                k = k_full
                v = v_full
            else:  # Full sequence processing or cache reset
                self.k_cache[:, :seq_len] = k
                self.v_cache[:, :seq_len] = v
                self.cache_size = seq_len
        
        # Reshape for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Handle various mask formats
            if mask.dim() == 2:  # [seq, seq]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:  # [batch, seq, seq]
                mask = mask.unsqueeze(1)
                
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        # Get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
        
    def caches_are_enabled(self):
        return self.cache_enabled
        
    def reset_caches(self):
        self.cache_size = 0

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
        
        # Add warning tracker to avoid spamming logs
        self._has_warned = False
        
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
        
        # Very important: make sure we have exactly seq_len=2 for our fixed positions case
        # If not, we need to pad or truncate the input
        if seq_len != 2:
            # Log warning only once to avoid spamming
            if not self._has_warned:
                print(f"Note: SimpleDecoderAttention adjusting sequence length from {seq_len} to 2 for fixed position model")
                self._has_warned = True
            
            if seq_len > 2:
                # Only use the first 2 positions if input is too long
                x = x[:, :2, :]
                seq_len = 2
            else:
                # Pad with zeros if input is too short
                pad = torch.zeros(batch_size, 2-seq_len, x.size(2), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
                seq_len = 2
        
        # Default positions if not provided - ensure we have exactly [0,1]
        if position_ids is None:
            if input_pos is not None:
                # Make sure input_pos has correct shape [batch_size, 2]
                if input_pos.size(1) != 2:
                    # Recreate with correct size
                    position_ids = torch.zeros(batch_size, 2, dtype=torch.long, device=x.device)
                    position_ids[:, 1] = 1
                else:
                    position_ids = input_pos
            else:
                # Explicitly create fixed [0, 1] positions
                position_ids = torch.zeros(batch_size, 2, dtype=torch.long, device=x.device)
                position_ids[:, 1] = 1  # Position 0 for first token, 1 for second
        
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
        
        # Get device and dtype from input tensor
        device, dtype = x.device, x.dtype
        
        # Ensure all tensors are on the same device
        q = q.to(device=device, dtype=dtype, non_blocking=False)
        k = k.to(device=device, dtype=dtype, non_blocking=False)
        v = v.to(device=device, dtype=dtype, non_blocking=False)
        
        # Force synchronization if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
        # Ensure all tensors are on the same device (debug print)
        print(f"Attention tensors - q: {q.device}, k: {k.device}, v: {v.device}")
    
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask - ensure same device first
        if mask is not None:
            # Ensure mask is on the same device as attn_weights - do this silently
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
    """Replace problematic attention modules with our improved implementation"""
    if hasattr(model.decoder, 'layers'):
        # Get device from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        for i, layer in enumerate(model.decoder.layers):
            if hasattr(layer, 'attn'):
                # Get dimensions from existing layer
                embed_dim = layer.attn.q_proj.out_features if hasattr(layer.attn, 'q_proj') else 1024
                num_heads = layer.attn.num_heads if hasattr(layer.attn, 'num_heads') else 8
                
                # Replace with our enhanced implementation
                new_attn = EnhancedDecoderAttention(embed_dim, num_heads)
                new_attn = new_attn.to(device=device, dtype=dtype)
                
                # Force synchronization to ensure module is on the correct device
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                # Replace the attention module
                layer.attn = new_attn
                print(f"Replaced decoder layer {i} with EnhancedDecoderAttention")
    
    return model
