import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class PrecomputeRotaryFrequencies(LoggingModule):
    """
    This class precomputes the RoPE frequencies based on the expected `max_seq_len` and `head_dim`.
    It uses real-valued arithmetic instead of complex numbers to ensure compatibility with MPS devices.

    Adapted from:
    https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

    Args:
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length expected.
        theta (float, optional): Base value for computing frequencies. Default is 10,000.
        device (str, optional): Device to store the frequencies on. Defaults to CUDA if available, else MPS, else CPU.
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0, device: str = None):
        super().__init__()
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        # Shape: (head_dim // 2)
        self.register_buffer('inv_freq', inv_freq)

    @log_io
    def forward(self):
        """
        Compute the cosine and sine frequencies for rotary positional encoding.

        Returns:
            dict: A dictionary containing 'cos' and 'sin' tensors, each of shape
            (1, max_seq_len, 1, head_dim).
        """
        # Compute position indices
        t = torch.arange(self.max_seq_len, device=self.device).type_as(self.inv_freq)  # Shape: (max_seq_len)

        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # Shape: (max_seq_len, head_dim // 2)

        # Concatenate frequencies to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)  # Shape: (max_seq_len, head_dim)

        # Compute cosine and sine embeddings
        freqs = {
            'cos': emb.cos()[None, :, None, :],  # Shape: (1, max_seq_len, 1, head_dim)
            'sin': emb.sin()[None, :, None, :]   # Shape: (1, max_seq_len, 1, head_dim)
        }
        return freqs


class SelfAttention(LoggingModule):
    """
    A flexible self-attention module.

    This module supports the following features:
    - Optional use of a custom mask
    - Automatic use of FlashAttention on CUDA/MPS devices for speed-up.
    - Works for both training and inference modes.
    - Optional Rotary Positional Encoding.

    Args:
        dim (int): Input and output dimension of the model.
        head_dim (int): Dimension of each attention head.
        num_q_heads (int): Number of query heads.
        num_kv_heads (int, optional): Number of key/value heads. Defaults to `num_q_heads`.
        max_seq_len (int, optional): Maximum sequence length.
        bias (bool): Whether to include bias in linear projections.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
        device (str, optional): Device to run the module on. Defaults to CUDA if available, else MPS, else CPU.
    """
    def __init__(
        self, 
        dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        bias: bool,
        dropout_rate: float = 0.1,
        device = None
    ):
        super().__init__()
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Define linear projections for queries, keys, and values
        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=bias, device=self.device)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias, device=self.device)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias, device=self.device)
            # it would be more efficient to do one Wqkv & then split its output later but I prefer readability
        
        # Output projection that mixes all the attention heads back together
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=bias, device=self.device)
        # this flag designates Wo to have a different parameter initialization as defined in model.py
        self.Wo.GPT_scale_init = 1

    def get_num_params(self):
        """
        Get the total number of parameters in the module.

        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
    
    @log_io
    def forward(self,
        x: torch.Tensor,
        freqs: dict = None,
        mask: torch.Tensor = None,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs (dict, optional): Precomputed rotary positional encoding frequencies.
            mask (torch.Tensor, optional): Attention mask tensor.
            training (bool, optional): Whether the module is in training mode.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections for queries, keys, and values
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            # shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads * head_dim)
            # where q versus k&v can have a different number of heads

        # Reshape projections to separate heads
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # you'll probably use RoPE, but if not then this can deactivate itself by just not inputting freqs
        if freqs is not None:
            if training: # training & inference work differently bc of kv caching
                q, k = self.apply_rotary_pos_emb(q, k, freqs['sin'].to(self.device), freqs['cos'].to(self.device))
            else:
                sin = freqs['sin'][:, :seq_len, :, :].to(self.device) 
                cos = freqs['cos'][:, :seq_len, :, :].to(self.device) # (1, seq_len, 1, head_dim // 2)
                q, k = self.apply_rotary_pos_emb(q, k, sin, cos)

        # adjusts keys and values to match the query heads count
        if self.num_kv_heads != self.num_q_heads:
            k, v = self.match_headcount(k, v) # (batch_size, cache_len + seq_len, num_q_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_q_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_q_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_q_heads, seq_len, head_dim)
        
        # perform flash attention if we've got a GPU
        if self.device in ['cuda', 'mps']: 
            scores = self.flash_attention(q, k, v, mask, training) # (batch_size, n_heads, seq_len, head_dim)
        else: 
            # otherwise self.device == 'cpu', then we have to manually calculate attention
            scores = self.regular_attention(q, k, v, mask, training) # (batch_size, n_heads, seq_len, head_dim)

        # Combine heads
        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) 
            # (batch_size, seq_len, n_heads * head_dim)
        output = self.Wo(scores) # (batch_size, seq_len, dim)
        if training: 
            output = F.dropout(output, self.dropout_rate)
        
        return output
    
    @log_io
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Apply rotary positional embeddings to queries and keys.

        Args:
            q (torch.Tensor): Queries tensor.
            k (torch.Tensor): Keys tensor.
            sin (torch.Tensor): Sine frequencies tensor.
            cos (torch.Tensor): Cosine frequencies tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        # the real component is simple
        q_real = q * cos # (batch_size, seq_len, num_q_heads, head_dim)
        k_real = k * cos # (batch_size, seq_len, num_q_heads, head_dim)
        
        # the imaginary component requires we mess with the order (hence rotate_half)
        q_imag = self.rotate_half(q) * sin
        k_imag = self.rotate_half(k) * sin

        # and here are our successfully rotates q&k vectors
        q = q_real + q_imag
        k = k_real + k_imag

        return q, k
        
    @log_io
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the tensor for rotary embeddings.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    @log_io
    def match_headcount(self, k: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Adjust key/value heads to match the number of query heads.

        Args:
            k (torch.Tensor): Keys tensor.
            v (torch.Tensor): Values tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted keys and values.
        """
        repeat_times = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_times, dim=2)
        v = v.repeat_interleave(repeat_times, dim=2)
        return k, v

    @log_io
    def flash_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor = None, 
        training: bool = False
    ) -> torch.tensor:
        """
        an efficient implementation of attention that reduces memory usage and computation time.
        Reference:
        - https://arxiv.org/abs/2205.14135
        - https://arxiv.org/abs/2307.08691

        Args:
            q (torch.Tensor): Queries tensor.
            k (torch.Tensor): Keys tensor.
            v (torch.Tensor): Values tensor.
            mask (torch.Tensor, optional): Attention mask.
            training (bool, optional): Whether the module is in training mode.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        dropout_p = self.dropout_rate if training else 0.0
        if mask is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

    @log_io
    def regular_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor = None, 
        training: bool = False
    ) -> torch.tensor:
        """
        Compute attention using standard implementation.

        Args:
            q (torch.Tensor): Queries tensor.
            k (torch.Tensor): Keys tensor.
            v (torch.Tensor): Values tensor.
            mask (torch.Tensor, optional): Attention mask.
            training (bool, optional): Whether the module is in training mode.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        # Compute attention logits (compare queries & keys)
        logits = self.attend(q, k, training) # (batch_size, num_q_heads, seq_len, seq_len)
            
        # Apply mask if provided
        if mask is not None: 
            # Adjust mask shape for inference
            if mask.shape[0] != mask.shape[1]: 
                mask = self.adjust_inference_mask(mask, q.shape[1], q.shape[2], k.shape[2])

            # here we mask out all the future-values
            logits = logits.masked_fill(~mask, float('-inf'))  # (batch_size, num_q_heads, seq_len, seq_len)

        # Compute attention scores (grab the relevant values that correspond to the attention logits)
        return self.project_values(logits, v, training) # (batch_size, n_heads, seq_len, head_dim)
            
    @log_io
    def attend(self, q: torch.Tensor, k: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Compute scaled dot-product attention logits.

        Args:
            q (torch.Tensor): Queries tensor.
            k (torch.Tensor): Keys tensor.

        Returns:
            torch.Tensor: Attention logits.
        """
        scale = self.head_dim ** -0.5
        return torch.matmul(q, k.transpose(-2, -1)) * scale # (batch_size, num_q_heads, seq_len, seq_len)

    @log_io
    def adjust_inference_mask(
        self, 
        mask: torch.Tensor, 
        num_heads: int, 
        q_seq_len: int, 
        k_seq_len: int
    ) -> torch.tensor:
        """
        Adjust the attention mask shape for inference.

        Args:
            mask (torch.Tensor): Original mask tensor.
            num_heads (int): Number of attention heads.
            q_seq_len (int): Sequence length of queries.
            k_seq_len (int): Sequence length of keys.

        Returns:
            torch.Tensor: Adjusted mask tensor.
        """
        return mask.unsqueeze(1).unsqueeze(1).expand(-1, num_heads, q_seq_len, k_seq_len)
    
    @log_io
    def project_values(
        self, 
        logits: torch.Tensor, 
        v: torch.Tensor, 
        training: bool = False
    ) -> torch.Tensor:
        """
        Compute attention output by applying softmax and projecting values.

        Args:
            logits (torch.Tensor): Attention logits.
            v (torch.Tensor): Values tensor.
            training (bool, optional): Whether the module is in training mode.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        scores = F.softmax(logits, dim=-1)
        if training: 
            scores = F.dropout(scores, self.dropout_rate)
        return scores @ v # (batch_size, n_heads, seq_len, head_dim)