import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class PrecomputeRotaryFrequencies(LoggingModule):
    """
    A class to pre-compute RoPE frequencies based on the expected max_seq_len and head_dim
    Executed with real-valued complex arithmetic rather than using torch's built-in complex64 type in order to ensure MPS compatibility
    Code heavily edited from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    """
    def __init__(self, head_dim, max_seq_len: int, theta: float = 10_000.0, learnable: bool = False, 
                 device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.learnable = learnable
        
        inv_freq = 1. / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)) # shape (head_dim // 2)
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer('inv_freq', inv_freq)

    @log_io
    def forward(self):
        t = torch.arange(self.max_seq_len).type_as(self.inv_freq) # shape (max_seq_len)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (max_seq_len, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, head_dim)
        freqs = {
            'cos': emb.cos()[None, :, None, :], # for real coefficients. shape (1, max_seq_len, 1, head_dim // 2)
            'sin': emb.sin()[None, :, None, :] # for imaginary coefficients. shape (1, max_seq_len, 1, head_dim // 2)
        }
        return freqs

class SelfAttention(LoggingModule):
    """
    A flexible self-attention module
    - Use a custom mask of your choice by passing in a mask tensor of dtype torch.bool & shape (keys_seq_len, queries_seq_len)
    - Knows whether to use FlashAttention to speed things up (if you've got a cuda/mps GPU) or regular attention (for CPUs)
    - Works for both training & inference (pass in training:bool to determine whether to perform dropout)
    - Optionally disable kv caching by just not passing in a kv_cache dictionary
    - Optionally disable Rotary Positional Encoding by just not passing in a freqs dictionary
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
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.dropout_rate = dropout_rate
        self.device = device

        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=bias, device=device)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias, device=device)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias, device=device)
            # it would be more efficient to do one Wqkv & then split its output later but I prefer readability
        
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=bias, device=device)
        # this flag designates Wo to have a different parameter initialization as defined in model.py
        self.Wo.GPT_scale_init = 1

    def get_num_params(self):
        """ Return the number of parameters in the module """
        return sum(p.numel() for p in self.parameters())
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs: dict = None,
        mask: torch.Tensor = None,
        cache_len: int = None,
        kv_cache: dict = None,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x) 
            # (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads * head_dim)
            # where q vs kv can have a different number of heads

        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # you'll probably use RoPE, but if not then this can deactivate itself by just not inputting freqs
        if freqs is not None: 
            q, k = self.apply_rotary_pos_emb(q, k, freqs, cache_len)

        # if we're performing inference and using kv caching
        if (cache_len is not None) & (kv_cache is not None): 
            kv_cache['k'][:, cache_len : cache_len + seq_len, :] = k
            kv_cache['v'][:, cache_len : cache_len + seq_len, :] = v

            k = kv_cache['k'][:, : cache_len + seq_len, :] # (batch_size, cache_len + seq_len, num_kv_heads, head_dim)
            v = kv_cache['v'][:, : cache_len + seq_len, :]

        # adjusts keys and values to match the query heads count so that attention can be performed
        if self.num_kv_heads != self.num_q_heads:
            k, v = self.match_headcount(k, v) # (batch_size, cache_len + seq_len, num_q_heads, head_dim)

        q = q.transpose(1, 2)  # (batch_size, num_q_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_q_heads, cache_len + seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_q_heads, cache_len + seq_len, head_dim)

        # perform flash attention if we've got a GPU
        if self.device in ['cuda', 'mps']: 
            scores = self.flash_attention(q, k, v, mask, training)
        else: 
            # otherwise we'll do regular inefficient attention
            logits = self.attend(q, k, training) # (batch_size, num_q_heads, seq_len, cache_len + seq_len)
            if mask is not None: logits = logits.masked_fill(mask, float('-inf'))  # (batch_size, num_q_heads, seq_len, cache_len + seq_len)
            scores = self.calc_output(logits, v, training) # (batch_size, n_heads, seq_len, head_dim)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (batch_size, seq_len, n_heads * head_dim)
        output = self.Wo(scores) # (batch_size, seq_len, dim)
        if training: output = F.dropout(output, self.dropout_rate)
        
        return output, kv_cache
    
    @log_io
    def apply_rotary_pos_emb(self, q, k, freqs_cis, cache_len):
        """
        calculates rotary positional encodings using real-valued complex arithmetic
        """
        # selecting the correct frequencies given our kv caching. during training cache_len==0 and q.shape[1]==max_seq_len
        if cache_len is None: cache_len = 0
        cos = freqs_cis['cos'][:, cache_len : q.shape[1] + cache_len, :, :].to(q.device) # (1, seq_len, 1, head_dim // 2)
        sin = freqs_cis['sin'][:, cache_len : q.shape[1] + cache_len, :, :].to(q.device)

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
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    @log_io
    def match_headcount(self, k: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_heads, dim=2)
        v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_heads, dim=2)
        return k, v

    @log_io
    def flash_attention(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        mask: torch.Tensor = None, 
        training: bool = False
    ) -> torch.tensor:
        """ 
        Flash-attention is a more efficient version of attention. Only gets called if you're using a GPU
        https://arxiv.org/abs/2205.14135
        https://arxiv.org/abs/2307.08691
        
        this is a separate function despite being so simple only for the purpose of easy display in `test_modules.ipynb`
        """
        if mask is not None:
            # we flip the mask bool values because the function expects True -> "let the model see this token" and False -> "mask this token"
            return F.scaled_dot_product_attention(q, k, v, attn_mask = mask.logical_not() , dropout_p = self.dropout_rate if training else 0.0)
        else:
            return F.scaled_dot_product_attention(q,k,v, dropout_p = self.dropout_rate if training else 0.0)

    @log_io
    def attend(self, q: torch.Tensor, k: torch.Tensor, training: bool = False) -> torch.Tensor:
        return torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5) # (batch_size, num_q_heads, seq_len, cache_len + seq_len)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, v: torch.Tensor, training: bool = False) -> torch.Tensor:
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        return scores @ v # (batch_size, n_heads, seq_len, head_dim)