import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class SelfAttention(LoggingModule):
    """
    A flexible self-attention module
    - Use a custom mask of your choice by passing in a mask tensor
    - Knows whether to use FlashAttention to speed things up (if you're on a GPU) or regular attention
    - Works for both training & inference (pass in training:bool to determine whether to perform dropout)
    - Optionally disable kv caching by just not passing in a kv_cache
    - Optionally disable Rotary Positional Encoding by just not passing in a freqs_cis tensor
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.dropout_rate = dropout_rate
        self.device = device

        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=bias)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=bias)
            # it would be more efficient to do one Wqkv & then split its output later but I prefer readability
        
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=bias)
        # this flag designates Wo to have a different parameter initialization as defined in model.py
        self.Wo.GPT_scale_init = 1

    def get_num_params(self):
        """ Return the number of parameters in the module """
        return sum(p.numel() for p in self.parameters())
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache_len: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x) 
            # (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads * head_dim)
            # where q vs kv can have a different number of heads

        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # you'll probably use RoPE, but if not then this can deactivate itself by just not inputting freqs_cis
        if freqs_cis is not None: 
            q, k = self.apply_rotary_emb(q, k, freqs_cis)

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

        # perform flash attention if we've got a GPU, or otherwise we'll do regular inefficient attention
        if self.device == 'cuda' or self.device == 'mps': 
            scores = self.flash_attention(q, k, v, mask, training)
        else:
            logits = self.attend(q, k, training)
            if mask is not None:
                logits = logits + mask  # (batch_size, num_q_heads, seq_len, cache_len + seq_len)
            scores = self.calc_output(logits, v, training) # (batch_size, n_heads, seq_len, head_dim)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (batch_size, seq_len, n_heads * head_dim)
        output = self.Wo(scores) # (batch_size, seq_len, dim)
        if training: output = F.dropout(output, self.dropout_rate)
        
        return output, kv_cache
    
    @log_io
    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @log_io
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(self, k: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_heads, dim=2)
        v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_heads, dim=2)
        return k, v

    @log_io
    def flash_attention(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        mask: Optional[torch.Tensor], 
        training: bool
    ) -> torch.tensor:
        """ 
        Flash-attention is a more efficient version of attention. Only gets called if you're using a GPU
        https://arxiv.org/abs/2205.14135
        https://arxiv.org/abs/2307.08691
        """
        if mask is not None:
            scores = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p = self.dropout_rate if training else 0.0
            )
        else:
            scores = F.scaled_dot_product_attention(
                q,k,v, 
                is_causal=False, 
                dropout_p = self.dropout_rate if training else 0.0
            )
        return scores

    @log_io
    def attend(self, q: torch.Tensor, k: torch.Tensor, training: bool) -> torch.Tensor:
        return torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, v: torch.Tensor, training: bool) -> torch.Tensor:
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        return scores @ v # (batch_size, n_heads, seq_len, head_dim)