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

class MQA(LoggingModule): # multi-query self-attention https://arxiv.org/abs/1911.02150
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
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=bias)
        
        # this flag designates Wo to have a different parameter initialization as defined in model.py
        self.Wo.GPT_scale_init = 1
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        kv_cache: dict = None,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if (cache_len is not None) & (kv_cache is not None): # if we're performing inference and using kv caching. it'll init at 0
            kv_cache['k'][:batch_size, cache_len : cache_len + seq_len] = xk
            kv_cache['v'][:batch_size, cache_len : cache_len + seq_len] = xv

            keys = kv_cache['k'][:batch_size, : cache_len + seq_len]
            values = kv_cache['v'][:batch_size, : cache_len + seq_len]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv
        queries = xq # for sake of keeping the naming scheme consistent

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            keys, values = self.match_headcount(keys, values) # (bs, cache_len + seq_len, num_q_heads, head_dim)

        queries = queries.transpose(1, 2)  # (bs, num_q_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)
        values = values.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)

        if self.device == 'cuda' or self.device == 'mps': # perform flash attention if we've got a GPU
            scores = self.flash_attention(queries, keys, values)
        else: # otherwise we'll do regular attention
            logits = self.attend(queries, keys, training)
            if mask is not None:
                logits = logits + mask  # (bs, num_q_heads, seq_len, cache_len + seq_len)
            scores = self.calc_output(logits, values, training) # (batch_size, n_heads, seq_len, head_dim)

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
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(self, keys: torch.Tensor, values: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        keys = torch.repeat_interleave(keys, self.num_q_heads // self.num_kv_heads, dim=2)
        values = torch.repeat_interleave(values, self.num_q_heads // self.num_kv_heads, dim=2)
        return keys, values

    @log_io
    def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=True):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    @log_io
    def attend(self, queries: torch.Tensor, keys: torch.Tensor, training: bool) -> torch.Tensor:
        return torch.matmul(queries, keys.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, values: torch.Tensor, training: bool) -> torch.Tensor:
        batch_size, _, seq_len, _ = logits.shape
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        return scores @ values # (batch_size, n_heads, seq_len, head_dim)