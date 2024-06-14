import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.mqa import precompute_freqs_cis
from modules.layer import Layer

class Model(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len + 3 # the 3 is the bos, eos, and padding tokens

        # the generate() function in inference.py references these values to build kv cache
        self.head_dim = cfg.head_dim 
        self.num_kv_heads = cfg.num_kv_heads
        
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0
        
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))
        self.final_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, 
            cfg.norm_bias, 
            cfg.eps
        )

        freqs_cis = precompute_freqs_cis(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.theta
        ).to(cfg.device)
        self.register_buffer('freqs_cis', freqs_cis)

        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), 
                          float("-inf"), 
                          device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len - 1) # ignore the padding token

    @log_io
    def forward(
        self, 
        input_token_ids: torch.Tensor, 
        cache_len: int = None,
        kv_cache: list = None,
        target_token_ids: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        input_token_ids = input_token_ids.to(self.device)
        if target_token_ids is not None:
            target_token_ids = target_token_ids.to(self.device)

        batch_size, seq_len = input_token_ids.shape

        if target_token_ids is not None: # if training
            assert input_token_ids.shape == target_token_ids.shape
            assert seq_len == self.max_seq_len
            mask = self.mask
            freqs_cis = self.freqs_cis
            training = True
            cache_len = None
        else: # if performing inference
            freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len]
            mask = self.mask[:seq_len, :seq_len]
            mask = torch.hstack([torch.zeros((seq_len, cache_len), device=self.device), mask])#.type_as(x)
            training = False

        # initialize first residual state
        x = self.token_embedder(input_token_ids) * self.scale # [batch_size, seq_len, dim]
        # run through the model's layers
        for i, layer in enumerate(self.layers):
            x, kv_cache_i = layer(
                x, 
                freqs_cis, 
                mask, 
                cache_len,
                kv_cache[i] if kv_cache is not None else None,
                training
            )
            # update the kv cache
            if kv_cache is not None: kv_cache[i] = kv_cache_i 
        # the final output of the model
        logits = self.final_norm(x) @ self.token_embedder.weight.t() # [batch_size, seq_len, vocab_len]

        if training:
            loss = self.criterion(
                logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )
        else:
            loss = None
            
        return logits, loss