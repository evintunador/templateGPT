import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.attention import SelfAttention
from modules.mlp import MLP

class Layer(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.second_norm = cfg.second_resid_norm
        self.dropout_rate = cfg.dropout_rate

        # attention connection
        self.pre_attn_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, cfg.norm_bias, 
            cfg.eps, 
            cfg.device
        )
        self.attn = SelfAttention(
            cfg.dim, cfg.head_dim,
            cfg.num_q_heads, cfg.num_kv_heads,
            cfg.max_seq_len,
            cfg.linear_bias,
            cfg.dropout_rate,
            cfg.device
        )
        if self.second_norm: 
            self.post_attn_norm = Norm(
                cfg.dim, 
                cfg.norm_type, 
                cfg.norm_affine, cfg.norm_bias, 
                cfg.eps, 
                cfg.device
            )

        # feedforward connection
        self.pre_mlp_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, cfg.norm_bias, 
            cfg.eps, 
            cfg.device
        ) 
        # ensures mlp_hidden_mult maintains the same parameter count if gated == true
        mult = cfg.mlp_hidden_mult * 2/3 if cfg.mlp_gated else cfg.mlp_hidden_mult
        self.mlp = MLP(
            cfg.dim, 
            int(cfg.dim * mult), 
            cfg.dim,
            cfg.mlp_nonlinearity,
            cfg.mlp_gated,
            cfg.linear_bias,
            cfg.dropout_rate,
            cfg.device
        )
        if self.second_norm: 
            self.post_mlp_norm = Norm(
                cfg.dim, 
                cfg.norm_type, 
                cfg.norm_affine, cfg.norm_bias, 
                cfg.eps, 
                cfg.device
            )

    def get_num_params(self):
        """ Return the number of parameters in the module """
        return sum(p.numel() for p in self.parameters())

    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs: dict = None,
        mask: torch.Tensor = None,
        training = False,
    ) -> torch.Tensor:
        dx = self.attn_connect(x, freqs, mask, training)
        x = x + dx
        x = x + self.mlp_connect(x, training)
        return x

    @log_io
    def attn_connect(
        self, 
        x: torch.Tensor, 
        freqs: dict, 
        mask: torch.Tensor, 
        training: bool,
    ) -> torch.Tensor:
        dx = self.attn(self.pre_attn_norm(x), freqs, mask, training)
        if training: dx = F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_attn_norm(dx)
        return dx

    @log_io
    def mlp_connect(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        dx = self.mlp(self.pre_mlp_norm(x), training)
        if training: dx = F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_mlp_norm(dx)
        return dx