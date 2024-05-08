import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class Norm(LoggingModule):
    def __init__(
        self, 
        dim: int, 
        norm_type: str = 'RMSNorm', 
        affine: bool = True, 
        bias: bool = True, 
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps

        # We start with ones for weight to keep the original scale initially, and zeros for bias.
        self.affine = affine
        self.bias = bias
        if affine:
            self.w = nn.Parameter(torch.ones(dim))
            if bias:
                self.b = nn.Parameter(torch.zeros(dim))
        elif bias:
            print('cannot have both affine==False and bias==True. Skipping bias')

        # Mapping norm types to their respective methods
        self.norm_type = norm_type
        self.norm_methods = {
            "RMSNorm": self.RMSNorm,
            "LayerNorm": self.LayerNorm,
            "CosineNorm": self.CosineNorm,
        }
        # Ensure the specified norm type exists, default to RMSNorm if not found
        if norm_type not in self.norm_methods:
            print(f'norm type {norm_type} not found. defaulting to RMSNorm')
            self.norm_type = "RMSNorm"

    @log_io
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_methods[self.norm_type](x)

        # Optionally apply the affine transformation
        if self.affine: 
            x = x * self.w
            if self.bias:
                x = x + self.b
            
        return x

    @log_io
    def CosineNorm(self, x):
        return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

    @log_io
    def LayerNorm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    @log_io
    def RMSNorm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)