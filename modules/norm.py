import torch
import torch.nn as nn

from modules.logging import LoggingModule, log_io

class Norm(LoggingModule):
    """
    Normalization module with support for RMSNorm, LayerNorm, and CosineNorm.

    Args:
        dim (int): Dimension of the input tensor.
        norm_type (str): Type of normalization to apply ('RMSNorm', 'LayerNorm', 'CosineNorm').
        affine (bool): Whether to include an affine transformation (scaling).
        bias (bool): Whether to include a bias term in the affine transformation.
        eps (float): A small value added to the denominator for numerical stability.
        device (str or torch.device): Device to run the module on.
    """

    def __init__(
        self,
        dim: int,
        norm_type: str,
        affine: bool = True,
        bias: bool = True,
        eps: float = 1e-6,
        device = None
    ):
        super().__init__()
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        self.eps = eps
        self.affine = affine
        self.bias = bias

        # Initialize parameters if affine is True
        if self.affine:
            self.w = nn.Parameter(torch.ones(dim)).to(device)
            if self.bias:
                self.b = nn.Parameter(torch.zeros(dim)).to(device)
        elif self.bias:
            print('Cannot have affine=False and bias=True. Skipping bias.')
            self.bias = False  # Ensure bias is set to False

        # Mapping norm types to their respective methods
        self.norm_methods = {
            "RMSNorm": self.rms_norm,
            "LayerNorm": self.layer_norm,
            "CosineNorm": self.cosine_norm,
        }
        # Ensure the specified norm type exists, default to RMSNorm if not found
        if norm_type not in self.norm_methods:
            print(f'Norm type "{norm_type}" not found. Defaulting to RMSNorm.')
            self.norm_type = "RMSNorm"
        else:
            self.norm_type = norm_type

    def get_num_params(self) -> int:
        """
        Calculate the total number of parameters in the module.

        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    @log_io
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the normalization module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Apply the selected normalization method
        x = self.norm_methods[self.norm_type](x)

        # Optionally apply the affine transformation
        if self.affine:
            x = x * self.w
            if self.bias:
                x = x + self.b

        return x

    @log_io
    def cosine_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Cosine Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Cosine-normalized tensor.
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm

    @log_io
    def layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Layer-normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    @log_io
    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: RMS-normalized tensor.
        """
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)