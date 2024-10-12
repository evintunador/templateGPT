import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class Mish(nn.Module):
    """
    Mish is an alternative to ReLU and its variants proposed in 2019.
    Reference: https://arxiv.org/abs/1908.08681

    I have no intention of using it; rather this nn.Module is designed to demonstrate how
    you would go about creating your own custom nonlinearity and add it to the list of options
    """
    def forward(self, x):
        return x * torch.tanh(torch.log1p(torch.exp(x)))

class MLP(LoggingModule):
    """
    Multilayer Perceptron (MLP) module with optional gating and dropout.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        nonlinearity (str): Name of the nonlinearity to use ('GeLU', 'SiLU', 'ReLU', 'Mish').
        gated (bool): Whether to use a gating mechanism.
        bias (bool): Whether to include bias in linear layers.
        dropout_rate (float): Dropout rate for regularization.
        device (str or torch.device): Device to run the module on.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: str = 'SiLU',
        gated: bool = True,
        bias: bool = False,
        dropout_rate: float = 0.1,
        device = None
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gated = gated
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

        # the up, down, and (optional) gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=bias, device=device)
        if gated: 
            self.Wgate = nn.Linear(input_dim, hidden_dim, bias=bias, device=device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=bias, device=device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1

        # Mapping nonlinearity names to modules
        self.nonlinearities = {
            "GeLU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU(),
            "Mish": Mish(), 
        }
        # Set the nonlinearity
        self.nonlinearity = self.nonlinearities.get(nonlinearity)
        if self.nonlinearity is None:
            self.nonlinearity = nn.SiLU()
            print(f'Nonlinearity "{nonlinearity}" not found. Defaulting to SiLU.')

    def get_num_params(self):
        """
        Calculate the total number of parameters in the module.

        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
        
    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            training (bool): If True, apply dropout.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        hidden = self.up_proj(x) * self.gate_proj(x) if self.gated else self.up_proj(x) 
            # (batch_size, seq_len, hidden_dim)

        # Apply dropout if in training mode
        if training: 
            hidden = F.dropout(hidden, self.dropout_rate) # (batch_size, seq_len, hidden_dim)

        return self.down_proj(hidden) # (batch_size, seq_len, output_dim)
    
    @log_io
    def up_proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Up projection of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        return self.Wup(x)
    
    @log_io
    def gate_proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gate projection of the input tensor. 
        Determines how much of the up projection to let through (assuming ReLU-like nonlinearity).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        return self.nonlinearity(self.Wgate(x))
    
    @log_io
    def down_proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Down projection of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        return self.Wdown(x)