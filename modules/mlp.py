import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class MLP(LoggingModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: str = 'GeLU',
        gated: bool = True,
        bias: bool = False, # i Stan Llama and set bias to false
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        # the up, down, and (optional) gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias)
        self.gated = gated
        if gated: self.Wgate = nn.Linear(input_dim, hidden_dim, bias)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias)

        # Mapping norm types to their respective methods
        self.nonlinearities = {
            "GeLU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU(),
        }
        # Ensure the specified norm type exists, default to GeLU if not found
        if nonlinearity not in self.nonlinearities:
            self.nonlinearity = nn.SiLU
            print(f'nonlinearity {nonlinearity} not found. defaulting to SiLU')
        else:
            self.nonlinearity = self.nonlinearities[nonlinearity]
        
    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.gated:
            hidden_neurons = self.nonlinearity(self.Wgate(x)) * self.Wup(x)
        else:
            hidden_neurons = self.nonlinearity(self.Wup(x))
        if training: hidden_neurons = F.dropout(hidden_neurons, self.dropout_rate)
        return self.Wdown(hidden_neurons)