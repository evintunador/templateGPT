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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        # the up, down, and (optional) gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=bias, device=device)
        self.gated = gated
        if gated: self.Wgate = nn.Linear(input_dim, hidden_dim, bias=bias, device=device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=bias, device=device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1

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

    def get_num_params(self):
        """ Return the number of parameters in the module """
        return sum(p.numel() for p in self.parameters())
        
    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.gated:
            hidden_neurons = self.nonlinearity(self.Wgate(x)) * self.Wup(x)
        else:
            hidden_neurons = self.nonlinearity(self.Wup(x))
        if training: hidden_neurons = F.dropout(hidden_neurons, self.dropout_rate)
        return self.Wdown(hidden_neurons)