import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from modules.mlp import MLP


@pytest.fixture
def cfg():
    """Configuration fixture."""
    class Config:
        dim = 64
        mlp_hidden_mult = 4
        linear_bias = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Config()


@pytest.fixture
def tcfg():
    """Training configuration fixture."""
    class TrainConfig:
        micro_batch_size = 2
        max_seq_len = 128
    return TrainConfig()


def test_mlp_gated(cfg, tcfg):
    """Test the MLP module with gating enabled."""
    module = MLP(
        input_dim=cfg.dim,
        hidden_dim=int(cfg.dim * cfg.mlp_hidden_mult * 2 / 3),
        output_dim=cfg.dim,
        nonlinearity='GeLU',
        gated=True,
        bias=cfg.linear_bias,
        dropout_rate=0.1,
        device=cfg.device
    ).to(cfg.device)

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x, training=True)

    assert output.shape == x.shape, "Output shape should match input shape."


def test_mlp_non_gated(cfg, tcfg):
    """Test the MLP module with gating disabled."""
    module = MLP(
        input_dim=cfg.dim,
        hidden_dim=cfg.dim * cfg.mlp_hidden_mult,
        output_dim=cfg.dim,
        nonlinearity='ReLU',
        gated=False,
        bias=True,
        dropout_rate=0.1,
        device=cfg.device
    ).to(cfg.device)

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x, training=True)

    assert output.shape == x.shape, "Output shape should match input shape."