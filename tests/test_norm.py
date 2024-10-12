import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from modules.norm import Norm


@pytest.fixture
def cfg():
    """Configuration fixture for model parameters."""
    class Config:
        dim = 64
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Config()


@pytest.fixture
def tcfg():
    """Training configuration fixture."""
    class TrainConfig:
        micro_batch_size = 2
        max_seq_len = 128
    return TrainConfig()


def test_norm_rms(cfg, tcfg):
    """Test the Norm module with RMSNorm."""
    module = Norm(
        dim=cfg.dim,
        norm_type='RMSNorm',
        affine=True,
        bias=True,
        eps=1e-6,
        device=cfg.device
    )

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."


def test_norm_layer(cfg, tcfg):
    """Test the Norm module with LayerNorm."""
    module = Norm(
        dim=cfg.dim,
        norm_type='LayerNorm',
        affine=True,
        bias=True,
        eps=1e-6,
        device=cfg.device
    )

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."


def test_norm_cosine(cfg, tcfg):
    """Test the Norm module with CosineNorm."""
    module = Norm(
        dim=cfg.dim,
        norm_type='CosineNorm',
        affine=True,
        bias=True,
        eps=1e-6,
        device=cfg.device
    )

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."


def test_norm_affine_false(cfg, tcfg):
    """Test the Norm module with affine=False and bias=False."""
    module = Norm(
        dim=cfg.dim,
        norm_type='RMSNorm',
        affine=False,
        bias=False,
        eps=1e-6,
        device=cfg.device
    )

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."
    assert len(list(module.parameters())) == 0, "No parameters should be created when affine=False and bias=False."


def test_norm_affine_false_bias_true(cfg, tcfg, caplog):
    """Test the Norm module with affine=False and bias=True."""
    module = Norm(
        dim=cfg.dim,
        norm_type='RMSNorm',
        affine=False,
        bias=True,
        eps=1e-6,
        device=cfg.device
    )

    x = torch.randn(tcfg.micro_batch_size, tcfg.max_seq_len, cfg.dim).to(cfg.device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."
    assert len(list(module.parameters())) == 0, "No parameters should be created when affine=False and bias=True."