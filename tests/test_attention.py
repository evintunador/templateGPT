import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from modules.attention import SelfAttention, PrecomputeRotaryFrequencies


@pytest.fixture
def cfg(device):
    """Configuration fixture."""
    class Config:
        dim = 64  # Model dimension
        head_dim = 16  # Dimension of each head
        num_q_heads = 4
        num_kv_heads = 4
        max_seq_len = 128
        linear_bias = False
        dropout_rate = 0.1
        device = None
        theta = 10000.0
    config = Config()
    config.device = device
    return config


@pytest.fixture
def tcfg():
    """Training configuration fixture."""
    class TrainConfig:
        micro_batch_size = 2
    return TrainConfig()


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_self_attention_training(cfg, tcfg):
    """Test the SelfAttention module in training mode on multiple devices."""
    # Check if the device is available
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if cfg.device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    

    # Initialize the module
    module = SelfAttention(
        dim=cfg.dim,
        head_dim=cfg.head_dim,
        num_q_heads=cfg.num_q_heads,
        num_kv_heads=cfg.num_kv_heads,
        max_seq_len=cfg.max_seq_len,
        bias=cfg.linear_bias,
        dropout_rate=cfg.dropout_rate,
        device=cfg.device
    )

    # Prepare input data
    batch_size = tcfg.micro_batch_size
    seq_len = cfg.max_seq_len
    x = torch.randn(batch_size, seq_len, cfg.dim, device=cfg.device)

    # Prepare freqs and mask
    precompute_freqs = PrecomputeRotaryFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        theta=cfg.theta,
        device=cfg.device
    )
    freqs = precompute_freqs()

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=cfg.device).tril()

    # Forward pass
    output = module(x, freqs=freqs, mask=mask, training=True)

    # Assertions
    assert output.shape == (batch_size, seq_len, cfg.dim), "Output shape mismatch"
    assert output.device.type == cfg.device, "Output device mismatch"


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_self_attention_inference(cfg, tcfg):
    """Test the SelfAttention module in inference mode on multiple devices."""
    # Check if the device is available
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if cfg.device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    # Initialize the module
    module = SelfAttention(
        dim=cfg.dim,
        head_dim=cfg.head_dim,
        num_q_heads=cfg.num_q_heads,
        num_kv_heads=cfg.num_kv_heads,
        max_seq_len=cfg.max_seq_len,
        bias=cfg.linear_bias,
        dropout_rate=cfg.dropout_rate,
        device=cfg.device
    )

    # Prepare input data (shorter sequence)
    batch_size = tcfg.micro_batch_size
    seq_len = cfg.max_seq_len // 2
    x = torch.randn(batch_size, seq_len, cfg.dim, device=cfg.device)

    # Prepare freqs and mask
    precompute_freqs = PrecomputeRotaryFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        theta=cfg.theta,
        device=cfg.device
    )
    freqs = precompute_freqs()

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=cfg.device).tril()

    # Forward pass
    output = module(x, freqs=freqs, mask=mask, training=False)

    # Assertions
    assert output.shape == (batch_size, seq_len, cfg.dim), "Output shape mismatch"
    assert output.device.type == cfg.device, "Output device mismatch"