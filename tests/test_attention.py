import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from modules.attention import SelfAttention, PrecomputeRotaryFrequencies


@pytest.fixture(params=["cuda", "mps", "cpu"])
def device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return device


@pytest.fixture
def cfg():  # removed device parameter
    """Configuration fixture."""
    class Config:
        dim = 64  # Model dimension
        head_dim = 16  # Dimension of each head
        num_q_heads = 4
        num_kv_heads = 4
        max_seq_len = 128
        linear_bias = False
        dropout_rate = 0.1
        theta = 10000.0  # removed device attribute
    return Config()


@pytest.fixture
def tcfg():
    """Training configuration fixture."""
    class TrainConfig:
        micro_batch_size = 2
    return TrainConfig()


@pytest.fixture
def self_attention(cfg):
    return SelfAttention(
        dim=cfg.dim,
        head_dim=cfg.head_dim,
        num_q_heads=cfg.num_q_heads,
        num_kv_heads=cfg.num_kv_heads,
        max_seq_len=cfg.max_seq_len,
        bias=cfg.linear_bias,
        dropout_rate=cfg.dropout_rate,
        device=cfg.device
    )


@pytest.fixture
def sample_data(cfg, tcfg):
    batch_size = tcfg.micro_batch_size
    seq_len = cfg.max_seq_len
    return {
        'x': torch.randn(batch_size, seq_len, cfg.dim, device=cfg.device),
        'q': torch.randn(batch_size, seq_len, cfg.num_q_heads, cfg.head_dim, device=cfg.device),
        'k': torch.randn(batch_size, seq_len, cfg.num_kv_heads, cfg.head_dim, device=cfg.device),
        'v': torch.randn(batch_size, seq_len, cfg.num_kv_heads, cfg.head_dim, device=cfg.device),
    }


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_rotary", [True, False])
def test_self_attention(cfg, tcfg, device, training, use_rotary):
    """Test the SelfAttention module under different configurations."""
    # Skip if device not available
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    # Initialize module
    module = SelfAttention(
        dim=cfg.dim,
        head_dim=cfg.head_dim,
        num_q_heads=cfg.num_q_heads,
        num_kv_heads=cfg.num_kv_heads,
        max_seq_len=cfg.max_seq_len,
        bias=cfg.linear_bias,
        dropout_rate=cfg.dropout_rate,
        device=device
    )

    # Prepare input data
    batch_size = tcfg.micro_batch_size
    seq_len = cfg.max_seq_len if training else cfg.max_seq_len // 2
    x = torch.randn(batch_size, seq_len, cfg.dim, device=device)

    # Prepare mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()

    # Optionally prepare rotary embeddings
    freqs = None
    if use_rotary:
        precompute_freqs = PrecomputeRotaryFrequencies(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.theta,
            device=device
        )
        freqs = precompute_freqs()

    # Forward pass
    output = module(x, freqs=freqs, mask=mask, training=training)

    # Assertions
    assert output.shape == (batch_size, seq_len, cfg.dim), "Output shape mismatch"
    assert output.device.type == device, "Output device mismatch"
    
    # Additional checks for different configurations
    if training:
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert all(p.grad is not None for p in module.parameters() if p.requires_grad)
    
    if use_rotary:
        # Verify rotary embeddings were applied (can check internal states or patterns)
        pass

@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_precompute_rotary_frequencies(cfg, device):
    """Test the PrecomputeRotaryFrequencies module."""
    # Skip if device not available
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    precompute_freqs = PrecomputeRotaryFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        theta=cfg.theta,
        device=device
    )
    freqs = precompute_freqs()

    expected_shape = (1, cfg.max_seq_len, 1, cfg.head_dim)
    assert 'sin' in freqs and 'cos' in freqs
    assert freqs['sin'].shape == expected_shape
    assert freqs['cos'].shape == expected_shape
    assert freqs['sin'].device.type == device
    assert freqs['cos'].device.type == device

@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_get_num_params(cfg, device):
    """Test the get_num_params method of SelfAttention module."""
    # Skip if device not available
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == 'mps' and not torch.backends.mps.is_available():
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
        device=device
    )

    # Get number of parameters
    num_params = module.get_num_params()

    # Assertions
    assert isinstance(num_params, int), "num_params should be an integer"
    assert num_params > 0, "num_params should be positive"