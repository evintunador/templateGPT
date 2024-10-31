import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from random import randint

from modules.norm import Norm
from modules.mlp import MLP
from modules.attention import SelfAttention, PrecomputeRotaryFrequencies
from modules.layer import Layer
from modules.model import Model
from config import ModelConfig, TrainConfig

@pytest.fixture
def cfg():
    return ModelConfig()

@pytest.fixture
def tcfg():
    return TrainConfig()

@pytest.fixture
def check_device(request):
    """Helper fixture to skip tests if the requested device is not available."""
    device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return device

@pytest.mark.parametrize("norm_type", ["RMSNorm", "LayerNorm", "CosineNorm"])
@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_norm(cfg, tcfg, norm_type, check_device):
    """Test the Norm module"""
    device = check_device

    module = Norm(
        dim=cfg.dim,
        norm_type=norm_type,
        affine=cfg.norm_affine,
        bias=cfg.norm_bias,
        eps=cfg.eps,
        device=device
    )

    x = torch.randn(tcfg.micro_batch_size, cfg.max_seq_len, cfg.dim).to(device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."
    assert output.device.type == device, "Output device mismatch"

@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("nonlinearity", ["GeLU", "SiLU", "ReLU", "Mish"])
@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_mlp_gated(cfg, tcfg, gated, nonlinearity, check_device):
    """Test the MLP module"""

    module = MLP(
        input_dim=cfg.dim,
        hidden_dim=int(cfg.dim * cfg.mlp_hidden_mult * 2 / 3),
        output_dim=cfg.dim,
        nonlinearity=nonlinearity,
        gated=gated,
        bias=cfg.linear_bias,
        dropout_rate=cfg.dropout_rate,
        device=check_device
    )

    x = torch.randn(tcfg.micro_batch_size, cfg.max_seq_len, cfg.dim).to(check_device)
    output = module(x)

    assert output.shape == x.shape, "Output shape should match input shape."
    assert output.device.type == check_device, "Output device mismatch"

@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_precompute_rotary_frequencies(cfg, check_device):
    """Test the PrecomputeRotaryFrequencies module."""
    device = check_device

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

@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_rotary", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_self_attention(cfg, tcfg, check_device, training, use_rotary, use_mask):
    """Test the SelfAttention module under different configurations."""
    device = check_device

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
    seq_len = cfg.max_seq_len if training else randint(1, cfg.max_seq_len)
    x = torch.randn(tcfg.micro_batch_size, seq_len, cfg.dim, device=device)

    # Prepare mask
    mask = None
    if use_mask:
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
    assert output.shape == (tcfg.micro_batch_size, seq_len, cfg.dim), "Output shape mismatch"
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

@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_rotary", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("use_second_norm", [True, False])
@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_layer(cfg, tcfg, check_device, training, use_rotary, use_mask, use_second_norm):
    """Test the transformer layer module under different configurations."""
    device = check_device

    # since Layer takes in the whole config we've gotta change values
    cfg.second_resid_norm = use_second_norm
    cfg.device = device

    # Initialize module
    module = Layer(cfg)

    # Prepare input data
    seq_len = cfg.max_seq_len if training else randint(1, cfg.max_seq_len)
    x = torch.randn(tcfg.micro_batch_size, seq_len, cfg.dim, device=device)

    # Prepare mask
    mask = None
    if use_mask:
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
    assert output.shape == (tcfg.micro_batch_size, seq_len, cfg.dim), "Output shape mismatch"
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

@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("pos_enc_type", ['learnable', 'Sinusoidal', 'RoPE'])
@pytest.mark.parametrize("out_weight_share", [True, False])
@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_model(cfg, tcfg, check_device, training, num_layers, pos_enc_type, out_weight_share):
    """Test the entire model under different configurations."""
    device = check_device

    # since Model takes in the whole config we've gotta change values
    cfg.num_layers = num_layers
    cfg.pos_enc_type = pos_enc_type
    cfg.out_weight_share = out_weight_share
    cfg.device = device

    # Initialize module
    module = Model(cfg)

    # Prepare input data
    seq_len = cfg.max_seq_len if training else randint(1, cfg.max_seq_len)
    input_token_ids = torch.randint(1, cfg.vocab_len, size = (tcfg.micro_batch_size, seq_len)).to(device)
    target_token_ids = None
    if training:
        target_token_ids = torch.randint_like(input_token_ids, cfg.vocab_len).to(device)

    # Forward pass
    output, loss = module(input_token_ids, target_token_ids)

    # Assertions
    assert output.shape == (tcfg.micro_batch_size, seq_len, cfg.vocab_len), "Output shape mismatch"
    assert output.device.type == device, "Output device mismatch"
    
    # Additional checks for different configurations
    if training:
        # Test gradient flow
        loss.backward()
        assert all(p.grad is not None for p in module.parameters() if p.requires_grad)