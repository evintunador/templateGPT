"""
THIS FILE DOES NOT CURRENTLY WORK. IT'S JUST A FIRST DRAFT WRITTEN BY CLAUDE
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from train import estimate_loss, scheduler_lambda, get_optimizer, save_model
from modules.model import Model
from config import ModelConfig, TrainConfig
import tempfile
import shutil

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

@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def encode(self, text):
            return torch.randint(0, 100, (len(text),))
    return MockTokenizer()

@pytest.fixture
def mock_data_loader():
    return iter([["test sentence 1", "test sentence 2"] for _ in range(10)])

@pytest.fixture
def model(cfg):
    return Model(cfg)

@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_estimate_loss(model, mock_tokenizer, mock_data_loader, check_device):
    model = model.to(check_device)
    losses = estimate_loss(
        model,
        mock_tokenizer,
        mock_data_loader,
        mock_data_loader,
        eval_samples=2
    )
    
    assert 'train' in losses
    assert 'val' in losses
    assert losses['train'].shape == torch.Size([2])
    assert losses['val'].shape == torch.Size([2])
    assert not torch.isnan(losses['train']).any()
    assert not torch.isnan(losses['val']).any()

def test_scheduler_lambda(tcfg):
    # Test warmup phase
    lr = scheduler_lambda(tcfg.warmup_iters // 2)
    assert tcfg.lr_init < lr < tcfg.lr_max
    
    # Test cosine annealing phase
    lr = scheduler_lambda(tcfg.warmup_iters + 100)
    assert tcfg.lr_min <= lr <= tcfg.lr_max
    
    # Test final constant phase
    lr = scheduler_lambda(tcfg.max_iters - tcfg.final_flat_iters + 10)
    assert lr == tcfg.lr_min

@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_get_optimizer(model, tcfg, check_device):
    model = model.to(check_device)
    optimizer = get_optimizer(model, tcfg)
    
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['weight_decay'] == tcfg.weight_decay
    assert optimizer.param_groups[1]['weight_decay'] == 0.0

def test_save_model(model, cfg, tcfg):
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        original_trained_dir = 'trained'
        try:
            # Temporarily modify the save path
            os.makedirs(temp_dir, exist_ok=True)
            os.environ['TRAINED_DIR'] = temp_dir
            
            # Test regular save
            log_data = [[1, 0.5, 1000, 2.0, 2.1, 8.2, 0.001, 0.5]]
            save_model(model, cfg, tcfg, log_data, checkpoint=False)
            
            save_path = os.path.join(temp_dir, tcfg.model_name)
            assert os.path.exists(save_path)
            assert os.path.exists(os.path.join(save_path, 'model.pth'))
            assert os.path.exists(os.path.join(save_path, 'model_config.json'))
            assert os.path.exists(os.path.join(save_path, 'train_config.json'))
            assert os.path.exists(os.path.join(save_path, 'log_data.csv'))
            
            # Test checkpoint save
            save_model(model, cfg, tcfg, log_data, checkpoint=True)
            checkpoints = [d for d in os.listdir(save_path) if d.startswith('checkpoint-')]
            assert len(checkpoints) == 1
            
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)