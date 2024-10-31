import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from tools import UnifiedDataLoader, split_dataset, get_data_loaders, torcherize_batch, import_from_nested_path, run_in_directory
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

@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_unified_dataloader(streaming, batch_size):
    """Test the UnifiedDataLoader with different configurations."""
    # Create a mock dataset
    mock_data = [{'text': f'text_{i}'} for i in range(10)]
    
    # Test initialization
    dataloader = UnifiedDataLoader(mock_data, streaming=streaming, batch_size=batch_size)
    
    # Test iteration
    batch = next(iter(dataloader))
    assert len(batch) <= batch_size
    assert isinstance(batch[0], str)
    
    if not streaming:
        # Test indexing for non-streaming
        assert dataloader[0] == 'text_0'
        with pytest.raises(IndexError):
            _ = dataloader[len(mock_data)]

@pytest.mark.parametrize("dataset_name", ['noanabeshima/TinyStoriesV2', 'HuggingFaceFW/fineweb'])
@pytest.mark.parametrize("streaming", [True, False])
def test_get_data_loaders(dataset_name, streaming):
    """Test the data loader creation function."""
    train_loader, val_loader = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=1,
        streaming=streaming
    )
    
    assert isinstance(train_loader, UnifiedDataLoader)
    assert isinstance(val_loader, UnifiedDataLoader)
    
    # Test basic iteration
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    assert isinstance(train_batch, list)
    assert isinstance(val_batch, list)
    assert len(train_batch) == 1
    assert len(val_batch) == 1

@pytest.mark.parametrize("check_device", ["cuda", "mps", "cpu"], indirect=True)
def test_torcherize_batch(cfg, check_device):
    """Test the batch tensorization function."""
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, bos=True, eos=True, pad=None):
            return [1] * pad  # Simple mock that returns a list of 1s
    
    tokenizer = MockTokenizer()
    batch = ["test text 1", "test text 2"]
    
    x, y = torcherize_batch(
        tokenizer=tokenizer,
        batch=batch,
        max_seq_len=cfg.max_seq_len,
        device=check_device
    )
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.device.type == check_device
    assert y.device.type == check_device
    assert x.shape == (len(batch), cfg.max_seq_len)
    assert y.shape == (len(batch), cfg.max_seq_len)

def test_import_from_nested_path():
    """Test the dynamic import functionality."""
    # Test with existing module
    imported = import_from_nested_path(['tools'], 'tools', ['UnifiedDataLoader'])
    assert 'UnifiedDataLoader' in imported
    assert imported['UnifiedDataLoader'] == UnifiedDataLoader
    
    # Test with non-existent module
    imported = import_from_nested_path(['nonexistent'], 'module', ['function'])
    assert not imported

def test_run_in_directory():
    """Test the directory context manager."""
    original_dir = os.getcwd()
    
    def test_func():
        return os.getcwd()
    
    # Test with existing directory
    result = run_in_directory(test_func, '..')
    assert result != original_dir
    assert os.getcwd() == original_dir  # Should be back to original
    
    # Test with nested directories
    result = run_in_directory(test_func, '../..')
    assert result != original_dir
    assert os.getcwd() == original_dir  # Should be back to original