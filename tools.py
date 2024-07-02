import torch

###########################################################
################ LOADING DATA #############################
###########################################################
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

def split_dataset(dataset, val_size: float = 0.0005):
    """
    a function that can split up an iterable dataset into train & val
    """
    # Calculate the number of validation samples
    val_size = int(1 / val_size)  # 1 out of every val_size samples

    def train_split():
        for i, example in enumerate(dataset):
            if i % val_size != 0:
                yield example

    def val_split():
        for i, example in enumerate(dataset):
            if i % val_size == 0:
                yield example

    return train_split(), val_split()

class UnifiedDataLoader(IterableDataset):
    """
    an iterable dataset meant to function the same whether you're streaming the data or have downloaded it
    """
    def __init__(self, dataset, streaming=True, batch_size=1):
        self.dataset = dataset
        self.streaming = streaming
        self.batch_size = batch_size
        self.iterator = iter(self.dataset) if streaming else None
        self.index = 0
        if not streaming:
            self.text_data = dataset['text']

    def __iter__(self):
        return self

    def __next__(self):
        if self.streaming:
            batch = [next(self.iterator)['text'] for _ in range(self.batch_size)]
        else:
            if self.index >= len(self.text_data):
                raise StopIteration
            end_idx = min(self.index + self.batch_size, len(self.text_data))
            batch = self.text_data[self.index:end_idx]
            self.index = end_idx
        
        return batch

    def __getitem__(self, idx):
        if self.streaming:
            raise IndexError("Streaming dataset doesn't support indexing")
        return self.dataset[idx]['text']

def get_data_loaders(dataset_name: str, batch_size: int = 1, streaming: bool = True, subset_name: str = None):
    # Create the datasets
    if dataset_name in ['noanabeshima/TinyStoriesV2']:
        train_dataset = load_dataset(dataset_name, split = 'train', streaming = streaming)
        val_dataset = load_dataset(dataset_name, split = 'validation', streaming = streaming)
    elif dataset_name in ['HuggingFaceFW/fineweb', 'HuggingFaceFW/fineweb-edu']:
        dataset = load_dataset(
            dataset_name, 
            name = 'sample-10BT' if subset_name is None else subset_name,
            streaming = streaming,
            split = 'train' # fineweb only has a 'train' split so we'll have to split it ourselves
        )
        if streaming:
            train_dataset, val_dataset = split_dataset(dataset)
        else:
            dataset_split = dataset.train_test_split(test_size=0.0005, seed=69, shuffle=False)
            train_dataset, val_dataset = dataset_split['train'], dataset_split['test']
    else:
        raise ValueError(f"dataset_name must be in ['noanabeshima/TinyStoriesV2', 'HuggingFaceFW/fineweb', 'HuggingFaceFW/fineweb-edu']\n"\
                        f"if you'd like to support more datasets then edit `get_data_loaders()` in `tools.py`")
    
    # Create the data loaders. UnifiedDataLoader is designed to work with both downloaded datasets & iterable streaming datasets
    train_data_loader = UnifiedDataLoader(train_dataset, streaming, batch_size)
    val_data_loader = UnifiedDataLoader(val_dataset, streaming, batch_size)
    
    return train_data_loader, val_data_loader

def torcherize_batch(
    tokenizer, 
    batch, 
    max_seq_len, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> (torch.Tensor, torch.Tensor):
    b = torch.zeros(len(batch), max_seq_len+1)
    for i, s in enumerate(batch):
        b[i] = torch.tensor(
            tokenizer.encode(s, bos=True, eos=True, pad=max_seq_len+1), 
            device=device
        )
    x, y = b[:,:max_seq_len], b[:, 1:]
    return x.to(torch.long), y.to(torch.long)

###########################################################
###################### DYNAMIC IMPORTING ##################
###########################################################

# allows us to import functions specific to a given model project, meaning you can change those functions in your project & stuff still works
import importlib
def import_from_nested_path(folders, file, items):
    try:
        # Construct the module path from a list of folders
        module_path = ".".join(folders) + "." + file
        
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Extract specific items (functions, classes, etc.)
        imported_items = {}
        for item in items:
            if hasattr(module, item):
                imported_items[item] = getattr(module, item)
            else:
                print(f"{item} is not available in {module_path}")
        return imported_items
                
    except ImportError as e:
        print(f"Failed to import module: {e}")

# Example usage
#imported_objects = import_from_nested_path(['tokenizers', 'bpe'], 'tokenizer', ['get_tokenizer'])
#get_tokenizer = imported_objects.get('get_tokenizer')

# a wrapper to force a given function to behave using a specified working directory rather than the current working directory
import os
def run_in_directory(func, path, *args, **kwargs):
    original_dir = os.getcwd()  # Save the current directory
    os.chdir(path)  # Change to the target directory
    try:
        result = func(*args, **kwargs)  # Execute the function
    finally:
        os.chdir(original_dir)  # Change back to the original directory
    return result

# Example usage
#def example_function():
    #print("Current Working Directory:", os.getcwd())

# Calling the function with a custom directory
#run_in_directory(example_function, "models/customGPT/")
    
###########################################################
############# SAVE / LOAD MODELS ##########################
###########################################################
import os
import json
from dataclasses import asdict
import time
import csv

def save_model(model, cfg, tcfg, log_data = None, checkpoint = False):
    if checkpoint == True:
        path = f'trained/{tcfg.model_name}/checkpoint-{time.strftime("%Y-%m-%d|%H-%M-%S")}'
    else:
        path = f'trained/{tcfg.model_name}'
    os.makedirs(path, exist_ok=True)
    
    if log_data is not None:
        # Save training data to CSV
        with open(f'{path}/log_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 
                'Learning Rate', 
                'Train Loss', 
                'Validation Loss', 
                'Perplexity', 
                'Time Elapsed'
            ])
            writer.writerows(log_data)
    
    # saving model
    torch.save(model.state_dict(), f'{path}/model.pth')
    
    # saving configs
    cfg_dict = asdict(cfg)
    with open(f'{path}/model_config.json', 'w') as f:
        json.dump(cfg_dict, f)
    tcfg_dict = asdict(tcfg)
    with open(f'{path}/train_config.json', 'w') as f:
        json.dump(tcfg_dict, f)

    print(f'model successfully saved to {path}')

def load_model(
    name: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
):
    model_name = f'trained/{name}'

    # the config
    from config import ModelConfig

    # Deserialize the JSON file back to a dictionary
    with open(f'{model_name}/model_config.json', 'r') as f:
        config_dict = json.load(f)
    
    # Convert the dictionary back to a Config object
    cfg = ModelConfig(**config_dict)
    cfg.device = device

    # the tokenizer
    imported_objects = import_from_nested_path(['custom_tokenizers', cfg.tokenizer], 'tokenizer', ['get_tokenizer'])
    get_tokenizer = imported_objects.get('get_tokenizer')
    tokenizer = get_tokenizer(size = cfg.vocab_len)

    # the model itself
    from modules.model import Model
    
    # Initialize a blank model
    model = Model(cfg).to(cfg.device) 
    
    # Load the saved state dictionary
    path = f'{model_name}/model.pth'
    model.load_state_dict(torch.load(path)) 

    return model, tokenizer, cfg
