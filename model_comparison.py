import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List
from tools import load_model, torcherize_batch, get_data_loaders, import_from_nested_path

def plot_column_from_csv(models, x_column, y_column, log_x=False, log_y=False, trim_percentage=0):
    """
    Reads multiple CSV files, extracts specific columns, and plots a graph comparing their values
    with options to use logarithmic scales on the axes and trim initial data.

    Parameters:
    - models: List of strings, where each string is a name of a model residing in models/
    - x_column: String, the name of the column to use as the x-axis.
    - y_column: String, the name of the column to use as the y-axis.
    - log_x: Boolean, whether to use a logarithmic scale on the x-axis (default is False).
    - log_y: Boolean, whether to use a logarithmic scale on the y-axis (default is False).
    - trim_percentage: Integer, percentage of the initial data to exclude from the plot (default is 0).
    """
    if not models or not x_column or not y_column:
        raise ValueError("Paths list and column names must be provided and not empty.")

    plt.figure(figsize=(10, 6))
    
    for model in models:
        path = f'trained/{model}/log_data.csv'
        try:
            data = pd.read_csv(path)
            if x_column not in data.columns or y_column not in data.columns:
                raise ValueError(f"Columns {x_column} or {y_column} not found in {path}")

            # Calculate the index to start slicing from
            start_index = int(len(data) * (trim_percentage / 100))

            # Slice data from the calculated start index
            data = data.iloc[start_index:]
            plt.plot(data[x_column], data[y_column], label=f'{model}')
        except Exception as e:
            print(f"Failed to process {path}: {str(e)}")
    
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.title(f'{y_column} Over Training {x_column}s')
    plt.xlabel(x_column + (' (log scale)' if log_x else ''))
    plt.ylabel(y_column + (' (log scale)' if log_y else ''))
    plt.legend(title="Model")
    plt.grid(True)
    plt.show()

def calculate_topk_accuracy(logits, targets, k=5, padding_idx=None):
    # Calculate top-k predictions
    topk_indices = torch.topk(logits, k, dim=2).indices
    
    # Create a mask for valid (non-padding) targets
    if padding_idx is not None:
        valid_targets = targets != padding_idx
    else:
        valid_targets = torch.ones_like(targets, dtype=torch.bool)
    
    # Calculate correctness only where the target is valid
    correct = topk_indices.eq(targets.unsqueeze(2).expand_as(topk_indices)) & valid_targets.unsqueeze(2)
    
    # Calculate accuracy only on valid targets
    valid_correct = correct.any(dim=2)[valid_targets]
    if valid_correct.numel() == 0:
        return torch.tensor(0.0)  # return 0 if there are no valid targets to avoid division by zero
    else:
        return valid_correct.float().mean()

def evaluate_models(models_to_compare: List, dataset_name: str, topk: int = 5, batch_size: int = 32):
    # Data preparation
    _, data_loader = get_data_loaders(dataset_name, batch_size = batch_size)
    text = next(iter(data_loader))

    # Evaluate models
    results = {}
    for model_name in models_to_compare:
        model, tokenizer, cfg = load_model(model_name)
    
        x, y = torcherize_batch(tokenizer, text, max_seq_len = cfg.max_seq_len)
        x, y = x.to(cfg.device), y.to(cfg.device)
        # x and y are tensors shape [batch_size, max_seq_len] of dtype torch.int64
        
        with torch.no_grad():
            logits, _ = model(x, cache_len = 0)
    
        topk = 5
        topk_accuracy = calculate_topk_accuracy(logits, y, k = topk, padding_idx = cfg.vocab_len - 1)
        topk_indices = torch.topk(logits, topk, dim=2).indices
        
        # Store results
        results[model_name] = topk_accuracy.item()

    return results