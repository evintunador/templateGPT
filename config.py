from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import time

@dataclass
class ModelConfig:
    """
    Design your GPT here
    Yes I know dropout_rate should probably be in TrainConfig but it was easier to implement from here
    """
    ### general hyperparameters
    dim: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization
    linear_bias: bool = False # whether to use bias weights on our linear layers. Llama3 does not and I'm partial to their choice
    out_weight_share: bool = True # whether to share weights between output layer and input embedding layer
    max_seq_len: int = 7 # 512 is the most my 8gb of ram can handle. I think GPT2 did 1024
    
    ### positional encoding
    # the method to use for helping the model understand the order of tokens.
    pos_enc_type: str = 'RoPE' # Options are:
        # 'RoPE': a relative positional encoding method used in most modern models https://arxiv.org/abs/2104.09864
        # 'learnable': an absolute pos enc method used in GPT-2. it's not too great and adds max_seq_len * dim parameters to learn
        # 'Sinusoidal': an absolute pos enc method used in the original "Attention is All You Need" paper https://arxiv.org/abs/1706.03762
    # a hyperparameter used in RoPE and Sinusoidal
    theta: float = 10_000 # 10_000 is the most common choice for both RoPE & Sinusoidal. Llama3 uses 50_000 for RoPE
        # does nothing if pos_enc_type=='learnable'

    ### tokenizer
    tokenizer: str = 'bpe_tinyStories' # must choose from one of the folders in 'tokenizers/'
        # current options: 'bpe_tinyStories', 'bpe_fineweb', 'bpe_fineweb-edu', 'byte'
            # it is possible to train a model on a dataset different from what your tokenizer was trained on
            # if you choose 'byte' then vocab_len will be ignored/overridden
    vocab_len: int = 2048 # options can be found in the `models/` sub-folder inside whatever tokenizer you just chose above^
        # for `bpe_tinyStories` the options are 512, 1024, 2048
        # for 'bpe_fineWeb' and 'bpe_fineWeb-edu' the options are 512, 1024, 2048, 4096, 8192, 16_384, 32_768

    ### Residual Layers
    num_layers: int = 2 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    second_resid_norm: bool = False # True adds an extra Norm after the attn & MLP, like in Grok. Only recommended if using RMSNorm
    
    ### Multi-Layer Perceptrion
    mlp_hidden_mult: float = 2 # how wide the hidden dimension of the MLP should be. if mlp_gated = True that's not quite the correct description but whatever
    mlp_nonlinearity: str = 'SiLU' # options are 'GeLU', 'SiLU', and 'ReLU'(not recommended)
    mlp_gated: bool = True # Turns SiLU into SwiGLU, GeLU into GeGLU, etc. https://arxiv.org/abs/2002.05202v1
    # ^ if gated == True, mlp_hidden_mult will automatically adjust to maintain parameter count

    ### Multi-Query Attention
    num_q_heads: int = 2 # `num_q_heads % num_kv_heads == 0` must be true
    num_kv_heads: int = 1 # set =num_q_heads to revert to regular multi-head attention (not recommended)
    head_dim: int = dim // num_q_heads # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention

    ### normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm'(recommended), 'LayerNorm', and 'CosineNorm'. Add more options in 'norm.py'
    norm_affine: bool = True # whether to use a linear layer after each norm. recommended especially if you're using LayerNorm or CosineNorm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

    def __post_init__(self):
        """
        These are just checks to make sure everything works ahead of time. do not edit them unelss you know what you're doing
        """
        
        # General
        assert isinstance(self.dim, int) and self.dim > 0, "dim must be a positive integer"
        assert self.device in ['cuda', 'mps', 'cpu'], "device must be 'cuda', 'mps', or 'cpu'"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be between 0 and 1"
        assert isinstance(self.linear_bias, bool), "linear_bias must be a boolean"
        assert isinstance(self.out_weight_share, bool), "out_weight_share must be a boolean"
        assert isinstance(self.max_seq_len, int) and self.max_seq_len > 0, "max_seq_len must be a positive integer"
    
        # Positional
        assert self.pos_enc_type in ['RoPE', 'learnable', 'Sinusoidal'], "pos_enc_type must be 'RoPE', 'learnable', or 'Sinusoidal'"
        assert self.theta > 0, "theta must be a positive number"
    
        # Tokenizer
        assert self.tokenizer in ['bpe_tinyStories', 'bpe_fineweb', 'bpe_fineweb-edu', 'byte'], "Invalid tokenizer"
        if self.tokenizer == 'byte': self.vocab_len = 259 # 259 = 256 bytes + 3 special tokens
        assert isinstance(self.vocab_len, int) and self.vocab_len > 0, "vocab_len must be a positive integer"
    
        # Residual layer
        assert isinstance(self.num_layers, int) and self.num_layers > 0, "num_layers must be a positive integer"
        assert isinstance(self.second_resid_norm, bool), "second_resid_norm must be a boolean"
    
        # MLP
        assert self.mlp_hidden_mult > 0, "mlp_hidden_mult must be a positive number"
        assert self.mlp_nonlinearity in ['GeLU', 'SiLU', 'ReLU'], "mlp_nonlinearity must be 'GeLU', 'SiLU', or 'ReLU'"
        assert isinstance(self.mlp_gated, bool), "mlp_gated must be a boolean"
    
        # Multi-Query Attention
        assert isinstance(self.num_q_heads, int) and self.num_q_heads > 0, "num_q_heads must be a positive integer"
        assert isinstance(self.num_kv_heads, int) and self.num_kv_heads > 0, "num_kv_heads must be a positive integer"
        assert self.num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        assert self.dim % self.num_q_heads == 0, "dim must be divisible by num_q_heads"
    
        # Normalization
        assert isinstance(self.scale_first_resid, bool), "scale_first_resid must be a boolean"
        assert self.norm_type in ['RMSNorm', 'LayerNorm', 'CosineNorm'], "norm_type must be 'RMSNorm', 'LayerNorm', or 'CosineNorm'"
        assert isinstance(self.norm_affine, bool), "norm_affine must be a boolean"
        assert isinstance(self.norm_bias, bool), "norm_bias must be a boolean"
        assert self.eps > 0, "eps must be a positive number"

@dataclass
class TrainConfig:
    """
    Design your training loop here
    """
    # name of the folder the model will be saved into
    model_name: str = f'{time.strftime("%Y-%m-%d|%H-%M-%S")}' # defaults to the time that config.py was imported

    ### dataset/dataloader: see https://huggingface.co/docs/datasets/en/loading
    # your HuggingFace training dataset's repo address
    dataset_name: str = 'noanabeshima/TinyStoriesV2'
        # options include 'noanabeshima/TinyStoriesV2', 'HuggingFaceFW/fineweb' and 'HuggingFaceFW/fineweb-edu'
        # for any other datasets you'll need to mess with the code in tools.py yourself and likely train your own tokenizer on it
        # the fineweb datasets use the 10 billion token samples, but you could change hat pretty easily in tools.py
    # this parameter is equivalent to `name` in datasets' `load_dataset()` function
    data_subset: str = None
        # options for fineweb include 'sample-10BT', 'sample-100BT', 'sample-350BT', 'CC-MAIN-2024-10', and 'Default' (all 5.4/15T tokens)
        # None defaults to `sample-10BT` for the finewebs
        # this parameter doesn't apply to tinyStoriesV2
    # whether to download all the data (False) or stream it as you need it (True)
    streaming: bool = False
        # tinyStoriesV2 takes up a bit over 2GB and fineweb sample-10BT takes up 28.5GB, so keep that in mind if you set to False
    
    ### batch size hyperparams
    # micro_batch_size * grad_accum_steps = effective batch size
    # micro_batch_size * grad_accum_steps * max_seq_len = total number of tokens per batch
    micro_batch_size: int = 3
    grad_accum_steps: int = 2
        # set grad_accum_steps = 1 to not do gradient accumulation

    ### training length
    # total number of batches to run over the course of training
    max_iters: int = 20 # we'll refer to iterations of batches instead of epochs over the dataset
    # how often to print out an update on how training is going
    eval_interval: int = 2#max_iters // 100 # doing this too often slows things down hella but also gives detailed log data
    # how many samples to take at each evaluation. more means a more accurate loss/perplexity calculation
    eval_samples: int = 1 # this number can slow things down. each sample is almost like doing an extra training iteration
    # how often to save a model checkpoint
    checkpoint_interval: int = None # eval_interval # set to None if you don't want to save checkpoints

    ### AdamW Hyperparameters https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.05
    grad_clip: float = 1.0 # this one's not actually part of AdamW but it feels at home here
    
    ### Learning Rate Schedule
        # to visualize the learning rate schedule, see cell 7 of training.ipynb
    # Initial learning rate to start from during the warmup
    lr_init: float = 1e-6
    # Maximum and minimum learning rates during annealing
    lr_max: float = 1e-2
    lr_min: float = 1e-4
        # if you'd like a flat learning rate, set lr_init = lr_min = lr_max and ignore the variables below
    
    # number of iterations for a linear warmup from lr_min to lr_max
    warmup_iters: int = int(max_iters * 0.1) # if you don't want to use a lr warmup, set = 0. you should def use it tho
    # number of iterations for a constant learning rate of lr_min at the end of training
    final_flat_iters: int = int(max_iters * 0.05) # if you don't want to use a final flat lr at the end, set = 0
    
    # type of annealment to use. Annealment is when the learning rate decreases over the course of training
    anneal_type: str = 'cos' # options: 'cos'(recommended) and 'lin'
    # number of times to bring the learning rate back up from lr_min to lr_max in-between the warmup & final flat
    num_restarts: int = 0 # if you don't want to use warm restarts, set =0 and ignore T_mult
    # relative length of each warm restart compared to the previous.
    T_mult: int = 2 # =1 makes all restarts the same length, <1 means they get shorter and >1 makes them longer
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self): # I DO NOT RECOMMEND EDITING THIS
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))

    def __post_init__(self):
        """
        These are just checks to make sure everything works ahead of time. do not edit them unelss you know what you're doing
        """
        
        # General 
        assert isinstance(self.model_name, str) and len(self.model_name) > 0, "model_name must be a non-empty string"
    
        # Dataset 
        assert isinstance(self.dataset_name, str) and len(self.dataset_name) > 0, "dataset_name must be a non-empty string"
        assert self.data_subset is None or isinstance(self.data_subset, str), "data_subset must be None or a string"
        assert isinstance(self.streaming, bool), "streaming must be a boolean"
    
        # Batch size 
        assert isinstance(self.micro_batch_size, int) and self.micro_batch_size > 0, "micro_batch_size must be a positive integer"
        assert isinstance(self.grad_accum_steps, int) and self.grad_accum_steps > 0, "grad_accum_steps must be a positive integer"
    
        # Training length 
        assert isinstance(self.max_iters, int) and self.max_iters > 0, "max_iters must be a positive integer"
        assert isinstance(self.eval_interval, int) and self.eval_interval > 0, "eval_interval must be a positive integer"
        assert isinstance(self.eval_samples, int) and self.eval_samples > 0, "eval_samples must be a positive integer"
        assert self.checkpoint_interval is None or (isinstance(self.checkpoint_interval, int) and self.checkpoint_interval > 0), \
        "checkpoint_interval must be None or a positive integer"
    
        # AdamW hyperparameter 
        assert 0 < self.beta1 < 1, "beta1 must be between 0 and 1"
        assert 0 < self.beta2 < 1, "beta2 must be between 0 and 1"
        assert self.epsilon > 0, "epsilon must be a positive number"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.grad_clip > 0, "grad_clip must be a positive number"
    
        # Learning rate schedule 
        assert self.lr_init > 0, "lr_init must be a positive number"
        assert self.lr_max > 0, "lr_max must be a positive number"
        assert self.lr_min > 0, "lr_min must be a positive number"
        assert self.lr_min <= self.lr_max, "lr_min must be less than or equal to lr_max"
        assert self.lr_init <= self.lr_max, "lr_init must be less than or equal to lr_max"
    
        assert isinstance(self.warmup_iters, int) and self.warmup_iters >= 0, "warmup_iters must be a non-negative integer"
        assert isinstance(self.final_flat_iters, int) and self.final_flat_iters >= 0, "final_flat_iters must be a non-negative integer"
        assert self.warmup_iters + self.final_flat_iters <= self.max_iters, "warmup_iters + final_flat_iters must be less than or equal to max_iters"
    
        assert self.anneal_type in ['cos', 'lin'], "anneal_type must be 'cos' or 'lin'"
        assert isinstance(self.num_restarts, int) and self.num_restarts >= 0, "num_restarts must be a non-negative integer"
        assert self.T_mult > 0, "T_mult must be a positive number"
    
        # Verify T_0 calculation
        try:
            T_0 = self.T_0()
            assert isinstance(T_0, (int, float)) and T_0 > 0, "T_0 calculation must return a positive number"
        except Exception as e:
            raise ValueError(f"Error in T_0 calculation: {str(e)}")