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
    # general hyperparameters
    dim: int = 64
    vocab_len: int = None # will be set later according to what tokenizer you choose
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # can't do MPS bc metal doesn't support complex64 used in RoPE
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # Residual Layers
    num_layers: int = 4 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    second_resid_norm: bool = False # True adds an extra Norm after the attn & MLP, like in Grok. Only recommended if using RMSNorm
    
    # Multi-Layer Perceptrion
    mlp_hidden_mult: int = 2.667 # if mlp_gated = True then this has 1.5x the parameters of if mlp_gated = False
    mlp_bias: bool = False # whether to use bias weights. Llama3 does not and I'm partial to their choice
    mlp_nonlinearity: str = 'SiLU' # options are 'GeLU', 'SiLU', and 'ReLU'(not recommended)
    mlp_gated: bool = True # Turns GeLU into GeGLU, giving you 50% more MLP parameters to train but also more expressiveness

    # Multi-Query Attention
    num_q_heads: int = 4 # `num_q_heads % num_kv_heads == 0` must be true
    num_kv_heads: int = 1 # set =num_q_heads to revert to regular multi-head attention (not recommended)
    head_dim: int = 16 # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention
    theta: float = 10_000 # 10_000 is the most common choice. Llama3 uses 50_000
    max_seq_len: int = 256 # 512 is the most my 8gb of ram can handle

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm'(recommended), 'LayerNorm', and 'CosineNorm'. Add more options in 'model.py'
    norm_affine: bool = True # whether to use a linear layer after each norm. recommended especially if you're using LayerNorm or CosineNorm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

    # inference (kv caching)
    max_batch_size: int = 1 # i haven't tried changing this from 1
    # it needs to be set >1 at the first model initialization if you ever want to be able to do batched inference. i should fix that
    # i think batched inference is probably broken rn bc of my shitty tokenizer. might fix in future

@dataclass
class TrainConfig:
    """
    Design your training loop here
    """
    # name of the folder the model will be saved into
    model_name = f'{time.strftime("%Y-%m-%d|%H-%M-%S")}'
    
    weight_decay: float = 0.02
    batch_size: int = 32
    
    # total number of batches to run over the course of training
    max_iters: int = 100 # i recommend at least 1_000
    # how often to print out an update on how training is going
    eval_interval: int = 10 # doing this too often slows things down hella but also gives detailed log data
    # how many samples to take at each evaluation. more means a more accurate loss/perplexity calculation
    eval_samples: int = 1 # this number can things down hella. each sample is almost like doing an extra training iteration
    # how often to save a model checkpoint
    checkpoint_interval: int = None # eval_interval # set to None if you don't want checkpoints
    
    ### to visualize the learning rate schedule you define here, see cell 7 of training.ipynb

    # Initial learning rate to start from during the warmup
    lr_init: float = 1e-5
    # Maximum and minimum learning rates during annealing
    lr_max: float = 1e-1
    lr_min: float = 1e-3
    # if you'd like a flat learning rate, set lr_init = lr_min = lr_max and ignore the variables below
    
    # number of iterations for a linear warmup from lr_min to lr_max
    warmup_iters: int = int(max_iters * 0.1) # if you don't want to use a lr warmup, set = 0
    # number of iterations for a constant learning rate of lr_min at the end of training
    final_flat_iters: int = int(max_iters * 0.1) # if you don't want to use a final flat lr at the end, set = 0
    
    # type of annealment to use. Annealment is when the learning rate decreases over the course of training
    anneal_type: str = 'cos' # options: 'cos'(recommended) and 'lin'
    # number of times to bring the learning rate back up from lr_min to lr_max in-between the warmup & final flat
    num_restarts: int = 3 # if you don't want to use warm restarts, set =0 and ignore T_mult
    # relative length of each warm restart compared to the previous.
    T_mult: int = 2 # =1 makes all restarts the same length, <1 means they get shorter and >1 makes them longer
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self): # I DO NOT RECOMMEND EDITING THIS
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))









@dataclass
class HyperParameterSearchConfig:
    """
    NOT CURRENTLY FUNCTIONAL; STILL IN PLANNING STAGE
    
    Determines the hyperparameters to be tested and the order to test them
    The first entry of each list will be tried first

    Keep in mind before defining all your values that these take A LOT of time to run as your lists get even slightly longer.
    To find out how many training runs are going to be *attempted* (they will be skipped over if ram usage is too high),
    what you do is take the length of all the lists and multiply those lengths together. 
    """

    ### ModelConfig
    # general
    dim = [128, 64]
    vocab_len = [1024]
    dropout_rate = [0.1]

    # Residual Layers
    num_layers = [12, 8]
    pre_connect_dropout = [False]
    second_resid_norm = [False]
    
    # MLP
    mlp_hidden_mult = [4, 2]
    mlp_bias = [False] 
    mlp_nonlinearity = ['GeLU']
    mlp_gated = [True]

    # attention
    num_q_heads = [12, 8, 4]
    num_kv_heads = [1] # need to add a conditional that prevents use of num_q_heads < num_kv_heads
    head_dim = [32, 16] 
    theta = [10_000]
    max_seq_len = [512] # using a longer seq_len isn't really feasible and using a shorter one is kinda useless

    # normalization
    scale_first_resid = [True]
    norm_type = ['RMSNorm']
    norm_affine = [True]
    norm_bias = [True] # only actually does anything if norm_affine == True

    ### TrainConfig
    weight_decay = [0.01]
    batch_size = [32, 24, 16] # need to assert only try a lower batch size for models that ran out of ram at higher batch sizes
    max_iters = [1000]
    
    # learning rate scheduling
    lr_max = [1e-2]
    lr_min = [1e-6, 1e-4, 1e-2] # need to assert if lr_min == lr_max then don't iterate over any of the below
    def warmup_iters(self): 
        return [int(self.max_iters * i) for i in [0.05]]
    def final_flat_iters(self):
        return [int(self.max_iters * i) for i in [0.1, 0.3]]
    anneal_type = ['cos']
    num_restarts = [0, 3] # need to assert if num_restarts == 0 then don't iterate over T_mult
    T_mult = [1, 2] 

    # 2*2*2*3*2 + 2*2*2*3*2*2*2 + 2*2*2*3*2*2*2*2 = 624 models
    # every time you hit a hp that makes downstreap hp's useless, you get to add & then continue on with the following multiple
    # need to make a function that calculates this number

    ### Hyperparameter Testing Order
    # this determines which lists will be iterated first. put hyperparameters that you're more interested in learning about in the front.
    # i recommend putting batch size last since the purpose of testing a smaller batch size is really just to try and fit bigger models into ram
    #def order(self):
        #return [] # need to fill in