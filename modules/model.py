import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.attention import PrecomputeRotaryFrequencies
from modules.layer import Layer

class Model(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len
        self.dropout_rate = cfg.dropout_rate

        # the generate() function in inference.py references these values to build kv cache
        self.head_dim = cfg.head_dim 
        self.num_kv_heads = cfg.num_kv_heads

        ### positional encodings
        self.pos_enc_type = cfg.pos_enc_type
        if cfg.pos_enc_type == 'learnable': # learnable, like in GPT-2
            self.pos_embedder = nn.Embedding(cfg.max_seq_len, cfg.dim, device=cfg.device)
        elif cfg.pos_enc_type == 'Sinusoidal': # sinusoidal, like in "Attention Is All You Need"
            position = torch.arange(cfg.max_seq_len, device=cfg.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, cfg.dim, 2, device=cfg.device) * (-math.log(cfg.theta) / cfg.dim))
            sinusoidal_embedding = torch.zeros(cfg.max_seq_len, cfg.dim, device=cfg.device)
            sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)
            sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_embedding', sinusoidal_embedding)
        elif cfg.pos_enc_type == 'RoPE': # pre-computing rotary positional embedding frequencies
            self.precompute_freqs = PrecomputeRotaryFrequencies(cfg.head_dim, cfg.max_seq_len, cfg.theta, cfg.device)
        else:
            raise InputError(f"positional encoding '{cfg.pos_enc_type}' unknown. Choose either 'RoPE', 'Sinusoidal', or 'learnable'")

        # residual state initialization
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim, device=cfg.device)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0

        # the causal attention mask
        self.mask = torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool, device=cfg.device).tril()
            # False -> "mask this token" while True -> "Let the model see this token"

        # the model itself
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))

        # the output projection
        self.final_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)
        self.output = nn.Linear(cfg.dim, self.vocab_len, bias=False, device=cfg.device)

        # optionally making the output linear layer tie weights to the input embedding matrix
        self.out_weight_share = cfg.out_weight_share
        if cfg.out_weight_share: self.token_embedder.weight = self.output.weight

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index = cfg.vocab_len -1) # ignore the padding token

        # initializing params to specific distributions. self.apply() applies the function to all parts of the model
        self.apply(self.__init__weights)
            # should i make this optional? not sure if this distribution is still used for modern models or just GPT2

    def __init__weights(self, module):
        """
        GPT-2 style parameter initialization for the final linear proj in Attn & MLP Layers
        The idea is to scale the distribution by the number of layers to ensure that the output variance ==1 rather than blowing up
        Not sure if this setup is still used for modern models. If not then I need to change this
        """
        if isinstance(module, nn.Linear): # for every linear layer
            std = 0.02 # distribution will be centered at 0 with standard deviation of 0.02

            # specific weight matrices at the end of each layer are given smaller std to keep the residual stream small
            if hasattr(module, 'GPT_scale_init'):
                std *= (2 * self.num_layers) ** -0.5

            # carries out the actual initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # biases should instead be initialized to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # the embedding matrix doesn't count as an nn.Linear so we've gotta do it again for that
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        The token embeddings get subtracted unless weight tying to the output layer is enabled
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding & (self.out_weight_share == False):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @log_io
    def forward(
        self, 
        input_token_ids: torch.Tensor, 
        target_token_ids: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Our GPT's primary forward function that calls all the other modules
        """
        input_token_ids = input_token_ids.to(self.device)
        if target_token_ids is not None:
            target_token_ids = target_token_ids.to(self.device)

        batch_size, seq_len = input_token_ids.shape

        if target_token_ids is not None: # training setup
            training = True
            assert input_token_ids.shape == target_token_ids.shape
            assert seq_len == self.max_seq_len
            mask = self.mask
        else: # inference setup
            training = False
            mask = self.mask[:seq_len, :seq_len]

        # setting up our positional encoding
        if self.pos_enc_type == 'learnable':
            pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device) # shape (seq_len)
            pos_emb = self.pos_embedder(pos) # shape (seq_len, dim)
            freqs = None # make sure not to pass in any RoPE frequencies into the model
        elif self.pos_enc_type == 'Sinusoidal':
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0).to(self.device) # shape (seq_len, dim)
            freqs = None # make sure not to pass in any RoPE frequencies into the model
        elif self.pos_enc_type == 'RoPE':
            # precomputing our RoPE frequencies
            freqs = self.precompute_freqs() 
                # dict {'sin': shape (1, max_seq_len, 1, head_dim), 'cos': shape (1, max_seq_len, 1, head_dim)}
        else:
            # bc of the InputError in the __init__ you shouldn't be able to get to here
            # but I guess if you did then you'd have a model with no awareness of position
            freqs = None
            
        # initializing the first residual state
        if self.pos_enc_type in ['learnable', 'Sinusoidal']:
            x = (self.token_embedder(input_token_ids) + pos_emb) * self.scale # (batch_size, seq_len, dim)
        else: # RoPE gets implemented inside the attention mechanism instead of at the residual state initialization
            x = self.token_embedder(input_token_ids) * self.scale # (batch_size, seq_len, dim)
        if training: x = F.dropout(x, self.dropout_rate)
        
        # run through the model's layers
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs, mask, training)
        
        # the final output of the model
        logits = self.output(self.final_norm(x)) # (batch_size, seq_len, vocab_len)
        
        if training:
            loss = self.criterion(
                logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )
            return logits, loss
        else: 
            return logits, None