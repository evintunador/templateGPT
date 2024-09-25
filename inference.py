import torch
from tqdm import tqdm

def sampler(
    logits: torch.Tensor,  # (batch_size, input_len, vocab_size)
    temperature: float = 1.0,  # controls randomness. set to 1.0 in order to not use temperature
    min_p: float = 0.05,  # min-p sampling threshold https://arxiv.org/abs/2407.01082
    top_k: int = None,  # max number of top tokens considered. set to None in order to not use
    top_p: float = None,  # cumulative probability threshold. set to 1.0 or preferably None in order to not use top_p
    cache_len: torch.Tensor = None,  # specify which slice of logits to use
):
    """Generate token predictions from logits."""
    vocab_len, device = logits.shape[-1], logits.device

    if cache_len is not None:
        logits = logits[torch.arange(logits.size(0)), cache_len, :]  # (batch_size, vocab_size)
    elif (cache_len is None) and logits.shape[1] == 1:
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
    else:
        print(logits.shape)
        raise ValueError('expected either batch_size=1 or a list of indices designating how long each prompt is to avoid padding')
        
    logits.div_(temperature)
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # both are (batch_size, vocab_size)

    # Min-p sampling
    if min_p is not None:
        probs_max = probs_sort[:, 0].unsqueeze(-1)
        min_p_threshold = min_p * probs_max
        min_p_mask = probs_sort < min_p_threshold
        probs_sort = torch.where(min_p_mask, 0, probs_sort)
        
    # Top-p filtering (if specified)
    if top_p is not None:
        probs_sum = torch.cumsum(probs_sort, dim=-1) # (batch_size, vocab_size)
        top_ps_mask = (probs_sum - probs_sort) > top_p # 0's are top-p selections & 1's are to be excluded
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

    # Top-k filtering (if specified)
    if top_k is not None:
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) >= top_k 
            # shape (vocab_size) tensor that iterates up by 1's
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) # (batch_size, vocab_size)
        top_ks_mask = top_ks_mask >= top_k

    # combining top-p with top-k and using whichever gives us fewer tokens
    if top_k is not None:
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalize
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)  
    # restore original order
    probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))  
    # sample from distribution
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    return next_token_id

def generate(
    prompt, # either a single string or a list of strings for the prompt
    model,  # nn.module that should output (batch_size,seq_len,vocab_len) and a list of dictionaries containing kv cache tensors
    tokenizer, # tokenizer of choice
    temperature: float = 1.0, # values above 1 increase entropy of predicted words. values near zero decrease entropy
    min_p: float = 0.05, # min-p sampling threshold https://arxiv.org/abs/2407.01082
    top_k: int = None, # optionally prevents model from sampling tokens that don't fit within the list of top k most likely tokens
    top_p: float = None, # optionally prevents the model from sampling tokens that don't fit within the cumsum range
    max_gen_len: int = None # maximum length you want the model to generate
):
    """Generate text from a prompt using the model and sampling settings."""
    # Convert min_p=0 to None to disable it if 0 was passed in
    min_p = None if min_p == 0 else min_p

    if type(prompt) == list: tokens_list = [tokenizer.encode(p) for p in prompt]
    elif type(prompt) == str: tokens_list = [tokenizer.encode(prompt)]
    else: raise TypeError(f'prompt is of type {type(prompt)} but should be string or list of strings')
    max_prompt_len = max(len(tokens) for tokens in tokens_list)
    print("max_prompt_len:\n", max_prompt_len)

    max_gen_len = model.max_seq_len - max_prompt_len if max_gen_len is None else min(max_gen_len, model.max_seq_len - max_prompt_len)
        # i think the way i'm doing this rn means that sequences other than the longest prompt will be given varying max_gen_len but whatevs
    print("max_gen_len:\n", max_gen_len)

    # making sure all sequences are of same length so that tensors work later. fill with pad tokens which will get calc'ed but not used
    tokens_padded = [
        tokens + [tokenizer.pad_id] * (max_prompt_len - len(tokens)) 
        for tokens in tokens_list]
    
    tokens = torch.tensor(tokens_padded, device=model.device).to(torch.long) # (batch_size, max_prompt_len)
    batch_size = tokens.shape[0]
    print("tokens:\n", tokens, tokens.shape)
    
    # for keeping track of if/when each sequence outputs an EOS token so that we can end its output
    eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    print("eos_flags:\n", eos_flags.shape)

    # Initialize kv caches for each layer
    kv_cache = [{ 
            "k": torch.zeros((batch_size, model.max_seq_len, model.num_kv_heads, model.head_dim), device=model.device),
            "v": torch.zeros((batch_size, model.max_seq_len, model.num_kv_heads, model.head_dim), device=model.device),
        } for _ in range(model.num_layers)]
    
    # Calculate the indices for the first generation step. each prompt could have a different length
    cache_len = torch.tensor([
        len(tokens_list[idx]) - 1 for idx in range(batch_size)
    ], device=model.device) # shape (batch_size)
    print("cache_len:\n", cache_len, cache_len.shape)
    
    ### now the actual inference loop. 
    # create a mask for the first run, which conctains prompts of variable lengths (assuming batched inference)
    #mask = (tokens != tokenizer.pad_id).bool() # (batch_size, max_seq_len)
    # create a mask for the first run, which looks like a regular causal tril mask used during training
    mask = torch.ones(max_prompt_len, max_prompt_len, dtype=torch.bool, device=model.device).tril()
        # (True for real tokens, False for padding)
    print("mask:\n", mask, mask.shape)
    
    # first iteration is unique bc it has the entire prompt not just one query vector, so part of it is outside the for loop
    with torch.no_grad():
        logits, kv_cache = model(
            input_token_ids = tokens,
            cache_len = cache_len,
            kv_cache = kv_cache,
            mask = mask
        )
    print(":\n", logits.shape)

    # turn the logits into probabilities and sample from them to get our next tokens
    next_tokens = sampler(logits, temperature, min_p, top_k, top_p, cache_len)
    print("next_tokens:\n", next_tokens, next_tokens.shape)
        
    for i in tqdm(range(1, max_gen_len), unit='tokens', leave=False):
        # the contents of the for loop are kinda out-of-order because those first two chunks above needed to be done differently

        # Update eos_flags
        eos_flags |= (next_tokens.squeeze() == tokenizer.eos_id)
        # Set next tokens to pad tokens for sequences that have reached EOS
        next_tokens[eos_flags] = tokenizer.pad_id
            # i think this might be redundant thanks to the post-processing step below this loop
        print("eos_flags:\n", eos_flags)
        print("next_tokens:\n", next_tokens)

        padding_vec = torch.ones_like(next_tokens) * tokenizer.pad_id
        tokens = torch.cat([tokens, padding_vec], dim=-1)
        # For the shorter sequences, update at the specific applicable index
        for idx in range(batch_size):
            #if not eos_flags[idx]: # <- i dont remember what that was for
            tokens[idx, cache_len[idx] + 1] = next_tokens[idx]
                # should this be a -1 now?
        print("tokens:\n", tokens)
        
        # if the model has outputted the eos token for all sequences in the batch, we're done
        if eos_flags.all(): break

        # update our kv cache length
        cache_len += 1
        # and our mask
        #mask = torch.nn.functional.pad(mask, (1, 0, 0, 0), value=True)
        # and our mask, which is a collection of bool vectors that are different for each sequence
        mask = (tokens != tokenizer.pad_id).bool() # (batch_size, max_seq_len)
        print("cache_len:\n", cache_len)
        print("mask:\n", mask.shape)
        
        # running the model
        with torch.no_grad():
            logits, kv_cache = model(
                input_token_ids = tokens[:,-1:],
                cache_len = cache_len,
                kv_cache = kv_cache,
                mask = mask
            )
        print("logits:\n", logits.shape)

        # turn the logits into probabilities and sample from them to get our next tokens
        next_tokens = sampler(logits, temperature, min_p, top_k, top_p, cache_len=None)
        print("next_tokens:\n", next_tokens, next_tokens.shape)
          
    # Post-processing step to replace all tokens after EOS with padding tokens. might be redundant but i'm lazy
    for idx in range(batch_size):
        eos_position = (tokens[idx] == tokenizer.eos_id).nonzero(as_tuple=True)[0]
        if eos_position.numel() > 0:
            eos_idx = eos_position[0].item()
            tokens[idx, eos_idx + 1:] = tokenizer.pad_id
    print("tokens:\n", tokens)
    
    decoded_sequences = [tokenizer.decode(seq.tolist()) for seq in tokens]
    return decoded_sequences

###########################################################
#################### RUNNING THIS FILE ####################
###########################################################

import argparse
from tools import load_model
from inference import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a specified model and prompts.")
    
    # Model selection
    parser.add_argument("model", help="Name of the pre-trained model to use")
    # Prompts
    parser.add_argument("prompts", nargs="+", help="One or more prompts for text generation")
    
    # Optional parameters
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--min_p", type=float, default=0.05, help="Min-p sampling threshold (default: 0.05). Set to 0 to disable. https://arxiv.org/abs/2407.01082")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering value (default: None)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering value (default: None)")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum generation length")
    parser.add_argument("--show_tokens", action="store_true", help="Display tokenization of the output")
    parser.add_argument("--device", type=str, 
                        default = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                        help = "Options are 'cuda' (nvidia gpu), 'mps' (apple silicon), or 'cpu' (any computer)")
    
    args = parser.parse_args()
    
    try:
        # Load the model
        model, tokenizer, cfg = load_model(args.model, args.device)
        
        # Generate text
        outputs = generate(
            args.prompts,
            model,
            tokenizer,
            temperature=args.temp,
            min_p=args.min_p,
            top_k=args.top_k,
            top_p=args.top_p,
            max_gen_len=args.max_len
        )
        
        # Print outputs
        for i, output in enumerate(outputs):
            if args.show_tokens:
                print(tokenizer.display(output))
            else:
                print(output)
            print()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")