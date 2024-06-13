import torch

def sampler(
    logits: torch.Tensor,  # (batch_size, input_len, vocab_size)
    temperature: float = 0.7,  # controls randomness. set to 1.0 in order to not use temperature
    top_p: float = 0.9,  # cumulative probability threshold. set to 1.0 or preferably None in order to not use top_p
    top_k: int = 50,  # max number of top tokens considered. set to tokenizer.vocab_len or preferably None in order to not use top_k
):
    """Generate token predictions from logits."""
    vocab_len, device = logits.shape[-1], logits.device
    
    logits = logits[:, -1, :]  # (batch_size, vocab_size)
    logits.div_(temperature)
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # both are (batch_size, vocab_size)

    # Top-p filtering
    if top_p is not None:
        probs_sum = torch.cumsum(probs_sort, dim=-1) # (batch_size, vocab_size)
        top_ps_mask = (probs_sum - probs_sort) > top_p # 0's are top-p selections & 1's are to be excluded
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

    # Top-k filtering
    if top_k is not None:
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) >= top_k # shape (vocab_size) tensor that iterates up by 1's
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) # (batch_size, vocab_size)
        top_ks_mask = top_ks_mask >= top_k

    # combining top-p with top-k and using whichever gives us fewer tokens
    if top_k is not None:
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalize
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)  
    # restore original order
    probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))  

    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id

def generate(
    prompt: str, # a single string for the prompt
    model,  # function that should output (batch_size,seq_len,vocab_len) tensor and a tensor w/ a single loss value (not used)
    tokenizer, # tokenizer of choice
    temperature: float = 0.7, # values above 1 increase entropy of predicted words. values near zero decrease entropy
    top_p: float = 0.9, # optionally prevents the model from sampling tokens that don't fit within the cumsum range
    top_k: int = 50, # optionally prevents the model from sampling tokens that don't fit within the list of top k most likely tokens
    max_gen_len: int = None, # maximum length you want the model to generate
    memory_saver_div: int = 1, # defaults to full max_seq_len**2 memory use. probably needs to be power of 2. not all models can take advantage of this
):
    """Generate text from a prompt using the model and sampling settings."""
    vocab_len, max_seq_len = tokenizer.vocab_len, model.max_seq_len
    
    max_context_window = max_seq_len // memory_saver_div
    if memory_saver_div != 1:
        assert ((memory_saver_div & (memory_saver_div-1)) == 0) & (memory_saver_div > 0), \
        f'memory_saver_div {memory_saver_div} must be power of 2'
        print(f'maximum attention matrix size in memory will be {max_context_window}x{max_seq_len} rather than {max_seq_len}x{max_seq_len}\n')

    tokens = tokenizer.encode(prompt)
    max_gen_len = max_seq_len - len(tokens) if max_gen_len is None else max_gen_len
    tokens = torch.tensor([tokens], device=model.device).to(torch.long)
    assert tokens.shape[0] == 1, f'batched inference is not currently supported.'

    cache_len = max(tokens.shape[1] - max_context_window, 0)
    for i in range(max_gen_len):
        with torch.no_grad():
            logits, _ = model(input_token_ids = tokens[:,-max_context_window:], cache_len = cache_len)
          
        # turn the logits into probabilities and sample from them
        next_token = sampler(logits, temperature, top_p, top_k)

        # if the model outputs the eos token, we're done
        if next_token == tokenizer.eos_id: break
        # otherwise, add our new token to the sequence
        tokens = torch.cat([tokens, next_token], dim=-1)

        # update our kv cache length
        if tokens.shape[1] >= max_context_window:
            cache_len += 1
    
    return tokenizer.decode(tokens.squeeze(0).tolist())
