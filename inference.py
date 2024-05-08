import torch

def sampler(
    logits: torch.Tensor,  # (batch_size, input_len, vocab_size)
    temperature: float,  # controls randomness
    top_p: float,  # cumulative probability threshold
    top_k: int,  # max number of top tokens considered
    vocab_len: int,  # vocabulary length
    device: str,  # specify the device tensors are on
):
    """Generate token predictions from logits."""
    logits = logits[:, -1, :]  # (batch_size, vocab_size)
    logits.div_(temperature)
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # both are (batch_size, vocab_size)

    # Top-p filtering
    probs_sum = torch.cumsum(probs_sort, dim=-1) # (batch_size, vocab_size)
    top_ps_mask = (probs_sum - probs_sort) > top_p # 0's are top-p selections & 1's are to be excluded
    probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

    # Top-k filtering
    top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) >= top_k # shape (vocab_size) tensor that iterates up by 1's
    #vocab_len, device=device
    top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) # (batch_size, vocab_size)
    top_ks_mask = top_ks_mask >= top_k

    # combining top-p with top-k and using whichever gives us fewer tokens
    probs_sort = torch.where(top_ks_mask, 0, probs_sort)
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)  # Re-normalize
    probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))  # restore original order

    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id

def generate(
    prompt: str,
    model,  # function that should output (batch_size,seq_len,vocab_len) tensor and a tensor w/ a single loss value (not used)
    tokenizer,
    max_gen_len: int = None,
    memory_saver_div: int = 1, # defaults to full max_seq_len**2 memory use. must be power of 2
    temperature=0.7,
    top_p=0.9,
    top_k=None,
):
    """Generate text from a prompt using the model and sampling settings."""
    vocab_len = tokenizer.vocab_len
    max_seq_len = model.max_seq_len
    assert ((memory_saver_div & (memory_saver_div-1)) == 0) & (memory_saver_div > 0), f'memory_saver_div {memory_saver_div} must be power of 2'
    max_context_window = max_seq_len // memory_saver_div
    if memory_saver_div != 1:
        print(f'maximum attention matrix size in memory will be {max_context_window}x{max_seq_len} rather than {max_seq_len}x{max_seq_len}\n')
    if top_k is None:
        top_k = tokenizer.vocab_len

    tokens = tokenizer.encode(prompt)
    max_gen_len = max_seq_len - len(tokens) if max_gen_len is None else max_gen_len
    tokens = torch.tensor([tokens], device=model.device).to(torch.long)
    assert tokens.shape[0] == 1, f'batched inference is not currently supported.'

    cache_len = max(tokens.shape[1] - max_context_window, 0)
    for i in range(max_gen_len):
        with torch.no_grad():
            logits, _ = model(tokens[:,-max_context_window:], cache_len)
          
        # turn the logits into probabilities and sample from them
        next_token = sampler(logits, temperature, top_p, top_k, vocab_len, model.device)

        # if the model outputs the eos token, we're done
        if next_token == tokenizer.eos_id: break
        # otherwise, add our new token to the sequence
        tokens = torch.cat([tokens, next_token], dim=-1)

        # update our kv cache length
        if tokens.shape[1] >= max_context_window:
            cache_len += 1
    
    return tokenizer.decode(tokens.squeeze(0).tolist())
