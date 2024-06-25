import torch
import inspect

from tools import torcherize_batch, save_model

###########################################################
#################### EVALUATION ###########################
###########################################################
@torch.no_grad()
def estimate_loss(model, tokenizer, train_data_loader, test_data_loader, eval_samples = 3): # to estimate loss during the training loop
    out = {} # dictionary to record & separate train loss from val loss
    model.eval() # sets model to eval mode so we're not keeping track of gradients
    for split in ['train', 'val']:
        losses = torch.zeros(eval_samples)
        for k in range(eval_samples):
            # grab a list of strings from either the train or test set
            batch = next(iter(train_data_loader)) if split == 'train' else next(iter(test_data_loader))
            # turn list of strings into tensor of token indices
            X, Y = torcherize_batch(tokenizer, batch, model.max_seq_len, model.device) 
            # run the model to get loss
            logits, loss = model(X, target_token_ids=Y)
            losses[k] = loss.item()
        out[split] = losses
    model.train() # just resets the model to training mode
    return out

###########################################################
#################### Learning Rate Schedule ###############
###########################################################
def scheduler_lambda(current_iter):
    from config import TrainConfig
    tcfg = TrainConfig()
    T_i = tcfg.T_0()
    if current_iter < tcfg.warmup_iters:
        # Linear warmup
        lr = tcfg.lr_init + (tcfg.lr_max - tcfg.lr_init) * (current_iter / tcfg.warmup_iters)
    elif current_iter < tcfg.max_iters - tcfg.final_flat_iters:
        # Cosine annealing with warm restarts
        cycle_iter = current_iter - tcfg.warmup_iters
        while cycle_iter >= T_i:
            cycle_iter -= T_i
            T_i *= tcfg.T_mult
        if tcfg.anneal_type == 'lin': 
            lr = tcfg.lr_max - (tcfg.lr_max - tcfg.lr_min) * (cycle_iter / T_i)
        else:
            # defaults to 'cos' learning rate annealing
            lr = tcfg.lr_min + 0.5 * (tcfg.lr_max - tcfg.lr_min) * (1 + torch.cos(torch.pi * torch.tensor(cycle_iter / T_i)))
    else:
        # Constant learning rate
        lr = tcfg.lr_min
    return lr

###########################################################
#################### Optimizer ############################
###########################################################

def get_optimizer(model, tcfg):
    """
    this function builds an optimizer ensuring weight decay is not used on affine layers in Norms
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': tcfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused (more efficient) version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and model.device == 'cuda' # only uses fused if the device is cuda
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=1.0, betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.epsilon)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

###########################################################
#################### TRAINING #############################
###########################################################
import time
import csv

def train(
    model, 
    tokenizer, 
    cfg, 
    optimizer,
    scheduler,
    tcfg, 
    train_data_loader,
    test_data_loader,
    log_data: list = None, 
    detect_anomoly: bool = False # use if you're getting crazy errors about a the gradient being broken
):
    if log_data is None: # for recording loss/ppl curves
        log_data = []
    
    # Enable anomaly detection. useful for really deep issues in the model where the gradient breaks
    if detect_anomoly: torch.autograd.set_detect_anomaly(True)

    # this provides some free performance assuming your GPU supports it
    torch.set_float32_matmul_precision('high')

    # initializing variable(s) that are referenced before assignment
    norm = 0.0

    # keeping track of total training time & tokens/sec
    tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum_steps * cfg.max_seq_len
    steps_since_last_eval = 0
    last_eval_time = time.time()
    start_time = time.time()
    
    for i in range(tcfg.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if (i % tcfg.eval_interval) == 0 or (i == tcfg.max_iters - 1):
            current_time = time.time()
            elapsed_time = current_time - start_time
            eval_elapsed_time = current_time - last_eval_time
            
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr is torch.Tensor: current_lr = current_lr.item()

            # estimate loss
            losses = estimate_loss(model, tokenizer, train_data_loader, test_data_loader, eval_samples = tcfg.eval_samples)
            
            # Calculate tokens per second
            total_tokens = tokens_per_step * steps_since_last_eval
            tokens_per_second = total_tokens / eval_elapsed_time if i > 0 else 0.0
            
            # Collect data for CSV & print it
            log_data.append([
                i, elapsed_time, tokens_per_second,
                losses['train'].mean().item(), losses['val'].mean().item(), torch.exp(losses['val']).mean().item(),
                current_lr, norm
            ])
            print(
                f"step: {i:04d}, time elapsed: {elapsed_time:.2f}s, tokens/s: {int(tokens_per_second):08d}, "
                f"train loss: {losses['train'].mean().item():.4f}, val loss: {losses['val'].mean().item():.4f}, "
                f"ppl: {torch.exp(losses['val']).mean().item():.0f}, lr: {current_lr:.8f}, grad norm: {norm:.4f}"
            )
            
            # Reset token count and last eval time
            steps_since_last_eval = 0
            last_eval_time = current_time

        # setup for training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        # we can simulate a larget batch size by accumulating gradients over many micro batches
        for micro_step in range(tcfg.grad_accum_steps):
            # sample a batch of data
            batch = next(iter(train_data_loader))
            x,y = torcherize_batch(tokenizer, batch, cfg.max_seq_len, cfg.device)
        
            # calc micro batch gradient
            with torch.autocast(device_type = cfg.device, dtype = torch.bfloat16): # enables mixed-precision training
                logits, loss = model(x, target_token_ids=y)
            loss = loss / tcfg.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
        # clip the gradients
        if tcfg.grad_clip is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        
        # update parameters
        optimizer.step()
        # wait for the GPU to finish work
        if model.device == 'cuda': torch.cuda.synchronize() 
        # Update the learning rate  
        scheduler.step() 
        
        steps_since_last_eval += 1
        
        # every once in awhile save a checkpoint of the model
        if tcfg.checkpoint_interval is not None:
            if (i % tcfg.checkpoint_interval == 0) or i == 0:
                save_model(model, cfg, tcfg, log_data, checkpoint=True)
    
    # Disable anomaly detection after the training loop
    if detect_anomoly: torch.autograd.set_detect_anomaly(False)

    return model, optimizer, log_data # do i need to send out train & test dataloader?