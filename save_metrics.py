import os
import json
import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    logits: [B, vocab_size] (already sliced to the last position)
    """
    if temperature <= 0:
        # temperature=0 means greedy, keep that escape hatch
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_val, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        # remove tokens with cumulative prob above top_p, but always keep the first
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def print_and_save_metrics(prefill_metrics, decode_metrics, use_cache, max_new_tokens):

    root_dir = './metrics_new'
    os.makedirs(root_dir, exist_ok = True)
    if use_cache:
        label = f'cache_{max_new_tokens}.json'
    else:
        label = f'no_cache_{max_new_tokens}.json'
        
    file_path = os.path.join(root_dir, label)

    assert file_path is not None , "Please ensure the file path is defined"
    
    prefill_time = (prefill_metrics['prefill_end'] -  prefill_metrics['prefill_start'])* 1000
    ttft = prefill_metrics['ttft'] *  1000
    decode_time = decode_metrics['decode_time'] * 1000
    itl = decode_metrics['decode_time'] / decode_metrics['num_tokens']
    e2e_latency = decode_metrics['decode_end'] - prefill_metrics['prefill_start']
    
    print(f'prefill_time = {prefill_time:.3f}ms')
    print(f'ttft = {ttft:.3f}ms')
    print(f'decode_time = {decode_time:.3f}ms')
    print(f'itl = {itl * 1000:.3f} ms')
    print(f'e2e_latency = {e2e_latency*1000:.3f}ms')
    print(f'Peak memory usage = {decode_metrics['peak_memory_usage']:.3f} GB')

    #create a dict and save

    metrics = {
        'prefill_time' : prefill_time,
        'ttft' :  ttft,
        'decode_time' : decode_time,
        'itl' : itl,
        'e2e_latency' : e2e_latency,
        'peak_memory_usage_in_GB' : decode_metrics['peak_memory_usage']
    }

    with open(file_path, "w") as file:
        json.dump(metrics, file)
        print(f'Metrics saved')