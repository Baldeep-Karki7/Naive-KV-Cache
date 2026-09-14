import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cache import LayerCache, KV_cache
from save_metrics import sample_next_token, print_and_save_metrics


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ============================================================
# Rotary Positional Embedding
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=32768, theta=1000000.0):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids):
        """
        x:
            [batch, num_heads, seq_len, head_dim]

        position_ids:
            [batch, seq_len]
        """

        # [batch, seq_len, head_dim / 2]
        freqs = torch.einsum(
            "bi,j->bij",
            position_ids.float(),
            self.inv_freq
        )

        # [batch, seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


# ============================================================
# GQA helper
# ============================================================

def repeat_kv(hidden_states, n_rep):
    """
    hidden_states:
        [batch, num_kv_heads, seq_len, head_dim]

    Returns:
        [batch, num_attention_heads, seq_len, head_dim]
    """

    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    hidden_states = hidden_states[:, :, None, :, :]

    hidden_states = hidden_states.expand(
        batch,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim
    )

    return hidden_states.reshape(
        batch,
        num_kv_heads * n_rep,
        seq_len,
        head_dim
    )


# ============================================================
# Qwen Attention
# ============================================================

class QwenAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        rope_theta,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.num_key_value_groups = (
            num_attention_heads // num_key_value_heads
        )

        # Qwen uses separate projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True,)

        self.k_proj = nn.Linear( hidden_size, num_key_value_heads * head_dim, bias=True,)

        self.v_proj = nn.Linear( hidden_size, num_key_value_heads * head_dim, bias=True,)

        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False,)

        self.rotary_emb = RotaryEmbedding( head_dim=head_dim, max_position_embeddings=max_position_embeddings, theta=rope_theta,)

    def forward(self, hidden_states, position_ids):

        batch_size, seq_len, _ = hidden_states.shape

        # ----------------------------------------------------
        # QKV projections
        # ----------------------------------------------------

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # ----------------------------------------------------
        # Reshape into heads
        # ----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        cos, sin = self.rotary_emb(
            q,
            position_ids,
        )

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )

        # ----------------------------------------------------
        # GQA
        # ----------------------------------------------------

        k = repeat_kv(k, self.num_key_value_groups)

        v = repeat_kv(v, self.num_key_value_groups)

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        scores = torch.matmul(q, k.transpose(-2, -1) )

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min,
        )

        attention_weights = F.softmax(scores, dim=-1,)

        output = torch.matmul(attention_weights, v,)

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        )

        output = self.o_proj(output)

        return output


# ============================================================
# Qwen MLP / SwiGLU
# ============================================================

class QwenMLP(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x):

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x

# ============================================================
# Decoder Layer
# ============================================================

class QwenDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(
            hidden_size,
            rms_norm_eps,
        )

        self.self_attn = QwenAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            rms_norm_eps,
        )

        self.mlp = QwenMLP(
            hidden_size,
            intermediate_size,
        )

    def forward(self, hidden_states, position_ids):

        # ----------------------------------------------------
        # Attention block
        # ----------------------------------------------------

        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states
        )

        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
        )

        hidden_states = residual + hidden_states

        # ----------------------------------------------------
        # MLP block
        # ----------------------------------------------------

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )

        hidden_states = self.mlp(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================
# Qwen Transformer
# ============================================================

class QwenModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList([
            QwenDecoderLayer(
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                head_dim=config["head_dim"],
                max_position_embeddings=config["max_position_embeddings"],
                rope_theta=config["rope_theta"],
                rms_norm_eps=config["rms_norm_eps"],
            )
            for _ in range(config["num_hidden_layers"])
        ])

        self.norm = RMSNorm(
            config["hidden_size"],
            config["rms_norm_eps"],
        )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        hidden_states = self.embed_tokens(
            input_ids
        )

        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        # ----------------------------------------------------
        # Transformer layers
        # ----------------------------------------------------

        for layer in self.layers:

            hidden_states = layer(
                hidden_states,
                position_ids,
            )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        hidden_states = self.norm(
            hidden_states
        )

        return hidden_states


# ============================================================
# Full Causal LM
# ============================================================

class QwenForCausalLM(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config =  config
        self.model = QwenModel(config)
        
        self.vocab_size = config["vocab_size"]
        self.kv_cache = KV_cache(config, device)

    def forward(self, input_ids):

        # [batch, seq_len, hidden_size]
        hidden_states = self.model(input_ids)

        logits = F.linear(
            hidden_states,
            self.model.embed_tokens.weight
        )
        return logits

    def warmup_without_cache(self, input_ids):
        self.eval()
        print(f'warming up without cache for 5 steps\n')
        with torch.no_grad():
            for _ in range(5):
                self(input_ids)
    
    def prefill_without_cache(self, input_ids, temp, top_p, top_k):
        #prefill
        print(f'Executing Prefill')
        self.eval()

        with torch.no_grad():
            #sync
            synchronize()
            prefill_start = time.perf_counter()
        
            logits = self(
                input_ids = input_ids
            )

            #torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            prefill_time = prefill_end - prefill_start
            
            next_token = sample_next_token(logits[:, -1, :], temperature=temp, top_k=top_k, top_p=top_p)
        
            synchronize()
            first_token_time = time.perf_counter()
            ttft = first_token_time - prefill_start
        
            prefill_metrics = {
                "prefill_start" : prefill_start,
                "prefill_end" : prefill_end,
                "prefill_time" : prefill_time,
                "ttft" : ttft
            }

            input_ids = torch.cat((input_ids, next_token), dim = 1)
            return prefill_metrics, input_ids

    def decode_without_cache(self, input_ids, temp, top_k, top_p, max_new_tokens):
        self.eval()
        print(f'Decoding..')
    
        token_times = []
    
        with torch.no_grad():

            synchronize()
            decode_start = time.perf_counter()
    
            for step in range(max_new_tokens - 1):

                # if step % 100 == 0:
                #     print(f'Decoding on step {step}\n')
                synchronize()
                token_start = time.perf_counter()
    
                # Recompute the entire sequence
                logits = self(
                    input_ids=input_ids,
                )
    
                # logits = outputs.logits
    
                next_token = sample_next_token(logits[:, -1, :], temperature=temp, top_k=top_k, top_p=top_p)
    
                synchronize()
                token_end = time.perf_counter()
    
                token_times.append(token_end - token_start)
    
                input_ids = torch.cat(
                    (input_ids, next_token),
                    dim=-1
                )
    
                # print(next_token.item())
    
                if next_token.item() == self.model.config['eos_token_id']:
                    break
    
            synchronize()
            decode_end = time.perf_counter()
        
        
        peak_memory_usage = torch.accelerator.memory.max_memory_allocated()
        decode_metrics = {
            "decode_start": decode_start,
            "decode_end": decode_end,
            "decode_time": decode_end - decode_start,
            "token_times": token_times,
            "num_tokens": len(token_times),
            "avg_token_time": sum(token_times) / len(token_times),
            "tokens_per_sec": len(token_times) / (decode_end - decode_start),
            "peak_memory_usage" : peak_memory_usage /(1024 ** 3)
        }
    
        return input_ids, decode_metrics

    
    def calc_after_rotary_emb(self, q, k, v, device, prefill = False, seq_len = None):
        if prefill:
            assert isinstance(seq_len, int), "For prefill, seq len needs to be defined for causal mask"
        num_key_value_groups = int(self.config['num_attention_heads'] / self.config['num_key_value_heads'])
        k = repeat_kv(k, num_key_value_groups)
        v = repeat_kv(v, num_key_value_groups)

        # print(q.shape, k.shape, q.dtype, k.dtype)
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        )
    
        scores = scores / math.sqrt(self.config['head_dim'])
    
        if prefill:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len,
                                                device= device,
                                                dtype=torch.bool,),diagonal=1,)
            scores = scores.masked_fill(causal_mask,
                                        torch.finfo(scores.dtype).min,)
    
        # print(scores.shape)
        attention_weights = F.softmax(scores, dim=-1,)
    
        output = torch.matmul(attention_weights, v)
        #return attention score
        return output
         
    
    def prefill_with_cache(self, input_ids, temp, top_k, top_p):
        B, T = input_ids.shape

        self.eval()
        with torch.no_grad():
            synchronize()
            prefill_start = time.perf_counter()

            hidden_states = self.model.embed_tokens(input_ids) # (1, T, 896)
            current_position = T
            # print(f'Position  = {current_position}')
            #positional_ids
            position_ids = torch.arange(T ,device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B,-1)

            for i, decoder_layer in enumerate(self.model.layers):
                # print(decoder_layer)
                residual = hidden_states
                ln1 = decoder_layer.input_layernorm(hidden_states)
        
                k = decoder_layer.self_attn.k_proj(ln1)
                v = decoder_layer.self_attn.v_proj(ln1)
                q = decoder_layer.self_attn.q_proj(ln1)

                k = k.view(B, T, self.config['num_key_value_heads'],-1).transpose(1,2)
                v = v.view(B, T, self.config['num_key_value_heads'],-1).transpose(1,2)
                q = q.view(B, T, self.config['num_attention_heads'],-1).transpose(1,2)
        
                #calc cos and sin
                cos, sin = decoder_layer.self_attn.rotary_emb(q, position_ids)
                q,k = apply_rotary_pos_emb(q, k, cos, sin)

                # print(k_buff.shape, v_buff.shape, cur_len)
                #update the kv cache
                self.kv_cache.cache[i].k[:, : ,:T, :] = k
                self.kv_cache.cache[i].v[:, :, :T, :] = v

                self.kv_cache.cache[i].seq_len = T

                #full causal attention score
                output = self.calc_after_rotary_emb(q, k, v,
                                      device = input_ids.device, prefill = True, seq_len = T)
        
                #merge the heads
                attn_output = output.transpose(1,2).contiguous()
                
                attn_output = attn_output.view(
                    B, T, self.config['num_attention_heads'] *  self.config['head_dim'])
                #(1, 18, 896)
        
                #project it
                attn_output = decoder_layer.self_attn.o_proj(attn_output) #(896, 896)
        
                #first residual
                hidden_states = residual + attn_output

                residual = hidden_states
                ln2 = decoder_layer.post_attention_layernorm(hidden_states)
                mlp_out = decoder_layer.mlp(ln2)

                #second residual
                hidden_states = residual + mlp_out

            #final norm applied once, after all layers
            hidden_states = self.model.norm(hidden_states)
            # return hidden_states

            #tied embedding weights
            logits = F.linear(
                hidden_states,
                self.model.embed_tokens.weight
            )
    
            #prefill end
            synchronize()
            prefill_end = time.perf_counter()
        
            # print(logits.shape)
            
        next_token = sample_next_token(logits[:, -1, :],temperature = temp,
                                       top_k = top_k, top_p = top_p)
        synchronize()
        first_token_time = time.perf_counter()
        
        ttft = first_token_time - prefill_start
        prefill_time = prefill_end - prefill_start

        prefill_metrics = {
                "prefill_start" : prefill_start,
                "prefill_end" : prefill_end,
                "prefill_time" : prefill_time,
                "ttft" : ttft
            }

        return next_token, current_position, prefill_metrics

    def warmup_with_cache(self, input_ids, temp, top_k, top_p):
        
        #create the cache here
        num_warmups = 5
        print(input_ids)
        B, T = input_ids.shape
        self.eval()
        
        for i in range(num_warmups):
            print(f'Warmup step = {i+1}')
            _, _, _ = self.prefill_with_cache(input_ids, temp = temp, top_p = top_p, top_k = top_k)

        print(f'Reset seq len to 0\n')
        for i in range(self.model.config["num_hidden_layers"]):
            self.kv_cache.cache[i].seq_len = 0
        
        synchronize()


    def decode_with_cache(self, next_token, current_position, temp, top_k, top_p, 
                          max_new_tokens = 2048):

        generated_count = 0
        generated_tokens = []
        token_times = []
        
        decode_start = time.perf_counter()
        self.eval()
    
        with torch.no_grad():
            while next_token.item() != 152643 and generated_count != max_new_tokens - 1:

                B, T = next_token.shape
                token_start = time.perf_counter()
    
                hidden_states = self.model.embed_tokens(next_token)
                position_ids = torch.tensor([[current_position]], device=next_token.device)
    
                for i, decoder_layer in enumerate(self.model.layers):
                    residual = hidden_states
                    ln1 = decoder_layer.input_layernorm(hidden_states)
    
                    # attention
                    k = decoder_layer.self_attn.k_proj(ln1)
                    v = decoder_layer.self_attn.v_proj(ln1)
                    q = decoder_layer.self_attn.q_proj(ln1)
    
                    k = k.view(B, 1, self.config['num_key_value_heads'], -1).transpose(1, 2)
                    v = v.view(B, 1, self.config['num_key_value_heads'], -1).transpose(1, 2)
                    q = q.view(B, 1, self.config['num_attention_heads'], -1).transpose(1, 2)
    
                    cos, sin = decoder_layer.self_attn.rotary_emb(q, position_ids)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    
                    #update the cache
                    # print(f'new token\n')
                    # print(k.shape, v.shape)
                    k, v = self.kv_cache.update(i, k, v)
    
                    output = self.calc_after_rotary_emb(
                        q, k, v,
                        device=next_token.device,
                        prefill=False,
                    )
    
                    # merge heads
                    attn_output = output.transpose(1, 2).contiguous()
                    attn_output = attn_output.view(
                        B, T, self.config['num_attention_heads'] * self.config['head_dim']
                    )
                    attn_output = decoder_layer.self_attn.o_proj(attn_output)
    
                    # first residual
                    hidden_states = residual + attn_output
    
                    # MLP block
                    residual = hidden_states
                    ln2 = decoder_layer.post_attention_layernorm(hidden_states)
                    mlp_out = decoder_layer.mlp(ln2)

                    # second residual
                    hidden_states = residual + mlp_out

                # final norm applied ONCE, after all layers
                hidden_states = self.model.norm(hidden_states)
    
                logits = F.linear(hidden_states, self.model.embed_tokens.weight)
    
                #sampling
                next_token = sample_next_token(logits[:, -1, :], temperature = temp,
                                               top_k = top_k, top_p = top_p)
    
                synchronize()
                token_end = time.perf_counter()
    
                generated_tokens.append(next_token.item())
                token_times.append(token_end - token_start)
                current_position += 1
                generated_count += 1
    
            synchronize()
            decode_end = time.perf_counter()
            
        peak_memory_usage = torch.accelerator.memory.max_memory_allocated()
        decode_metrics = {
            "decode_start": decode_start,
            "decode_end": decode_end,
            "decode_time": decode_end - decode_start,
            "token_times": token_times,
            "num_tokens": len(token_times),
            "avg_token_time": sum(token_times) / len(token_times),
            "tokens_per_sec": len(token_times) / (decode_end - decode_start),
            "itl" : (decode_end - decode_start) / len(generated_tokens),
            "peak_memory_usage" : peak_memory_usage / (1024 ** 3)
        }
    
        return generated_tokens, decode_metrics

                
    def generate(self, tokenizer, input_ids, temp, top_k, top_p, use_cache = False, max_new_tokens = 1024):
        
        input_string_ids = input_ids
        print(f'Status : use_cache is {use_cache}')
        
        #reset peak memory for every generate
        # torch.cuda.reset_peak_memory_stats()

        self.eval()
        if not use_cache:
            with torch.no_grad():
                #warmup for some steps
                self.warmup_without_cache(input_ids)
                
                # torch.cuda.reset_peak_memory_stats()
                
                #prefill without cache
                prefill_metrics, input_ids = self.prefill_without_cache(input_ids, temp = temp, top_k = top_k, top_p = top_p)
                #decode_without_cache(self, input_ids, temp, top_k, top_p, max_new_tokens=100)
                #decode without cache
                input_ids, decode_metrics = self.decode_without_cache(input_ids, temp = temp, top_k = top_k, top_p = top_p, max_new_tokens = 256)

        else:
            with torch.no_grad():
                
                #warmup with cache
                #cache is created inside and the same cache is used for this generation
                self.warmup_with_cache(input_ids, temp = temp, top_k = top_k, top_p = top_p)

                #after warmup reset the max memory
                # torch.cuda.reset_peak_memory_stats()
                
                #prefill
                next_token, current_position, prefill_metrics = self.prefill_with_cache(input_ids, temp, top_k, top_p)
                
                #decode
                generated_tokens, decode_metrics = self.decode_with_cache(
                    next_token = next_token, current_position = current_position,
                    temp = temp, top_k = top_k, top_p = top_p)

        if not use_cache:
            generated_tokens = input_ids.squeeze(0).tolist()
        
        print_and_save_metrics(prefill_metrics, decode_metrics, file_path = file_path, use_cache = use_cache)

            
        #deocode here
        print(f'Metrics saved to file {None}\n\n')
        text = tokenizer.decode(input_ids.squeeze(0).tolist())
        text+= tokenizer.decode(generated_tokens)
        return text