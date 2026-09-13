import torch
import time

class KV_cache:
    def __init__(self, config, device, dtype = torch.float16):
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['head_dim']
        num_layers = config['num_hidden_layers']

        B = 1 #batch_size
        capacity = 512
        
        self.cache = [
            LayerCache(
                batch_size = B,
                num_kv_heads = num_kv_heads,
                timesteps = capacity,
                head_dim = head_dim,
                device = device,
                dtype = dtype
            ) 
            for _ in range(num_layers)]

    def update(self, layer_idx, k, v):
        self.cache[layer_idx].update(k, v)

class LayerCache:
    def __init__(self, batch_size, num_kv_heads, timesteps, head_dim, device, dtype):
        self.k = torch.zeros(batch_size,num_kv_heads, timesteps, head_dim, device = device, dtype=dtype)
        self.v = torch.zeros(batch_size,num_kv_heads, timesteps, head_dim, device = device, dtype=dtype)
        self.seq_len = 0
        self.capacity = timesteps

    def update(self, k,v) :
        if self.seq_len >= self.capacity:
            #new capacity 
            new_capacity_shape = (self.k.shape[0], self.k.shape[1], self.capacity + 256, self.k.shape[3])

            self.capacity += 256


            new_k = torch.zeros(new_capacity_shape,device = self.k.device, dtype = self.k.dtype)

            new_v = torch.zeros(new_capacity_shape,device = self.v.device, dtype = self.v.dtype)

            #copy the old keys and values
            new_k[:, :, :self.seq_len, :] = self.k[:, :, :self.seq_len, :]
            new_v[:, :, :self.seq_len, :] = self.v[:, :, :self.seq_len, :]

            #set new k to self.k
            self.k = new_k
            self.v = new_v

        self.k[:, :, self.seq_len: self.seq_len + 1, :] = k
        self.v[:, :, self.seq_len: self.seq_len + 1, :] = v

        self.seq_len +=1
        return self.k, self.v


        

        