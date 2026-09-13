import json
import torch
from safetensors.torch import load_file
import numpy
from transformers import AutoTokenizer

from model import QwenForCausalLM

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


with open("./qwen2.5-0.5b-Instruct/config.json") as f:
    config = json.load(f)
config['head_dim'] = int(config['hidden_size'] / config['num_attention_heads'])


model = QwenForCausalLM(config = config).to(device)

weights = load_file(
    "./qwen2.5-0.5b-Instruct/model.safetensors"
)
model.load_state_dict(weights, strict = True)
print(f'Model Weights loaded\n')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

prompt = "Terence Tao recently said that millenial maths problems are like non-renewable energy. AI is becoming"

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

sample_config = {
    'temperature' : 1.0,
    'top_k' : 50,
    'top_p' : 0.9
}


text = model.generate(tokenizer, input_ids, max_new_tokens = 1024, temp = sample_config['temperature'], top_k = sample_config['top_k'], top_p = sample_config['top_p'], use_cache = False)


print(text)
