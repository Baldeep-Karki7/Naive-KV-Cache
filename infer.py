import json
import torch
from safetensors.torch import load_file
import numpy
from transformers import AutoTokenizer

from generation_model import QwenForCausalLM


def load_model(path_config, device):
    with open(f'{path_config['model_config_path']}') as f:
        config = json.load(f)

    config['head_dim'] = int(config['hidden_size'] / config['num_attention_heads'])

    #load the weights
    weights = load_file(
        path_config['weights_path'])

    model = QwenForCausalLM(config = config, device = device)

    #load state dict
    model.load_state_dict(weights, strict = True)
    print(f'Model Weights loaded\n')
    return model.to(device = device)


