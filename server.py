from fastapi import FastAPI
import torch
import yaml
from transformers import AutoTokenizer
import threading


model_lock = threading.Lock()

from infer import load_model


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

assert config is not None, "yaml file not loaded"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = load_model(path_config= config['path_config'],device=device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

assert model is not None, "Model not Loaded"
assert tokenizer is not None, "Tokenizer not loaded"


sampling_config = config['sample_config']

app = FastAPI()

def tokenize_input(prompt : str):
    assert prompt is not None

    input_ids = tokenizer.encode(prompt, return_tensors = "pt").to(device)
    print('Tokenization complete\n')
    return input_ids

@app.post("/generate")
def generate(prompt : str):

    input_ids = tokenize_input(prompt)
    with model_lock:
        text = model.generate(
            tokenizer, input_ids, temp = sampling_config['temperature'],
                        top_k = sampling_config['top_k'], top_p = sampling_config['top_p'], use_cache = True, max_new_tokens = 512
        )

    return {"answer" : text}



