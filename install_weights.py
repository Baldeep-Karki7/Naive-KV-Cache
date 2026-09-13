import torch
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./qwen2.5-0.5b-Instruct",
)