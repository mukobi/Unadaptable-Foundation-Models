import os
import logging
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger()

class MLPNet(nn.Module):
    """N-layer MLP model with dropout."""

    def __init__(self, hidden_layer_dims=(1024, 1024), dropout=0.2):
        super(MLPNet, self).__init__()
        # Dynamically create layers based on hidden_layer_dims
        prev_dim = 784
        layers = []
        for dim in hidden_layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # TODO test without
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

class HuggingFaceModel:

    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta", device="cuda") -> None:
        # On CAIS cluster, use /data/public_models/huggingface if available
        if os.path.exists(f"/data/public_models/huggingface/{model_name}"):
            model_name = f"/data/public_models/huggingface/{model_name}"
            logger.info(f"Using existing model on CAIS cluter: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.device = device
    
    def __call__(self, x: str | List[str] | List[List[str]]):
        x = self.tokenizer(x, return_tensors="pt").to(self.device)
        return self.model(**x).logits
    
    def detokenize(self, x) -> List[str]:
        return self.tokenizer.batch_decode(x)

def load_model(model_name: str, device="cuda"):
    if model_name.lower() == "mlp":
        model = MLPNet().to(device)
    else:
        model = HuggingFaceModel(model_name, device)
    return model
