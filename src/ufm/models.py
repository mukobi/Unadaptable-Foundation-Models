import os
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class MLPNet(nn.Module):
    """N-layer MLP model with dropout."""

    def __init__(self, hidden_layer_dims=[1024, 1024], dropout=0.2):
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

    def __init__(self, model_name: str = "zephyr/zephyr-7b-beta") -> None:
        if os.path.exists(f"/data/public_models/{model_name}"):
            model_name = f"/data/public_models/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def forward(self, x):
        x = self.tokenizer(x, return_tensors="pt")
        return self.model(**x)