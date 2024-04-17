from torch import nn
from torch.nn import functional as F


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


def load_model(model_name: str, device: str) -> nn.Module:
    if model_name == "MLP":
        model = MLPNet().to(device)
    else:
        raise ValueError(f"Unknown model type {model_name}")
    return model
