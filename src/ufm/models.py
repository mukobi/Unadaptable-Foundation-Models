from logging import Logger

import torch
from torch import nn
from torch.nn import functional as F
from wandb.sdk.wandb_config import Config
import wandb

from src.ufm import pretrain

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


def load_pretrained_model(model_name: str, device: str, logger: Logger, pretrain_config: dict = None) -> nn.Module:
    """
    Load a pretrained model based on the model name.
    For models that require pretraining (Ex. mnist), the pretrain_config is used to pretrain the model. Uses cached models if available.
    For llms, we use pretrained models from huggingface.
    """
    if model_name == "MLP_MNIST":
        try:
            artifact = wandb.run.use_artifact("mnist_pretrained_model:latest", type='model')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + "/mnist_pretrained_model.pth"
            return torch.load(artifact_dir)
        except Exception as e:
            print(e)
            logger.info("Pretrained model not found. Pretraining model...")
            model = MLPNet().to(device)
            model = pretrain.run_mnist_pretrain(model, device, pretrain_config, logger)
            torch.save(model, "mnist_pretrained_model.pth")
            artifact = wandb.Artifact("mnist_pretrained_model", type="model")
            artifact.add_file("mnist_pretrained_model.pth")
            wandb.run.log_artifact(artifact)
            return model
            
    else:
        raise ValueError(f"Unknown model type {model_name}")
    
