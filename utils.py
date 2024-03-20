from functools import partial
from typing import Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data import *
from models import *
from unadapt import *


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, device="cuda"):
    if model_name.lower() == "mlp":
        model = MLPNet().to(device)
    else:
        raise NotImplementedError
    return model


def get_dataset(dataset_name: str, batch_size: int = 64, test_batch_size: int = 1000):
    if dataset_name.lower() == "mnist":
        return get_mnist_data(batch_size, test_batch_size)
    elif dataset_name.lower() == "fashionmnist":
        return get_fashion_mnist_data(batch_size, test_batch_size)
    else:
        raise NotImplementedError


def get_unadapt_method(
    unadapt_config,
) -> Callable[[nn.Module, torch.utils.data.DataLoader], None]:
    if unadapt_config.method == "prune":
        return partial(
            apply_pruning_to_model, prune_percentage=unadapt_config.prune_percentage
        )
    elif unadapt_config.method == "rescale":
        return partial(
            apply_weight_rescaling_to_model,
            rescale_factor=unadapt_config.rescale_factor,
        )
    else:
        raise NotImplementedError


def load_config(config_path: str) -> DictConfig:
    """Load a config file of a given path (absolute or relative to cwd)."""
    conf = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    print(OmegaConf.to_yaml(conf))
    return conf


def train(model, device, train_loader, num_epochs=1, learning_rate=1e-3, gamma=0.7):
    """
    Training function.

    Returns a list of losses.
    """

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    model.train()
    num_total_batches = len(train_loader) * num_epochs
    progress_bar = tqdm(total=num_total_batches, position=0, leave=True)
    losses = []
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            progress_bar.update()
            losses.append(loss.item())
        if epoch % 1 == 0 or epoch == num_epochs:
            # Avg loss over this epoch
            avg_loss = sum(losses[-len(train_loader) :]) / len(train_loader)
            print(
                f"Train Epoch: {epoch}/{num_epochs} ({100 * epoch / num_epochs:.2f}%) Average Loss: {avg_loss:.6f}"
            )
        scheduler.step()
    progress_bar.close()
    return losses


def test(model, device, test_loader):
    """Testing function."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    return test_loss, correct / len(test_loader.dataset)
