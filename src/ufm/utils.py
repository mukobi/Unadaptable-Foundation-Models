import logging
import random

import numpy as np

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ValidationError
from torch.optim.lr_scheduler import StepLR

from ufm.data import *
from ufm.models import *
from ufm.unadapt import *


logger = logging.getLogger()


def validate_config(cfg: DictConfig) -> DictConfig:
    """
    Apply suite of config validations, raising
    """
    # Logging
    verbosity = cfg.get("verbosity", 1)
    if verbosity == 2:
        # Debug, most verbose
        logger.setLevel(logging.DEBUG)
    elif verbosity == 1:
        # All but debug
        logger.setLevel(logging.INFO)
    elif verbosity == 0:
        # Supress warnings; Most quiet; Still writes errors
        logger.setLevel(logging.ERROR)
    else:
        raise ValidationError(f"Invalid value for 'verbosity': {verbosity}")

    # Tags must be list if provided
    if isinstance(cfg.get("tags", None), str):
        cfg["tags"] = [cfg["tags"]]

    # Print info
    logger.info(OmegaConf.to_yaml(cfg))

    return cfg


def set_seed(seed: int) -> None:
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name: str, batch_size: int = 64, test_batch_size: int = 1000):
    if dataset_name.lower() == "mnist":
        return get_mnist_data(batch_size, test_batch_size)
    elif dataset_name.lower() == "fashionmnist":
        return get_fashion_mnist_data(batch_size, test_batch_size)
    else:
        raise NotImplementedError


def get_unadaptable_model(
    model: nn.Module, unadapt_config: DictConfig, device, train_loader
) -> nn.Module:
    if unadapt_config.method == "prune":
        return apply_pruning(model, unadapt_config.prune_percentage)
    elif unadapt_config.method == "rescale":
        return apply_weight_rescaling(model, unadapt_config.rescale_factor)
    elif unadapt_config.method == "zeroth":
        return apply_zeroth_order_learning(model, unadapt_config, device, train_loader)
    elif unadapt_config.method == "gradient":
        return apply_gradient_learning(model, unadapt_config, device, train_loader)
    else:
        raise NotImplementedError


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
        for data, target in train_loader:
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
