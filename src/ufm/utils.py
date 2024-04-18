import random

import numpy as np
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR
import wandb

from ufm.data import *
from ufm.models import *
from ufm.unadapt import *


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

def check_base_results_saved(base_name: str) -> bool:
    """Check if the base model results are already saved."""
    api = wandb.Api()
    try:
        run = api.runs(f"unadaptable-foundation-models", filters={"tags": base_name})
        if len(run) == 0:
            return False
        return True
    except:
        return False


def train(model, device, train_loader, num_epochs=1, learning_rate=1e-3, gamma=0.7) -> list:
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
