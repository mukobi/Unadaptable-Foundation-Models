from logging import Logger

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from wandb.sdk.wandb_config import Config

from src.ufm.data import get_dataset


def run_mnist_pretrain(model: nn.Module, device: str, config: dict, logger: Logger) -> nn.Module:
    """
    Training function.

    Returns a list of losses.
    """
    #  num_epochs=1, learning_rate=1e-3, gamma=0.7
    num_epochs = config["epochs"]
    learning_rate = config["lr"]
    gamma = config["gamma"]
    
    train_loader, test_loader = get_dataset(
        config["dataset"],
        config["batch_size"],
        config["test_batch_size"],
    )

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
            wandb.log({"pretrain/train_loss": loss.item()})
            losses.append(loss.item())
        if epoch % 1 == 0 or epoch == num_epochs:
            # Avg loss over this epoch
            avg_loss = sum(losses[-len(train_loader) :]) / len(train_loader)
            print(
                f"Train Epoch: {epoch}/{num_epochs} ({100 * epoch / num_epochs:.2f}%) Average Loss: {avg_loss:.6f}"
            )
        scheduler.step()
    progress_bar.close()
    return model
