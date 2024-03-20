from typing import Tuple

import datasets as huggingface_datasets
import torch.utils.data
from torchvision import datasets, transforms


def get_mnist_data(
    batch_size: int = 128, test_batch_size: int = 1000
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )

    return mnist_train_loader, mnist_test_loader


def get_fashion_mnist_data(
    batch_size: int = 128, test_batch_size: int = 1000
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    fashion_mnist_train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    fashion_mnist_test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )

    return fashion_mnist_train_loader, fashion_mnist_test_loader
