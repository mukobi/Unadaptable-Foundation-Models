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


def get_huggingface_data(
    dataset_path: str,
    dataset_subset: str,
    batch_size: int = 128, 
    # test_batch_size: int = 1000
) -> torch.utils.data.DataLoader:
    """
    Loads a Hugging Face dataset given its path and subset.
    NOTE: you may have to split the dataset after loading.

    Args:
        dataset_path (str): The path to the Hugging Face dataset.
        dataset_subset (str): The subset of the dataset to load.
        batch_size (int, optional): The batch size for training data. Defaults to 128.
        test_batch_size (int, optional): The batch size for test data. Defaults to 1000.

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader object containing the loaded dataset.
    """

    dataset = huggingface_datasets.load_dataset(dataset_path, dataset_subset)

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # test_dataset = torch.utils.data.DataLoader(
    #     dataset['test'],
    #     batch_size=test_batch_size,
    #     shuffle=True,
    # )

    return dataset_loader