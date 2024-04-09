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

# TODO: Only support the datasets we plan to use (and specify their train/test splits etc manually for each)
# No need to support any arbitrary hf dataset
def get_huggingface_data(
    dataset_path: str,
    dataset_subset: str,
    batch_size: int = 128, 
    test_batch_size: int = 100,
    test_num_rows: int = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Loads a dataset from Hugging Face Datasets library and returns train and test data loaders. Validation set not supported.
    If only train split is available, the data is split into train and test sets based on the test_num_rows parameter. 

    Args:
        dataset_path (str): The path to the dataset.
        dataset_subset (str): The name of the dataset subset to load.
        batch_size (int, optional): The batch size for the train data loader. Defaults to 128.
        test_batch_size (int, optional): The batch size for the test data loader. Defaults to 100.
        test_num_rows (int, optional): The number of rows to use for the test set. Required if only train split is available.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing the train and test data loaders.
    """

    dataset = huggingface_datasets.load_dataset(dataset_path, dataset_subset)
    splits = dataset.keys()
    if list(splits) == ['train']:
        assert test_num_rows is not None, 'only train split is available. test_num_rows must be provided to split the data into train and test sets.'
        assert test_num_rows < len(dataset['train']), 'test_num_rows must be less than the number of rows in the dataset.'
        assert test_num_rows > 0, 'test_num_rows must be greater than 0.'
        assert isinstance(test_num_rows, int), 'test_num_rows must be an integer.'
        assert test_batch_size <= test_num_rows, 'test_batch_size must be less than or equal to test_num_rows.'
        num_rows = len(dataset['train'])
        train_num_rows = num_rows - test_num_rows
        assert batch_size <= train_num_rows, 'batch_size must be less than or equal to the number of rows in the train set.'
    
        train_set = dataset['train'].select(range(train_num_rows))
        test_set = dataset['train'].select(range(train_num_rows, num_rows))

    elif 'train' in splits and 'test' in splits:
        train_set = dataset['train']
        test_set = dataset['test']
        assert batch_size <= len(train_set), 'batch_size must be less than or equal to the number of rows in the train set.'
        assert test_batch_size <= len(test_set), 'test_batch_size must be less than or equal to the number of rows in the test set.'

    else:
        print(f'Available splits: {splits}')
        raise ValueError('Dataset must have either a train or test split or both')

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
    )

    return train_loader, test_loader