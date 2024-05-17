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


def get_hf_data(
    dataset_identifier: str,  # Dataset name and subset name
    seed: int = 42,
    # batch_size: int = 128, 
    # test_batch_size: int = 100,
) -> huggingface_datasets.Dataset:
    """
    Retrieves a DatasetDict object with splits 'train' and 'validation' for a given dataset identifier.
    Creates 'validation' split if it does not exist.

    Args:
        dataset_identifier (str): The identifier for the dataset. Supported identifiers are:
            - "cyber" for the 'cais/wmdp-corpora' dataset with the 'cyber-forget-corpus' subset.
            - "harmfulqa" for the 'declare-lab/HarmfulQA' dataset.
            - "toxic" for the 'allenai/real-toxicity-prompts' dataset.
            - "pile" for the 'NeelNanda/pile-10k' dataset.
        seed (int, optional): The seed for shuffling the dataset. Defaults to 42.

    Returns:
        torch.utils.data.DataLoader: A DataLoader object containing the specified dataset.

    Raises:
        ValueError: If the dataset identifier is not recognized.

    Note:
        - For the "cyber" dataset, the subset name is fixed to 'cyber-forget-corpus' and the split is 'train'.
        - For the "harmfulqa", "toxic", and "pile" datasets, there is no subset name and the split is 'train'.
        - If you would like to add support for a new dataset, please contact owen-yeung.
    """

    if dataset_identifier == "cyber":
        dataset_name = 'cais/wmdp-corpora'
        subset_name = 'cyber-forget-corpus'  # only 1k rows, test_batch shouldn't exceed
        split = 'train'

    elif dataset_identifier == "harmfulqa":
        dataset_name = 'declare-lab/HarmfulQA'
        subset_name = None
        split = 'train'

    elif dataset_identifier == "toxic":
        dataset_name = 'allenai/real-toxicity-prompts'
        subset_name = None
        split = 'train'

    elif dataset_identifier == "pile":
        dataset_name = 'NeelNanda/pile-10k'
        subset_name = None
        split = 'train'

    else:
        raise ValueError(
            'Dataset identifier not recognized. See docstring for supported identifiers.'
            'Contact owen-yeung if you would like a new dataset supported.'
        )

    dataset = huggingface_datasets.load_dataset(dataset_name, subset_name)[split].shuffle(seed=seed)

    if 'validation' not in dataset:
        # dataset = dataset.train_test_split(test_size=0.1)
        dataset = huggingface_datasets.DatasetDict({
            'train': dataset.select(range(int(0.9 * len(dataset)))),  # 90% for training
            'validation': dataset.select(range(int(0.9 * len(dataset)), len(dataset)))  # remaining 10% for testing
        })
        # assert 'validation' in dataset
        # assert 'train' in dataset

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    return dataset

# def DEPRECATED_get_huggingface_data(
#     dataset_path: str,
#     dataset_subset: str,
#     batch_size: int = 128, 
#     test_batch_size: int = 100,
#     test_num_rows: int = None,
# ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
#     """
#     Currently supports: (TODO add parameters to call each dataset/subset)
#     - WMDP
#     - HarmfulQA
#     - Pile

#     [DEPRECATED DOCSTRING]

#     Loads a dataset from Hugging Face Datasets library and returns train and test data loaders. Validation set not supported.
#     If only train split is available, the data is split into train and test sets based on the test_num_rows parameter. 

#     Args:
#         dataset_path (str): The path to the dataset.
#         dataset_subset (str): The name of the dataset subset to load.
#         batch_size (int, optional): The batch size for the train data loader. Defaults to 128.
#         test_batch_size (int, optional): The batch size for the test data loader. Defaults to 100.
#         test_num_rows (int, optional): The number of rows to use for the test set. Required if only train split is available.

#     Returns:
#         Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing the train and test data loaders.
#     """

#     dataset = huggingface_datasets.load_dataset(dataset_path, dataset_subset)
#     splits = dataset.keys()
#     if list(splits) == ['train']:
#         assert test_num_rows is not None, 'only train split is available. test_num_rows must be provided to split the data into train and test sets.'
#         assert test_num_rows < len(dataset['train']), 'test_num_rows must be less than the number of rows in the dataset.'
#         assert test_num_rows > 0, 'test_num_rows must be greater than 0.'
#         assert isinstance(test_num_rows, int), 'test_num_rows must be an integer.'
#         assert test_batch_size <= test_num_rows, 'test_batch_size must be less than or equal to test_num_rows.'
#         num_rows = len(dataset['train'])
#         train_num_rows = num_rows - test_num_rows
#         assert batch_size <= train_num_rows, 'batch_size must be less than or equal to the number of rows in the train set.'

#         train_set = dataset['train'].select(range(train_num_rows))
#         test_set = dataset['train'].select(range(train_num_rows, num_rows))

#     elif 'train' in splits and 'test' in splits:
#         train_set = dataset['train']
#         test_set = dataset['test']
#         assert batch_size <= len(train_set), 'batch_size must be less than or equal to the number of rows in the train set.'
#         assert test_batch_size <= len(test_set), 'test_batch_size must be less than or equal to the number of rows in the test set.'

#     else:
#         print(f'Available splits: {splits}')
#         raise ValueError('Dataset must have either a train or test split or both')

#     train_loader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#     )

#     test_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=test_batch_size,
#         shuffle=True,
#     )

#     return train_loader, test_loader
