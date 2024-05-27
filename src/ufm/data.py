from typing import Callable, Tuple, Union

import wandb
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase


def get_mnist_data(
    batch_size: int = 128, test_batch_size: int = 1000
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_train_loader = DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    mnist_test_loader = DataLoader(
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
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    fashion_mnist_train_loader = DataLoader(
        datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    fashion_mnist_test_loader = DataLoader(
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


def cyber_tokenizer(ufm_tokenizer, device) -> Callable:
    def tokenize_function(examples):
        return ufm_tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors='pt',
        ).to(device)

    return tokenize_function


def get_hf_data(
    dataset_identifier: str,  # Dataset name and subset name
    device: str,
    tokenizer: Union[PreTrainedTokenizerBase, AutoTokenizer],
) -> Tuple[DatasetDict, DataCollatorForLanguageModeling]:
    """
    Retrieves a DatasetDict object with splits 'train' and 'validation' for a given dataset identifier.
    Creates 'validation' split if it does not exist.

    Args:
        dataset_identifier (str): The identifier for the dataset. Supported identifiers are:
            - "cyber" for the 'cais/wmdp-corpora' dataset with the 'cyber-forget-corpus' subset.
            - "harmfulqa" for the 'declare-lab/HarmfulQA' dataset.
            - "toxic" for the 'allenai/real-toxicity-prompts' dataset.
            - "pile" for the 'NeelNanda/pile-10k' dataset.
        device: The device to move the dataset to.
        tokenizer: The tokenizer to use for tokenizing the dataset. Pulled from the UFMLanguageModel

    Returns:
        A tuple containing the dataset and data loader.

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
        remove_cols = ["text"]
        tokenize_function = cyber_tokenizer(tokenizer, device)

    elif dataset_identifier == "harmfulqa":
        dataset_name = 'declare-lab/HarmfulQA'
        subset_name = None
        split = 'train'
        remove_cols = ...
        tokenize_function = ...

    elif dataset_identifier == "toxic":
        dataset_name = 'allenai/real-toxicity-prompts'
        subset_name = None
        split = 'train'
        remove_cols = ...
        tokenize_function = ...

    elif dataset_identifier == "pile":
        dataset_name = 'NeelNanda/pile-10k'
        subset_name = None
        split = 'train'
        remove_cols = ...
        tokenize_function = ...

    else:
        raise ValueError(
            'Dataset identifier not recognized. See docstring for supported identifiers.'
            'Contact owen-yeung if you would like a new dataset supported.'
        )

    dataset = load_dataset(dataset_name, subset_name)[split].shuffle(seed=wandb.config.seed)

    if 'validation' not in dataset:
        # dataset = dataset.train_test_split(test_size=0.1)
        dataset = DatasetDict(
            {
                'train': dataset.select(range(int(0.9 * len(dataset)))),  # 90% for training
                'validation': dataset.select(range(int(0.9 * len(dataset)), len(dataset)))  # remaining 10% for testing
            }
        )

    # PADDING for Llama is weird
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_datasets, data_collator
