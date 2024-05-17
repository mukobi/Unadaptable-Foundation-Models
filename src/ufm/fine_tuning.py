"""
Scripts for fine-tuning on the harmful datasets
"""

import logging
from logging import Logger
# from models import HuggingFaceModel  # HF model is a wrapper with model AND tokenizer
from typing import TYPE_CHECKING

from omegaconf.errors import ValidationError
from transformers import Trainer, TrainingArguments

from ufm.data import get_hf_data

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger()

FINETUNE_DATASETS = [
    "cyber",
    "pile",
    "harmfulqa",
    "toxic",
]


def validate_finetune_cfg(cfg: "DictConfig") -> "DictConfig":
    """
    Validate config for finetuning
    """
    # Validate config has dataset name and is either 'cyber' or 'pile'
    if "dataset" not in cfg:
        raise ValidationError("Missing 'dataset' in finetuning config")

    dataset = cfg["dataset"]
    if cfg["dataset"] not in FINETUNE_DATASETS:
        raise ValidationError(
            f"Finetune dataset must be one of {FINETUNE_DATASETS}"
        )

    return cfg


def run_fine_tune(
    # model_unadapted: HuggingFaceModel,
    model_unadapted,
    config: "DictConfig",
    training_task: str = "supervised-fine-tuning"
):
    """
    Fine tunes model_unadapted on the dataset specified in config
    Returns fine-tuning validation losses
    Only supports supervised-fine-tuning for now
    Only supports text-only datasets (cyber and pile) for now

    'config' is specifically the 'finetune' struct of global config
    """
    # Validate
    config = validate_finetune_cfg(config)

    # column_name = config['column']
    dataset_identifier = config['dataset']

    tokenizer = model_unadapted.tokenizer
    model = model_unadapted.model

    # Load dataset
    logger.info(f"Loading dataset {config['dataset']} ...")
    dataset = get_hf_data(dataset_identifier)  # TODO batch size config etc

    # assert train splits exist
    assert 'train' in dataset

    # # If no validation set create one
    # if 'validation' not in dataset:
    #     # dataset = dataset.train_test_split(test_size=0.1)
    #     DatasetDict({
    #         'train': dataset['train'].shuffle(seed=42).select(range(int(0.9 * len(dataset['train'])))),  # 90% for training
    #         'test': dataset['train'].shuffle(seed=42).select(range(int(0.9 * len(dataset['train'])), len(dataset['train'])))  # remaining 10% for testing
    #     })
    #     assert 'validation' in dataset

    if training_task == "supervised-fine-tuning":
        if dataset_identifier in ['cyber', 'pile']:
            column_name = 'text'

        def tokenize_function(examples):
            # TODO check if padding and truncation are correct
            return tokenizer(examples[column_name], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

        # TODO training_args should take in relevant config
        training_args = TrainingArguments(
            # report_to="wandb", by default it reports to all connected loggers
            evaluation_strategy="steps",
            eval_steps="10",
        )

        # training_args with relevant config
        # training_args = TrainingArguments(
        #     output_dir="./results",
        #     num_train_epochs=3,
        #     per_device_train_batch_size=8,
        #     per_device_eval_batch_size=8,
        #     warmup_steps=500,
        #     weight_decay=0.01,
        #     logging_dir="./logs",
        #     logging_steps=10,
        # )

        trainer = Trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        logger.info("Fine-tuning model...")
        trainer.train()

        # eval_results = trainer.evaluate()
        eval_loss = trainer.state.log_history['eval_loss']

        return eval_loss  # validation loss for fine-tuning

    else:
        raise NotImplementedError(
            f"Only supervised-fine-tuning fine-tuning is supported for now. Got {training_task} instead."
        )
