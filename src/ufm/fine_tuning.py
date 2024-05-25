"""
Scripts for fine-tuning on the harmful datasets
"""

import logging
from typing import Dict, Optional, Union

from omegaconf import DictConfig
from omegaconf.errors import ValidationError
from torch import nn
from transformers import (
    AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase, Trainer, TrainingArguments
)

from ufm.data import get_hf_data
from ufm.models import UFMLangaugeModel  # HF model is a wrapper with model AND tokenizer

logger = logging.getLogger(__name__)

FINETUNE_DATASETS = [
    "cyber",
    "pile",
    # NOT YET SUPPORTED, needs preprocessing
    # "harmfulqa", 
    # "toxic",
]


def validate_finetune_cfg(cfg: DictConfig) -> DictConfig:
    """
    Validate config for finetuning
    """
    # Validate config has dataset name and is either 'cyber' or 'pile'
    if "dataset" not in cfg:
        raise ValidationError("Missing 'dataset' in finetuning config")

    # dataset = cfg["dataset"]
    if cfg["dataset"] not in FINETUNE_DATASETS:
        raise ValidationError(
            f"Finetune dataset must be one of {FINETUNE_DATASETS}"
        )

    return cfg


def run_llm_fine_tune(
    model: Union["UFMLangaugeModel", "nn.Module"],
    config: dict,
    tokenizer: Optional[Union[PreTrainedTokenizerBase, AutoTokenizer]] = None,
) -> Dict[str, list]:
    """
    Finetunes a model on the dataset specified in config
    Returns fine-tuning validation losses
    Only supports supervised-fine-tuning for now
    Only supports text-only datasets (cyber and pile) for now

    :param model: UFMLanguageModel or nn.Module. If nn.Module, tokenizer must be provided
    :param config: DictConfig with finetuning config
    :param tokenizer: Optional tokenizer. Required if model is nn.Module
    :returns: dict of evals
    """
    if isinstance(model, nn.Module):
        if tokenizer is None:
            raise ValueError(
                f"If not UFMLanguageModel, tokenizer must be provided. Received {type(model)}."
            )
        nn_model = model
    elif isinstance(model, UFMLangaugeModel):
        nn_model = model.model
        tokenizer = model.tokenizer
    else:
        # Ambiguous
        raise ValueError(f"Model must be either UFMLanguageModel or nn.Module. Received {type(model)}.")

    config = DictConfig(config)
    # Validate
    config = validate_finetune_cfg(config)

    # column_name = config['column']
    dataset_identifier = config.dataset

    # Load dataset
    logger.info(f"Loading dataset {config.dataset} ...")
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

    if config.training_task == "supervised-fine-tuning":
        if dataset_identifier in ['cyber', 'pile']:
            column_name = 'text'
        else:
            raise NotImplementedError(
                f"Only text datasets are supported for now. Got {dataset_identifier} instead."
            )

        def tokenize_function(examples):
            # TODO check if padding and truncation are correct
            return tokenizer(
                examples[column_name],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors='pt',
            ).to('cpu')

        # PADDING for Llama is weird
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=column_name)
        tokenized_datasets.set_format("torch")

        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # See here for more info on the different params
        # https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#trainingarguments
        training_args = TrainingArguments(
            output_dir="./",
            num_train_epochs=config.epochs,
            learning_rate=config.lr,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.test_batch_size,
            save_strategy="no",
            # label_names=["text"],
            report_to="wandb",  # by default it reports to all connected loggers
            evaluation_strategy="steps",
            eval_steps=3,
            max_steps=6,
        )

        # training_args with relevant config
        # training_args = TrainingArguments(w
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
            model=nn_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # tokenizer=tokenizer,
            data_collator=data_collator,
        )

        logger.info("Fine-tuning model...")
        trainer.train()

        # eval_results = trainer.evaluate()
        logger.debug(f"Training log history:\n{trainer.state.log_history}")
        # eval_loss = trainer.state.log_history['eval_loss']

        evals = {
            "eval_loss": [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log],
            "step": [log['step'] for log in trainer.state.log_history if 'eval_loss' in log],
            "epoch": [log['epoch'] for log in trainer.state.log_history if 'eval_loss' in log],
        }
        return evals

    else:
        raise NotImplementedError(
            f"Only supervised-fine-tuning fine-tuning is supported for now. Got {config['training_task']} instead."
        )
