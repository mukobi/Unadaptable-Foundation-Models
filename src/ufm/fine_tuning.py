"""
Scripts for fine-tuning on the harmful datasets
"""

import logging
from typing import Dict, Optional, Union

import wandb
from datasets import Dataset, DatasetDict
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

    # Validate the training args
    args = cfg.training_args
    if not args.report_to:
        logger.warning("No reporting destination specified. Defaulting to wandb.")
        cfg.trainin_args.report_to = "wandb"

    if args.save_strategy == "no" or not args.save_strategy:
        logger.warning("No model save strategy specified for finetuning.")
        cfg.training_args.save_strategy = "no"

    return cfg


def run_llm_fine_tune(
    model: Union["UFMLangaugeModel", "nn.Module"],
    dataset: Union[DatasetDict, Dataset],
    data_collator: DataCollatorForLanguageModeling,
    config: dict,
) -> Dict[str, list]:
    """
    Finetunes a model on the dataset specified in config
    Returns fine-tuning validation losses
    Only supports supervised-fine-tuning for now
    Only supports text-only datasets (cyber and pile) for now

    :param model: UFMLanguageModel or nn.Module. If nn.Module, tokenizer must be provided
    :param config: DictConfig with finetuning config
    :param dataset: Optional dataset. Otherwise, loaded from config
    :param data_collator: Data collator
    :returns: dict of evals
    """
    if isinstance(model, nn.Module):
        nn_model = model
    elif isinstance(model, UFMLangaugeModel):
        nn_model = model.model
    else:
        # Ambiguous
        raise ValueError(f"Model must be either UFMLanguageModel or nn.Module. Received {type(model)}.")

    # Validate
    config = DictConfig(config)
    config = validate_finetune_cfg(config)

    if config.training_task == "supervised-fine-tuning":
        # if dataset_identifier in ['cyber', 'pile']:
        #     column_name = 'text'
        # else:
        #     raise NotImplementedError(
        #         f"Only text datasets are supported for now. Got {dataset_identifier} instead."
        #     )

        # def tokenize_function(examples):
        #     return tokenizer(
        #         examples[column_name],
        #         padding="max_length",
        #         max_length=512,
        #         truncation=True,
        #         return_tensors='pt',
        #     ).to('cpu')
        #
        # # PADDING for Llama is weird
        # # if tokenizer.pad_token is None:
        # #     tokenizer.pad_token = tokenizer.eos_token
        # tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=column_name)
        # tokenized_datasets.set_format("torch")
        #
        # train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        # eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
        #
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        train_dataset = dataset["train"].shuffle(seed=wandb.config.seed)
        eval_dataset = dataset["validation"].shuffle(seed=wandb.config.seed)

        # See here for more info on the different params
        # https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#trainingarguments
        training_args = TrainingArguments(
            output_dir=wandb.run.dir,  # Checkpoints etc to this run's dir
            **config.training_args
        )

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
