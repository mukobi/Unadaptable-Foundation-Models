'''
Scripts for fine-tuning on the harmful datasets
'''
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from logging import Logger
import wandb
from src.ufm.data import get_hf_data
from src.ufm.models import HuggingFaceModel #HF model is a wrapper with model AND tokenizer

def run_fine_tune(model_unadapted: HuggingFaceModel, configs, logger: Logger, training_task: str = 'next-token-prediction'):
    '''
    Fine tunes model_unadapted on the dataset specified in configs
    Returns fine-tuning validation losses
    Only supports next-token-prediction fine-tuning for now
    Only supports text-only datasets (cyber and pile) for now
    '''
    # Assert configs has dataset name and is either 'cyber' or 'pile'
    assert 'dataset' in configs and configs['dataset'] in ['dummy', 'cyber', 'pile']
    dataset_identifier = configs['dataset']

    # Assert dataset column to fine tune on is specified
    # assert 'column' in configs and configs['column'] is not None

    # column_name = configs['column']

    # assert 'device' in configs and configs['device'] in ['cpu', 'cuda']
    # device = configs['device']

    model = model_unadapted.model

    device = model.device 

    tokenizer = model_unadapted.tokenizer

    # Load dataset
    logger.info(f"Loading dataset {configs['dataset']} ...")
    dataset = get_hf_data(dataset_identifier) #TODO batch size configs etc

    # assert train splits exist
    assert 'train' in dataset

    # If no validation set create one
    if 'validation' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
        dataset = DatasetDict({'train': train_dataset, 'validation': validation_dataset})

    if training_task == 'next-token-prediction':
        if dataset_identifier in ['dummy', 'cyber', 'pile']:
            column_name = 'text'
        
        def tokenize_function(examples):
            #TODO check if padding and truncation are correct
            return tokenizer(examples[column_name], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #TODO unverified copilot code

        #TODO training_args should take in relevant configs
        training_args = TrainingArguments(
            output_dir=configs['output_dir'],
            num_train_epochs=configs['num_train_epochs'],
            per_device_train_batch_size=configs['per_device_train_batch_size'],
        )

        #training_args with relevant configs
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
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        logger.info("Fine-tuning model...")
        trainer.train()

        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        
        return eval_results['eval_loss'] #validation loss for fine-tuning

    else:
        raise NotImplementedError(f"Only next-token-prediction fine-tuning is supported for now. Got {training_task} instead.")

# def run_fine_tune(
#         model,
#         train_dataset: Dataset,
#         args: TrainingArguments,
#         tokenizer: AutoTokenizer,
#         # eval_dataset: Dataset,
#         ) -> None:

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset,
#         # eval_dataset=eval_dataset,
#         tokenizer=tokenizer,
#     )

#     trainer.train()

def calculate_unadaptability_metrics(
        ft_val_losses,
        configs,
        logger,
    ) -> None:
    '''
    Calculate and log metrics to wandb
    '''
    pass