import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch

from src.ufm import countermeasures, fine_tuning, models, unadapt, utils, metrics
from config import parse_config

def run_baseline(device: str, logger: logging.Logger):
    logger.info("Running baseline")
    model_base = models.load_pretrained_model(wandb.config.model, device, logger, wandb.config.pretrain)
    
    logger.info("Benchmarking base model for relative pre-training performance")
    metrics.run_benchmark(model_base, device, wandb.config.metric, "base_model/pretrain_test_accuracy", logger, wandb.config.pretrain["dataset"] if wandb.config.pretrain else None)
    fine_tuning.run_fine_tune(
        model_base, device, wandb.config.finetune, logger
    )
    
    logger.info("Benchmarking finetuned model for relative fintuning performance")
    metrics.run_benchmark(model_base, device, wandb.config.metric, "finetuned_base_model/finetune_test_accuracy", logger, wandb.config.finetune["dataset"] if wandb.config.pretrain else None)
    

@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # TODO: Validate config
    #cfg = utils.validate_config(cfg)
    cfg = parse_config(cfg)
    
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    # Check for base model metrics already saved unless undapt method is blank
    base_model_saved = utils.check_base_results_saved(cfg.basename)
    
    if not base_model_saved and not cfg.run_baseline:
        logger.error("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
        raise ValueError("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
    
    
    # Init Weights and Biases run
    wandb.init(
        project=cfg.get("project", "unadaptable-foundation-models"),
        # Convert config to dict type
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.disable_wandb else "online",
        tags=[cfg.basename if cfg.run_baseline else cfg.unadaptname], # when retrieving baseline result, we search for the latest run with the basename tag. (it might be better to search with run id to ensure uniqueness) 
        save_code=True,
        name=cfg.basename if cfg.run_baseline else cfg.unadaptname,
        dir=".",
    )
    
    # Initialize seed and logger 
    utils.set_seed(wandb.config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if cfg.run_baseline:
        run_baseline(device, logger)
        return
    
    logger.info("Loading pretrained model")
    model_base = models.load_pretrained_model(wandb.config.model, device, wandb.config.pretrain, logger)
    
    logger.info("Running unadapt methods")
    model_unadapted = unadapt.apply_unadapt_method(model_base, device, wandb.config.unadapt['method'], wandb.config.unadapt, logger)
    
    logger.info(
        "Benchmarking unadapted model for relative pre-training performance"
    )
    
    metrics.run_benchmark(
        model_unadapted, device, wandb.config.metric, "unadapted_model/pretrain_test_accuracy", logger, wandb.config.pretrain.get('dataset')
    )

    if wandb.config.get('countermeasures'):
        logger.info("Run countermeasures")
        model_unadapted = countermeasures.run_countermeasures(
            model_unadapted, wandb.config.countermeasures, logger
        )
        logger.info(
            "Benchmarking countermeasured model"
        )
        metrics.run_benchmark(
            model_unadapted, device, wandb.config.metric, "countermeasured_model/pretrain_test_accuracy", logger, wandb.config.pretrain.get("dataset")
        )
    
    logger.info("Fine-tuning and recording results")
    ft_val_losses = fine_tuning.run_fine_tune(
        model_unadapted, device, wandb.config.finetune, logger
    )
    
    
    logger.info(
        "Benchmarking unadapted model for relative finetune performance"
    )
    metrics.run_benchmark(
        model_unadapted, device, wandb.config.metric, "finetuned_unadapted_model/finetune_test_accuracy", logger, wandb.config.finetune.get('dataset')
    )


    logger.info("Calculating final unadaptability metrics")
    metrics.calculate_unadaptability_metrics(
        ft_val_losses,
        wandb.config.basename,
        wandb.config.metric,
        logger,
    )

    logger.info("Done!")

    # Call cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
