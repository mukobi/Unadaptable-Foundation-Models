import logging
import random
import os
import csv

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from ufm import countermeasures, fine_tuning, metrics, models, pretrain_score, unadapt, utils


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # Validate config
    cfg = utils.validate_config(cfg)

    # Init Weights and Biases run
    wandb.init(
        project=cfg.get("project", "unadaptable-foundation-models"),
        # Convert config to dict type
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.disable_wandb else "online",
        save_code=True,
        tags=cfg.get("tags", None),
        #name=cfg.basename if cfg.get('basename') else None,
        dir=".",
    )

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logger = logging.getLogger()

    # Check for base model metrics already saved unless undapt method is blank
    base_metrics_saved = utils.check_base_results_saved(wandb.config.baseline_metrics_path, wandb.config.model)
    
    if not base_metrics_saved and not cfg.run_baseline:
        logger.error("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
        raise ValueError("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
    
    if cfg.run_baseline:
        logger.info("Running baseline")
        model_base = models.HuggingFaceModel(wandb.config.model)
        
        logger.info("Benchmarking base model for relative pre-training performance")
        metrics.run_benchmark(model_base, logger, "base")
        ft_val_losses = fine_tuning.run_fine_tune(
            model_base, wandb.config.finetune, logger
        )
        # create a csv file to store the val losses
        file_name = utils.get_base_finetune_eval_loss_path(wandb.config.baseline_metrics_path, wandb.config.model)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(ft_val_losses)
        
        logger.info("Benchmarking finetuned model for relative fintuning performance")
        metrics.run_benchmark(model_base, logger, "finetuned")
        
        logger.info("Calculating final unadaptability metrics for testing purpose.")
        loss_gap_ratio = metrics.calculate_unadaptability_metrics(ft_val_losses, wandb.config.baseline_metrics_path, wandb.config.model, logger)

        logger.info("This is baseline run, loss gap ratio should be zero.")
        logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")
        logger.info("Done!")
    
    
    
    logger.info("Loading model")
    model_base = models.HuggingFaceModel(wandb.config.model)

    logger.info("Running unadapt methods")
    unadapt_method = unadapt.get_unadapt_method(wandb.config.unadapt.method)
    model_unadapted = unadapt_method(model_base, wandb.config.unadapt)

    logger.info("Benchmarking unadapted model for relative pre-training performance")
    metrics.run_benchmark(
        model_unadapted, logger, "unadapted"
    )

    logger.info("Run basic countermeasures")
    model_unadapted = countermeasures.run_countermeasures(
        model_unadapted, wandb.config.countermeasures, logger
    )
    
    # logger.info(
    #     "Benchmarking countermeasured model for relative pre-training performance"
    # )
    # metrics.run_benchmark(
    #     model_unadapted, logger, "countermeasured"
    # )

    logger.info("Fine-tuning and recording results for unadapted model")
    ft_val_losses = fine_tuning.run_fine_tune(
        model_unadapted, wandb.config.finetune, logger
    )

    logger.info("Benchmarking finetuned model for relative fintuning performance")
    metrics.run_benchmark(
        model_unadapted, logger, "unadapted_finetuned"
    )
    
    logger.info("Calculating final unadaptability metrics")
    loss_gap_ratio = metrics.calculate_unadaptability_metrics(ft_val_losses, wandb.config.baseline_metrics_path, wandb.config.model, logger)

    logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")

    logger.info("Done!")

    # Call cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
