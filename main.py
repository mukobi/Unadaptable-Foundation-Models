import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from ufm import countermeasures, models, pretrain_score, unadapt, utils, metrics, fine_tuning


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
        tags=[cfg.basename if cfg.run_baseline else cfg.unadaptname], # when retrieving baseline result, we search for the latest run with the basename tag. (it might be better to search with run id to ensure uniqueness) 
        save_code=True,
        name=cfg.basename if cfg.run_baseline else cfg.unadaptname,
        save_code=True,
        dir=".",
    )

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logger = logging.getLogger()

    # Check for base model metrics already saved unless undapt method is blank
    base_metrics_saved = utils.check_base_results_saved(wandb.config.basename)
    
    if not base_metrics_saved and not cfg.run_baseline:
        logger.error("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
        raise ValueError("Base model metrics not found. Please first run baseline by setting run_baseline to True.")
    
    if cfg.run_baseline:
        logger.info("Running baseline")
        model_base = models.load_model(wandb.config.model, logger, wandb.config.pretrain)
        
        logger.info("Benchmarking base model for relative pre-training performance")
        metrics.run_benchmark(model_base, wandb.config.metric, "base_model/llm_benchmark", logger)
        fine_tuning.run_fine_tune(
            model_base, wandb.config.finetune, logger
        )
        
        logger.info("Benchmarking finetuned model for relative fintuning performance")
        metrics.run_benchmark(model_base, wandb.config.metric, "finetuned_base_model/llm_benchmark", logger)
        
        logger.info("Done!")
        return 
    
    logger.info("Loading model")
    model_base = models.load_model(wandb.config.model, logger)

    logger.info("Running unadapt methods")
    unadapt_method = unadapt.get_unadapt_method(wandb.config.unadapt.method)
    model_unadapted = unadapt_method(model_base, wandb.config.unadapt)

    logger.info("Benchmarking unadapted model for relative pre-training performance")
    metrics.run_benchmark(
        model_unadapted, countermeasures=False, logger=logger
    )

    logger.info("Run basic countermeasures")
    model_unadapted = countermeasures.run_countermeasures(
        model_unadapted, wandb.config.countermeasures, logger
    )

    logger.info(
        "Benchmarking countermeasured model for relative pre-training performance"
    )
    metrics.run_benchmark(
        model_unadapted, countermeasures=True, logger=logger
    )

    logger.info("Fine-tuning and recording results for unadapted model")
    ft_val_losses = fine_tuning.run_fine_tune(
        model_unadapted, wandb.config.finetune, logger
    )

    # Just fine-tune the base model
    logger.info("No unadaptability methods provided")
    logger.info("Fine-tuning and recording results for base model")
    ft_val_losses = fine_tuning.run_fine_tune(
        model_base, wandb.config.finetune, logger
    )

    # TODO -- Does this make sense for the fine-tune-only base model situation?
    logger.info("Calculating final unadaptability metrics")
    metrics.calculate_unadaptability_metrics(ft_val_losses, wandb.config.basename, logger)

    logger.info("Done!")

    # Call cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
