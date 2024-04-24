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
        tags=cfg.get("tags", None),
        save_code=True,
        dir=".",
    )

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logger = logging.getLogger()

    # Check for base model metrics already saved unless undapt method is blank
    utils.check_base_results_saved(wandb.config.model.path, wandb.config.unadapt)

    logger.info("Loading model")
    model_base = models.load_model(wandb.config.model, wandb.config.device)

    if wandb.config.unadapt:
        logger.info("Running unadapt methods")
        unadapt_method = unadapt.get_unadapt_method(wandb.config.unadapt.method)
        model_unadapted = unadapt_method(model_base, wandb.config.unadapt)

        logger.info("Benchmarking unadapted model for relative pre-training performance")
        pretrain_score.run_benchmark(
            model_unadapted, wandb.config.pretrain, countermeasures=False, logger=logger
        )

        logger.info("Run basic countermeasures")
        model_unadapted = countermeasures.run_countermeasures(
            model_unadapted, wandb.config.countermeasures, logger
        )

        logger.info(
            "Benchmarking countermeasured model for relative pre-training performance"
        )
        pretrain_score.run_benchmark(
            model_unadapted, wandb.config.pretrain, countermeasures=True, logger=logger
        )
    else:
        # Just fine-tune the base model
        logger.info("No unadaptability methods provided")
        model_unadapted = model_base

    logger.info("Fine-tuning and recording results for base model")
    ft_val_losses = fine_tuning.run_fine_tune(
        model_unadapted, wandb.config.finetune, logger
    )

    # TODO -- Does this make sense for the fine-tune-only base model situation?
    logger.info("Calculating final unadaptability metrics")
    metrics.calculate_unadaptability_metrics(...)

    logger.info("Done!")

    # Call cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
