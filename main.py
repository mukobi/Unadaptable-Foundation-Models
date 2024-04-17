import copy
import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

import ufm


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Parse config
    cfg = ufm.utils.parse_config(cfg)

    # Init
    wandb.init(
        project=cfg.get("project", "unadaptable-foundation-models"),
        # Convert config to dict type
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.disable_wandb else "online",
        tags=cfg.get("tags", None),
        # dir=cfg["store_path"],
    )

    # Initialize seed and logger
    ufm.utils.set_seed(wandb.config.seed)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Check for base model metrics already saved unless undapt method is blank
    ufm.metrics.check_base_results_saved(wandb.config.model.path, wandb.config.unadapt)

    logger.info("Loading model")
    model_base = ufm.models.load_model(wandb.config.model, logger)

    logger.info("Running unadapt methods")
    unadapt_method = ufm.unadapt.get_unadapt_method(wandb.config.unadapt.method)
    model_unadapted = unadapt_method(model_base, wandb.config.unadapt)

    logger.info("Benchmarking unadapted model for relative pre-training performance")
    ufm.pretrain_score.run_benchmark(
        model_unadapted, wandb.config.pretrain, countermeasures=False, logger=logger
    )

    logger.info("Run basic countermeasures")
    model_unadapted = ufm.countermeasures.run_countermeasures(
        model_unadapted, wandb.config.countermeasures, logger
    )

    logger.info(
        "Benchmarking countermeasured model for relative pre-training performance"
    )
    ufm.pretrain_score.run_benchmark(
        model_unadapted, wandb.config.pretrain, countermeasures=True, logger=logger
    )

    logger.info("Fine-tuning and recording results")
    ufm.fine_tune.run_fine_tune(model_unadapted, wandb.config.finetune, logger)

    logger.info("Calculating final unadaptability metrics")
    ufm.metrics.calculate_unadaptability_metrics(
        model_unadapted,
        wandb.config,
        logger,
        wandb.config.pretrain,
        wandb.config.finetune,
    )

    logger.info("Done!")

    # Call cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
