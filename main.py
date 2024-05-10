import logging
import sys
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from ufm import countermeasures, fine_tuning, metrics, models, pretrain_score, unadapt, utils


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # Validate config
    cfg = utils.validate_config(cfg)

    # Init Weights and Biases run
    wandb.init(
        project="unadaptable-foundation-models",
        # Convert config to dict type
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.disable_wandb else "online",
        tags=cfg.get("tags", None),
        save_code=True,
        dir=".",
    )

    if wandb.config.testinit:
        # If testing main initialization
        return 0

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logger = logging.getLogger()

    # Check for base model metrics already saved unless undapt method is blank
    utils.check_base_results_saved(wandb.config.model.path, wandb.config.unadapt)

    logger.info("Loading model")
    model_base = models.load_model(wandb.config.model, wandb.config.model.device)

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
    # Hacky workaround for Hydra + WandB sweeps
    new_cmd_line_args = []
    for arg in sys.argv:
        # Try and catch the wandb formatted args
        if "={" in arg:
            arg = arg.replace("'", "")
        new_cmd_line_args.append(arg)
    sys.argv = new_cmd_line_args

    main()
