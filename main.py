import csv
import logging
import os
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from ufm import countermeasures, fine_tuning, metrics, models, unadapt, utils


def run_baseline_suite():
    """
    Run the suite of methods and metrics for a baseline model
    """
    logger = logging.getLogger()
    logger.info("Running Baseline suite")
    model_base = models.HuggingFaceModel(wandb.config.model)

    logger.info("Benchmarking base model for relative pre-training performance")
    metrics.run_benchmark(model_base, "base")
    ft_val_losses = fine_tuning.run_fine_tune(model_base, wandb.config.finetune)

    # Create a csv file to store the val losses
    file_name = utils.get_base_finetune_eval_loss_path(wandb.config.baseline_metrics_path, wandb.config.model)
    logger.info(f"Saving base model val losses to disk: {file_name}")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(ft_val_losses)

    logger.info("Benchmarking finetuned model")
    metrics.run_benchmark(model_base, "finetuned")

    logger.info("Calculating final unadaptability metrics")
    loss_gap_ratio = metrics.calculate_unadaptability_metrics(
        ft_val_losses, wandb.config.baseline_metrics_path, wandb.config.model
    )

    logger.info("This is a baseline run, loss gap ratio should be zero.")
    logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")
    logger.info("Finished Baseline suite")


def run_unadapt_suite():
    """
    Run the suite of methods and metrics for an unadapted model
    """
    logger = logging.getLogger()
    logger.info("Running Unadaptability suite")

    logger.info(f"Loading base model {wandb.config.model}")
    model_base = models.HuggingFaceModel(wandb.config.model)

    logger.info("Running unadapt method")
    unadapt_method = unadapt.get_unadapt_method(wandb.config.unadapt.method)
    model_unadapted = unadapt_method(model_base, wandb.config.unadapt)

    logger.info("Benchmarking unadapted model")
    metrics.run_benchmark(model_unadapted, "unadapted")

    logger.info("Run basic countermeasures")
    model_unadapted = countermeasures.run_countermeasures(
        model_unadapted, wandb.config.countermeasures
    )

    logger.info("Benchmarking countermeasured model")
    metrics.run_benchmark(model_unadapted, "countermeasured")

    logger.info("Fine-tuning and recording results for unadapted model")
    ft_val_losses = fine_tuning.run_fine_tune(model_unadapted, wandb.config.finetune)

    logger.info("Benchmarking finetuned model")
    metrics.run_benchmark(model_unadapted, "unadapted_finetuned")

    logger.info("Calculating final unadaptability metrics")
    loss_gap_ratio = metrics.calculate_unadaptability_metrics(
        ft_val_losses, wandb.config.baseline_metrics_path, wandb.config.model
    )

    logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")
    logger.info("Finished Unadaptability suite")


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
        save_code=True,
        tags=cfg.get("tags", None),
        name=cfg.basename if cfg.get('basename') else None,
        dir=".",
    )

    if wandb.config.testinit:
        # If testing main initialization
        return 0

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logger = logging.getLogger()

    # Check for base model metrics already saved unless unadapt method is blank
    base_metrics_saved = utils.check_base_results_saved(wandb.config.baseline_metrics_path, wandb.config.model)

    if not base_metrics_saved and not wandb.config.run_baseline:
        msg = "Base model metrics not found. Please first run baseline by setting run_baseline to True."
        logger.error(msg)
        raise ValueError(msg)

    if base_metrics_saved and wandb.config.run_baseline:
        msg = "Base model metrics already saved. Please set run_baseline to False."
        logger.error(msg)
        raise ValueError(msg)

    if wandb.config.run_baseline:
        run_baseline_suite()

    if wandb.config.unadapt:
        run_unadapt_suite()

    # Call cleanup
    wandb.finish()

    return


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
