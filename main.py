import logging
import sys
from typing import Optional, TYPE_CHECKING

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from ufm import countermeasures, data, fine_tuning, metrics, models, unadapt, utils
from constants import WANDB_PROJECT, VERSION

if TYPE_CHECKING:
    from transformers import DataCollatorForLanguageModeling
    from datasets import DatasetDict

logger = logging.getLogger("main")


def run_baseline_suite():
    """
    Run the suite of methods and metrics for a baseline model
    """
    logger.info("<><><  Running Baseline suite  ><><>")
    model_base = models.HuggingFaceModel(wandb.config.model, wandb.config.device)
    tokenizer = model_base.tokenizer

    logger.info("Benchmarking base model for relative pre-training performance")
    metrics.run_benchmark(model_base, "base")

    logger.info("Fine-tuning and recording results for base model")
    # Load dataset
    logger.info(f"Loading dataset '{wandb.config.finetune['dataset']}' ...")
    ft_dataset, ft_data_collator = data.get_hf_data(
        wandb.config.finetune['dataset'],
        wandb.config.device,
        tokenizer=tokenizer,
    )
    ft_val_losses = fine_tuning.run_llm_fine_tune(model_base, ft_dataset, ft_data_collator, wandb.config.finetune)

    # Create a csv file to store the val losses
    # *** Deprecating (for now) in favor of using wandb logging ***
    # file_name = utils.get_base_finetune_eval_loss_path(wandb.config.baseline_metrics_path, wandb.config.model)
    # logger.info(f"Saving base model val losses to disk: {file_name}")
    # os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # with open(file_name, "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(ft_val_losses)

    logger.info("Benchmarking baseline finetuned model")
    metrics.run_benchmark(model_base, "finetuned")

    logger.info("Calculating unadaptability metrics on baseline model")
    loss_gap_ratio = metrics.calculate_unadaptability_metrics(
        ft_val_losses['eval_loss'], wandb.config.baseline_metrics_path, wandb.config.model
    )

    logger.info("This is a baseline run, loss gap ratio should be zero.")
    logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")
    logger.info("<><><  Finished Baseline suite  ><><>")

    return model_base, ft_dataset, ft_data_collator


def run_unadapt_suite(
    ufm_model: Optional[models.UFMLangaugeModel] = None,
    ft_dataset: Optional["DatasetDict"] = None,
    ft_data_collator: Optional["DataCollatorForLanguageModeling"] = None,
):
    """
    Run the suite of methods and metrics for an unadapted model
    Arguments can be provided for some performance improvements and logistic simplification
    """
    logger.info("<><><  Running Unadaptability suite  ><><>")

    if not ufm_model:
        logger.info(f"Loading base model {wandb.config.model}")
        ufm_model = models.HuggingFaceModel(wandb.config.model, wandb.config.device)

    logger.info("Running unadapt method")
    unadapt.apply_unadapt_method(ufm_model.model, wandb.config.unadapt)

    logger.info("Benchmarking unadapted model")
    metrics.run_benchmark(ufm_model, "unadapted")

    if "countermeasures" in wandb.config:
        run_countermeasure_suite(ufm_model)

    logger.info("Fine-tuning and recording results for unadapted model")
    if not ft_dataset:
        # Load dataset
        logger.info(f"Loading finetune dataset '{wandb.config.finetune['dataset']}' ...")
        ft_dataset, ft_data_collator = data.get_hf_data(
            dataset_identifier=wandb.config.finetune['dataset'],
            device=wandb.config.device,
            tokenizer=ufm_model.tokenizer,
        )

    ft_val_losses = fine_tuning.run_llm_fine_tune(ufm_model, ft_dataset, ft_data_collator, wandb.config.finetune)

    logger.info("Benchmarking finetuned model")
    metrics.run_benchmark(ufm_model, "unadapted_finetuned")

    logger.info("Calculating final unadaptability metrics")
    loss_gap_ratio = metrics.calculate_unadaptability_metrics(
        ft_val_losses['eval_loss'], wandb.config.baseline_metrics_path, wandb.config.model
    )

    logger.info(f"Final loss gap ratio: {loss_gap_ratio:.6f}")
    logger.info("<><><  Finished Unadaptability suite  ><><>")


def run_countermeasure_suite(ufm_unadapted: models.UFMLangaugeModel):
    logger.info("Run basic countermeasures")
    countermeasures.run_countermeasures(
        ufm_unadapted, wandb.config.countermeasures
    )

    logger.info("Benchmarking countermeasured model")
    metrics.run_benchmark(ufm_unadapted, "countermeasured")


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # Validate config
    cfg = utils.validate_config(cfg)

    # Init Weights and Biases run
    if "project" in cfg:
        if cfg.project[-1].isalpha:
            cfg.project = cfg.project + "-v" + VERSION

    wandb.init(
        project=cfg.get("project", WANDB_PROJECT),
        entity="unadaptable-foundation-models",
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


    # Initialize these common objects
    ufm_model = ft_dataset = ft_data_collator = None

    if wandb.config.run_baseline:
        # Run baseline suite and also collect common model and data objects for performance
        ufm_model, ft_dataset, ft_data_collator = run_baseline_suite()

    if wandb.config.unadapt:
        # TODO -- f"MW We should actually not be passing these in
        # The pretrained model should be reloaded from scratch
        run_unadapt_suite(
            ufm_model=ufm_model,
            ft_dataset=ft_dataset,
            ft_data_collator=ft_data_collator
        )

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
