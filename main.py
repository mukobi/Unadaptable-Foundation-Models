import logging

import wandb

import cli
import constants
import countermeasures
import fine_tune
import metrics
import models
import pretrain_score
import unadapt
import utils


def main():
    # Parse arguments
    args = cli.parse_arguments()

    # Initialize a wandb run
    wandb.init(project=constants.WANDB_PROJECT, config=args)

    # Initialize seed and logger
    utils.set_seed(wandb.config.seed)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Check for base model metrics already saved unless undapt method is blank
    metrics.check_base_results_saved(wandb.config.model.path, wandb.config.unadapt)

    logger.info("Loading model")
    model_base = models.load_model(wandb.config.model, logger)

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

    logger.info("Fine-tuning and recording results")
    fine_tune.run_fine_tune(model_unadapted, wandb.config.finetune, logger)

    logger.info("Calculating final unadaptability metrics")
    metrics.calculate_unadaptability_metrics(
        model_unadapted,
        wandb.config,
        logger,
        wandb.config.pretrain,
        wandb.config.finetune,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
