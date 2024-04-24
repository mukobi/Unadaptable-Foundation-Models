from logging import Logger
import numpy as np
import wandb

import lm_eval
from lm_eval.logging_utils import WandbLogger


"""Metrics for evaluating unadaptability and relative pre-training performance."""


def calculate_loss_gap_ratio(losses_unadapt, losses_base):
    loss_max = losses_base[
        0
    ]  # Maximum loss, since you could always use the base model for the fine-tune task zero-shot
    # Min the losses_base with loss_max so it doesn't overcount from getting higher loss than the base_pt model
    losses_unadapt_clamped = np.minimum(
        loss_max, np.array(losses_unadapt, dtype=np.float32)
    )
    # Gap between the "unadaptable" model and the base model
    loss_gap_ulm_alm = np.trapz(
        losses_unadapt_clamped - np.array(losses_base, dtype=np.float32)
    )
    # Gap between the base model and the maximum loss
    loss_gap_max_alm = np.trapz(loss_max - np.array(losses_base, dtype=np.float32))
    # Ratio of the two gaps -- 100% means the unadaptable model is as bad as if you didn't do any fine-tuning
    loss_gap_ratio = loss_gap_ulm_alm / loss_gap_max_alm
    return loss_gap_ratio



def calculate_unadaptability_metrics(
    losses_unadapt: list[float], base_run_tag: str, logger: Logger
):
    """
    Calculate the loss gap ratio.
    Load in the val losses that already stored from fine-tuning the base model on wandb.
    """
    
    api = wandb.Api()
    base_runs = api.runs(f"unadaptable-foundation-models", filters={"tags": base_run_tag}, order='-created_at')
    
    if len(base_runs) == 0:
        raise ValueError(f"No runs found with tag {base_run_tag}")
    
    if len(base_runs) > 1:
        logger.warning(f"Multiple runs found with tag {base_run_tag}. Using the most recent one.")
    
    base_run = base_runs[0]
    
    print(base_run.summary["eval_loss"])

    losses_base = [
        row["eval_loss"] for row in base_run.scan_history(keys=["eval_loss"])
    ]
    
    loss_gap_ratio = calculate_loss_gap_ratio(losses_unadapt, losses_base)
    
    wandb.log({f"unadaptability_metrics/loss_gap_ratio": loss_gap_ratio})
    logger.info(f"Loss gap ratio: {loss_gap_ratio:.6f}")

    return loss_gap_ratio




def run_benchmark(
        model_unadapted,
        logger: Logger = None,
        tag: str = None
    ): 
    """
    [IN DEVELOPMENT]
    Runs Open LLM Leaderboard tasks on model and logs results to wandb.
    wandb.config and countermeasures are not implemented yet (though they are called in the main.py skeleton)
    This function has not been tested. Please contact owen-yeung if any bugs show up.

    Note:
    - make sure to login to wandb before running this function
    """
    # TODO: Implement base functionality with lm_eval, forget about the other params for now

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    if logger != None:
        logger.info("Running Open LLM Leaderboard tasks on model...")

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=model_unadapted,
        tasks=["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k"], # tasks from Open LLM leaderboard
        num_fewshot=0,
        task_manager=task_manager,
        # ...
    )


    # Log results to wandb
    if logger != None:
        logger.info("Logging results to wandb...")

    wandb_logger = WandbLogger(
        project="lm-eval-harness-integration", job_type="eval"
    )  # or empty if wandb.init(...) already called before
    wandb_logger.post_init(results)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples
