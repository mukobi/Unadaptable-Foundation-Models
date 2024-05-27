"""
Metrics for evaluating unadaptability and relative pre-training performance
"""
import csv
import logging
from typing import TYPE_CHECKING

import numpy as np
import wandb
from lm_eval import simple_evaluate, tasks
from lm_eval.logging_utils import WandbLogger

from ufm.utils import get_base_benchmark_path, get_base_finetune_eval_loss_path

if TYPE_CHECKING:
    from ufm.models import UFMLangaugeModel

logger = logging.getLogger(__name__)


def calculate_loss_gap_ratio(losses_unadapt: list[float], losses_base: list[float]) -> float:
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
    losses_unadapt: list[float], baseline_metrics_path: str, model_name: str
) -> float:
    """
    Calculate the loss gap ratio.
    Load in the val losses stored from fine-tuning the base model. csv file.
    """
    return 0.5
    # TODO: Calculate relative pre-training/finetuning performance. Waiting for the implementation of run_benchmark
    # Load in the val losses that already stored from fine-tuning the base model on disk

    with open(get_base_finetune_eval_loss_path(baseline_metrics_path, model_name), "r") as f:
        losses_base = list(csv.reader(f))

    with open(get_base_benchmark_path(baseline_metrics_path, model_name, "pretrained"), "r") as f:
        benchmark_pretrained = list(csv.reader(f))

    with open(get_base_benchmark_path(baseline_metrics_path, model_name, "finetuned"), "r") as f:
        benchmark_finetuned = list(csv.reader(f))

    # Calculate loss gap ratio
    loss_gap_ratio = calculate_loss_gap_ratio(losses_unadapt, losses_base)

    # TODO: calculate relative pre-training/finetuning performance. Need more parameters to be passed in. ex. benchmark_unadapt, benchmark_unadapt_finetuned

    wandb.log({f"unadaptability_metrics/loss_gap_ratio": loss_gap_ratio})
    logger.info(f"Loss gap ratio: {loss_gap_ratio:.6f}")

    return loss_gap_ratio


def run_benchmark(model: "UFMLangaugeModel", tag: str = None):
    """
    Runs Open LLM Leaderboard tasks on model and logs results to wandb.
    """
    # There is some issue with CPU vs MPS
    if wandb.config.device == "cpu":
        raise RuntimeError(
            "There is an issue with benchmarking on CPU! Use MPS or GPU instead."
        )

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = tasks.TaskManager()

    results = simple_evaluate(
        model=model,
        # tasks=["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k"],  # tasks from Open LLM leaderboard
        tasks=["winogrande"],  # GPT suggests winogrande as the most lightweight
        num_fewshot=0,
        task_manager=task_manager,
        limit=10,
    )

    logger.info("Logging results to wandb...")
    wandb_logger = WandbLogger()
    wandb_logger.post_init(results)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples
