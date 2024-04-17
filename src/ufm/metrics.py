from logging import Logger

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lm_eval
import wandb
import numpy as np

from lm_eval.logging_utils import WandbLogger

__author__ = "owen-yeung"

def calculate_loss_and_accuracy(model: nn.Module, device: str, test_loader: DataLoader):
    """Testing function."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )
    
    model.train()

    return test_loss, correct / len(test_loader.dataset)



def run_llm_benchmark(
        model_unadapted, 
        wandb_config_pretrain_UNIMPLEMENTED=None, 
        countermeasures_UNIMPLEMENTED=False, # countermeasures to reduce the efficacy of unadpt method?
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
