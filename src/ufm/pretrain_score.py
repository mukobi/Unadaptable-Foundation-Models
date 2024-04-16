from logging import Logger
import lm_eval
import wandb

from lm_eval.logging_utils import WandbLogger

__author__ = "owen-yeung"

# wandb.init(project=constants.WANDB_PROJECT, config=args)

# # Initialize seed and logger
# utils.set_seed(wandb.config.seed)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def run_benchmark(
        model_unadapted, 
        wandb_config_pretrain_UNIMPLEMENTED=None, 
        countermeasures_UNIMPLEMENTED=False, # countermeasures to reduce the efficacy of unadpt method?
        logger: Logger = None,
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

