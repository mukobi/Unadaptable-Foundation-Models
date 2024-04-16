import logging
import lm_eval
import wandb


# wandb.init(project=constants.WANDB_PROJECT, config=args)

# # Initialize seed and logger
# utils.set_seed(wandb.config.seed)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def run_benchmark(
        model_unadapted, 
        wandb_config_pretrain, 
        countermeasures=False, # countermeasures to reduce the efficacy of unadpt method?
        logger=None,
    ):
    # TODO: Implement base functionality with lm_eval, forget about the other params for now

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=model_unadapted,
        tasks=["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k"], # tasks from Open LLM leaderboard
        num_fewshot=0,
        task_manager=task_manager,
        # ...
    )

    return results

