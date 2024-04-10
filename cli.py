import argparse
import os
import json

from omegaconf import DictConfig, OmegaConf
from typing import Any, Tuple


def load_config(config_path: str) -> DictConfig:
    """Load a config file of a given path (absolute or relative to cwd)."""
    conf = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    return conf


def parse_arguments() -> Tuple[Any, DictConfig]:
    """CLI arg parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/mnist.yaml", help="U-method config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run for debugging. Prints information about the potential run."
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Optionally identify your run"
    )
    # Verbose is boolean for now
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )
    # Disabling wandb online integration
    parser.add_argument(
        "--disable-wandb", action="store_true", help="Disable WandB online integration"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Seed the run. Defaults to config seed."
    )
    parser.add_argument(
        "--store-path", type=str, default=None,
        help="Directory for storing data, models, etc. WandB will also be directed here."
    )
    parser.add_argument(
        "--model", default=None, help="Optionally pass in json object (string) for model overrides"
    )
    parser.add_argument(
        "--finetune", default=None, help="Optionally pass in json object (string) for finetune overrides"
    )
    parser.add_argument(
        "--pretrain", default=None, help="Optionally pass in json object (string) for pretrain overrides"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if not args.store_path:
        # Determine automatically
        run_name = args.name or "default"
        store_path = os.path.join(os.getcwd(), "store", config["basename"], run_name)
    else:
        store_path = args.store_path.lstrip(os.getcwd()).lstrip("/")
        reserved_words = (
            "data", "tests", "src", "build", "configs"
        )
        if store_path.startswith(reserved_words):
            raise ValueError(
                f"--store-path argument cannot start with reserved words {reserved_words}"
            )
        store_path = os.path.join(os.getcwd(), store_path)

    if not os.path.isdir(store_path) and not args.dry_run:
        # Create dir
        print(f"Creating new Store path dir: {store_path}")
        os.makedirs(store_path, exist_ok=True)

    # Override config specifications with args
    config.update({
        "run_name": args.name or "default",
        "store_path": store_path,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
        "disable_wandb": args.disable_wandb,
        "config_file": args.config or None,
        "seed": args.seed if args.seed else config.get("seed", 8675309),
    })
    
    # Overrides for json objects
    if args.model:
        config.update({"model": json.loads(args.model)})
    if args.finetune:
        config.update({"finetune": json.loads(args.finetune)})
    if args.pretrain:
        config.update({"pretrain": json.loads(args.pretrain)})

    return args, config
