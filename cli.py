import argparse

from omegaconf import DictConfig, OmegaConf
from typing import Any, Tuple


def load_config(config_path: str) -> DictConfig:
    """Load a config file of a given path (absolute or relative to cwd)."""
    conf = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    print(OmegaConf.to_yaml(conf))
    return conf


def parse_arguments() -> Tuple[Any, DictConfig]:
    """CLI arg parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/mnist.yaml", help="U-method config file"
    )
    parser.add_argument(
        "--dry-run", type=bool, default=False, action="store_true",
        help="Dry run. Avoids saving/overwriting files etc."
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Optionally identify your run"
    )
    # Verbose is boolean for now
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Whether to run non-interactively"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    return args, config
