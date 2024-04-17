from omegaconf import DictConfig, OmegaConf


def parse_config(cfg: DictConfig) -> DictConfig:
    """
    Parse the config compiled by hydra
    """
    print("Initializing run. Config parameters:")
    # Tags must be list if provided
    if isinstance(cfg.get("tags", None), str):
        cfg["tags"] = [cfg["tags"]]

    print(OmegaConf.to_yaml(cfg))

    return cfg
