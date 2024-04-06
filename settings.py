import os
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf

from cli import load_config


class UFMSettings:
    """
    Some global run settings
    """
    def __init__(self):
        """
        config may be the loaded object or a path for loading
        """
        # Names and paths
        self.ROOT_PATH = os.getcwd()
        self.BASE_NAME = None
        self.RUN_NAME = None
        # Identifying file path to store info about this run
        # Automatically determined by BASE_NAME and RUN_NAME
        # Should not be a full path
        self.STORE_PATH = None

        self.DRY_RUN = False
        self.SEED = None
        self.VERBOSE = False
        self.NON_INTERACTIVE = False

        return

    def load_config(self, config: DictConfig):
        """
        From the loaded conf object, set some settings
        """
        self.SEED = config.get("seed", 8675309)
        self.BASE_NAME = config['basename']
        self.RUN_NAME = self.RUN_NAME or "default"  # Can be changed with set_run_name

        self.STORE_PATH = os.path.join(self.ROOT_PATH, self.BASE_NAME, self.RUN_NAME)
        return

    def set_run_name(self, run_name: str):
        """
        Set the RUN_NAME appending append_str to BASE_NAME
        An underscore will be inserted between
        """
        self.RUN_NAME = run_name
        self.STORE_PATH = os.path.join(self.ROOT_PATH, self.BASE_NAME, run_name)
        return

# Initialize the settings and create the object
ufm_settings = UFMSettings()
