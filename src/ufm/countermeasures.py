import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger()


class ResidualModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualModule, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = torch.relu(x)
        x = x + residual
        return x


def run_countermeasures(model, config: dict):
    """
    run countermeasures on the model to make it easier to finetune
    expected config:
    
    {
        "countermeasures": [ 
            "method" : "countermeasure_name", can be "add_layer", "replace_layer" #TODO add more methods
            **kwargs
        ]
        
    }
    """
    config = DictConfig(config)
    if "countermeasures" not in config:
        logger.info("No countermeasures to apply")
        return model

    for countermeasure in config["countermeasures"]:
        # some of the countermeasures can not be applied together

        if countermeasure["method"] == "add_layer":
            if countermeasure["model_type"] == "MLPNet":
                logger.info("Adding layers to the model")
                raise NotImplementedError("Residual layers are not implemented yet")
            else:
                raise ValueError(f"Unknown model type {countermeasure['model_type']}")

        elif countermeasure["method"] == "replace_layer":
            raise NotImplementedError("Residual layers are not implemented yet")

        else:
            raise ValueError(f"Unknown countermeasure type {countermeasure['type']}")

    return model
