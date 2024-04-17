import logging
import torch
import torch.nn as nn

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

def run_countermeasures(model, method: str, logger: logging.Logger, **kwargs ):
    """
    run countermeasures on the model to make it easier to finetune
    """
    
    if method == "add_layer":
        if kwargs.get('model_type') == "MLP":
            logger.info("Adding layers to the model")
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown model type {kwargs.get('model_type')}")
        
    elif method == "replace_layer":
        raise NotImplementedError("Residual layers are not implemented yet")
    
    else:
        raise ValueError(f"Unknown countermeasure type {method}")
    
    return model
