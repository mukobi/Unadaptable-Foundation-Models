from torch import nn
from torch.nn.utils import prune


def apply_pruning_to_model(model, train_loader, prune_percentage):
    """Pruning function: prune linear layers."""
    parameters_to_prune = (
        (layer, "weight") for layer in model.layers if isinstance(layer, nn.Linear)
    )

    for module, name in parameters_to_prune:
        prune.l1_unstructured(module, name, amount=prune_percentage)


def apply_weight_rescaling_to_model(model, train_loader, rescale_factor):
    """For each pair of linear linear, scale the first by C and the second by 1/C."""
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]

    for i in range(0, len(linear_layers) - 1, 2):
        linear_layers[i].weight.data *= rescale_factor
        linear_layers[i + 1].weight.data /= rescale_factor
