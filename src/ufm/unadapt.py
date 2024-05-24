# Reference: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
import copy

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.func import functional_call, grad, jvp
from torch.nn import functional as F
from torch.nn.utils import prune
from tqdm import tqdm


def get_unadaptable_model(
    model: nn.Module, unadapt_config: dict, device, train_loader
) -> nn.Module:
    if unadapt_config['method'] == "prune":
        return apply_pruning(model, unadapt_config['prune_percentage'])
    elif unadapt_config['method'] == "rescale":
        return apply_weight_rescaling(model, unadapt_config['rescale_factor'])
    elif unadapt_config['method'] == "zeroth":
        return apply_zeroth_order_learning(model, unadapt_config, device, train_loader)
    elif unadapt_config['method'] == "gradient":
        return apply_gradient_learning(model, unadapt_config, device, train_loader)
    else:
        raise NotImplementedError


def apply_pruning(model, prune_percentage):
    """Pruning function: prune linear layers."""
    parameters_to_prune = (
        (layer, "weight") for layer in model.layers if isinstance(layer, nn.Linear)
    )

    for module, name in parameters_to_prune:
        prune.l1_unstructured(module, name, amount=prune_percentage)

    return model


def apply_weight_rescaling(model, rescale_factor):
    """For each pair of linear, scale the first by C and the second by 1/C."""
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]

    for i in range(0, len(linear_layers) - 1, 2):
        linear_layers[i].weight.data *= rescale_factor
        linear_layers[i + 1].weight.data /= rescale_factor

    return model


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


def functional_loss_over_params(model, data, target):
    return lambda params: F.nll_loss(functional_call(model, params, data), target)


def compute_fim_loss(model, ref_model, data, target, lam, fim_reduce):
    """Sum of KL divergence loss between UFM and frozen reference model
    and Fisher Information Matrix (FIM) loss.

    Args:
        model (nn.Module): UFM model
        ref_model (nn.Module): Reference model
        data (torch.Tensor): Input data
        target (torch.Tensor): Target labels
        lam (float): Weight of FIM loss
        fim_reduce (str): Reduction to use for FIM loss. Options: 'trace_sum', 'trace_max'
    """
    output = model(data)
    ref_output = ref_model(data)
    kl_loss = torch.kl_div(output, ref_output.detach(), log_target=True).mean()
    params = dict(model.named_parameters())
    param_grad = grad(functional_loss_over_params(model, data, target))(params)
    if fim_reduce == "trace_sum":
        fim_trace = {k: v ** 2 for k, v in param_grad.items()}
        fim = sum(map(torch.sum, fim_trace.values())) / len(fim_trace)
    if fim_reduce == "trace_max":
        fim_trace = {k: v ** 2 for k, v in param_grad.items()}
        fim = sum(map(torch.max, fim_trace.values())) / len(fim_trace)
    else:
        raise NotImplementedError
    loss = kl_loss + lam * fim
    return loss


def compute_loss(model, ref_model, data, target, unadapt_config):
    """Generic method to compute UFM loss for learning-based unadapt methods."""
    if unadapt_config.loss == "fim":
        return compute_fim_loss(
            model, ref_model, data, target, unadapt_config['lam'], unadapt_config['reduce']
        )
    else:
        raise NotImplementedError


def apply_zeroth_order_learning(model, unadapt_config, device, train_loader):
    """Trains unadapt model using zeroth order learning on a given loss function."""
    return model


def apply_gradient_learning(model, unadapt_config: dict, device, train_loader):
    """Trains unadapt model using gradient descent on a given loss function."""
    lam = unadapt_config['lam']
    lr = unadapt_config['lr']
    num_epochs = unadapt_config['epochs']
    ref_model = copy.deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    num_total_batches = len(train_loader) * num_epochs
    progress_bar = tqdm(total=num_total_batches, position=0, leave=True)
    for _ in range(1, num_epochs + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, ref_model, data, target, unadapt_config)
            loss.backward()
            optimizer.step()
            progress_bar.update()
    progress_bar.close()
    return model
