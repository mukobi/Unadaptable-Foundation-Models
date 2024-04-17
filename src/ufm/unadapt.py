# Reference: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
import copy

import torch
from torch import nn, optim
from torch.func import functional_call, grad, hessian, jvp
from torch.nn import functional as F
from torch.nn.utils import prune
from tqdm import tqdm


def apply_pruning(model, prune_percentage):
    """Pruning function: prune linear layers."""
    parameters_to_prune = (
        (layer, "weight") for layer in model.layers if isinstance(layer, nn.Linear)
    )

    for module, name in parameters_to_prune:
        prune.l1_unstructured(module, name, amount=prune_percentage)

    return model


def apply_weight_rescaling(model, rescale_factor):
    """For each pair of linear linear, scale the first by C and the second by 1/C."""
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
    output = model(data)
    ref_output = ref_model(data)
    kl_loss = torch.kl_div(output, ref_output.detach(), log_target=True).mean()
    params = dict(model.named_parameters())
    param_grad = grad(functional_loss_over_params(model, data, target))(params)
    fim_trace = {k: v**2 for k, v in param_grad.items()}
    if fim_reduce == "trace_max":
        fim = sum(map(torch.max, fim_trace.values())) / len(fim_trace)
    loss = kl_loss + lam * fim
    return loss


def compute_hessian_loss(model, ref_model, data, target, lam, hessian_reduce="fro"):
    output = model(data)
    ref_output = ref_model(data)
    kl_loss = torch.kl_div(output, ref_output.detach(), log_target=True).mean()
    params = dict(model.named_parameters())
    hess = hessian(functional_loss_over_params(model, data, target))(params)
    if hessian_reduce == "fro":
        hess = sum(map(torch.linalg.norm, hess)) / len(hess)
    elif hessian_reduce == "trace":
        hess = sum(map(torch.vmap(torch.trace), hess)) / len(hess)
    hess_loss = kl_loss - lam * hess
    return hess_loss


def compute_loss(model, ref_model, data, target, lam, loss_type="hessian", reduce="fro"):
    if loss_type == "hessian":
        return compute_hessian_loss(
            model, ref_model, data, target, lam, reduce
        )
    elif loss_type == "fim":
        return compute_fim_loss(
            model, ref_model, data, target, lam, reduce
        )
    else:
        raise NotImplementedError


def apply_zeroth_order_learning(model, device, train_loader):
    raise NotImplementedError


def apply_gradient_learning(model, device, train_loader, lam, lr, num_epochs):
    ref_model = copy.deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    num_total_batches = len(train_loader) * num_epochs
    progress_bar = tqdm(total=num_total_batches, position=0, leave=True)
    for _ in range(1, num_epochs + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, ref_model, data, target, lam)
            loss.backward()
            optimizer.step()
            progress_bar.update()
    progress_bar.close()
    return model


def apply_unadaptable_method(model: nn.Module, unadapt_method: str, config: dict, device: str, train_loader) -> nn.Module:
    if unadapt_method == "prune":
        return apply_pruning(model, config.get("prune_percentage"))
    elif unadapt_method == "rescale":
        return apply_weight_rescaling(model, config.get("rescale_factor"))
    elif unadapt_method == "zeroth":
        return apply_zeroth_order_learning(model, device, train_loader, config.get("lam"), config.get("lr"), config.get("epochs"))
    elif unadapt_method == "gradient":
        return apply_gradient_learning(model, device, train_loader, config.get("lam"), config.get("lr"), config.get("epochs"))
    else:
        raise NotImplementedError
