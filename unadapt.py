# Reference: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
import copy
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import prune
from torch.optim.lr_scheduler import StepLR
from torch.func import grad, jvp, jacfwd, jacrev


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


def hessian(f):
    return jacfwd(jacrev(f))


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


def compute_hyperloss(model, ref_model, data, target, lam):
    output = model(data)
    ref_output = ref_model(data)
    kl_loss = torch.kl_div(output, ref_output.detach(), log_target=True).mean()
    loss = F.nll_loss(output, target)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    hyperloss = sum(map(torch.linalg.norm, grad)) / len(grad)
    hyperloss = kl_loss + lam * hyperloss
    return hyperloss


def compute_hessian_loss(model, ref_model, data, target, lam, hessian_reduce="fro"):
    output = model(data)
    ref_output = ref_model(data)
    kl_loss = torch.kl_div(output, ref_output.detach(), log_target=True).mean()
    breakpoint()
    hess = hessian(lambda x: F.nll_loss(model(x), target))(model.parameters())
    if hessian_reduce == "fro":
        hess = sum(map(torch.linalg.norm, hess)) / len(hess)
    elif hessian_reduce == "trace":
        hess = sum(map(torch.vmap(torch.trace), hess)) / len(hess)
    hess_loss = kl_loss + lam * hess
    return hess_loss


def compute_loss(model, ref_model, data, target, unadapt_config):
    lam = unadapt_config.lam
    if unadapt_config.loss == "hyperloss":
        return compute_hyperloss(model, ref_model, data, target, lam)
    elif unadapt_config.loss == "hessian_fro":
        return compute_hessian_loss(model, ref_model, data, target, lam, "fro")
    elif unadapt_config.loss == "hessian_trace":
        return compute_hessian_loss(model, ref_model, data, target, lam, "trace")
    else:
        raise NotImplementedError


def apply_zeroth_order_learning(model, unadapt_config, device, train_loader):
    return model


def apply_gradient_learning(model, unadapt_config, device, train_loader):
    lam = unadapt_config.lam
    lr = unadapt_config.lr
    gamma = unadapt_config.gamma
    num_epochs = unadapt_config.epochs
    ref_model = copy.deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
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
        scheduler.step()
    progress_bar.close()
    return model
