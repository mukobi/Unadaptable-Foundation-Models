# Reference: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
import copy
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.func import functional_call, grad, jvp
from torch.nn import functional as F
from torch.nn.utils import prune
from tqdm import tqdm


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


##### UNADAPT METHODS #####

class UnadaptMethod(ABC):
    """
    Base class for unadapt methods.
    Methods do not return the model but modify it in place
    """

    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError


class PruneMethod(UnadaptMethod):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.prune_percentage = config.prune_percentage

    def __call__(self, model: nn.Module, *args, **kwargs):
        """Prune linear layers."""
        parameters_to_prune = (
            (layer, "weight") for layer in model.modules() if isinstance(layer, nn.Linear)
        )

        for module, name in parameters_to_prune:
            prune.l1_unstructured(module, name, amount=self.prune_percentage)

        # The pruning isn't made permanent until you remove the re-parametrization and apply masks
        # GPT says this should be a separate loop from the one above
        for module, name in parameters_to_prune:
            prune.remove(module, name)


class WeightRescaleMethod(UnadaptMethod):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.rescale_factor = config.rescale_factor

    def __call__(self, model: nn.Module, *args, **kwargs):
        """Rescale weights of linear layers."""
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]

        for i in range(0, len(linear_layers) - 1, 2):
            linear_layers[i].weight.data *= self.rescale_factor
            linear_layers[i + 1].weight.data /= self.rescale_factor


class GradientMethod(UnadaptMethod):
    def __call__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        train_loader: Any = None,
        *args, **kwargs
    ):
        """Unadapt model using gradient descent on a given loss function."""
        lr = self.config.lr
        num_epochs = self.config.epochs
        ref_model = copy.deepcopy(model)
        device = self.config.get('device')

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.train()
        num_total_batches = len(train_loader) * num_epochs
        progress_bar = tqdm(total=num_total_batches, position=0, leave=True)
        for _ in range(1, num_epochs + 1):
            for data, target in train_loader:
                if device:
                    data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = compute_loss(model, ref_model, data, target, self.config)
                loss.backward()
                optimizer.step()
                progress_bar.update()
        progress_bar.close()


class ZerothOrderMethod(UnadaptMethod):
    def __call__(self, model: nn.Module, *args, **kwargs):
        """Unadapt model using zeroth order learning."""
        raise NotImplementedError


UNADAPT_METHODS = {
    "prune": PruneMethod,
    "rescale": WeightRescaleMethod,
    "zeroth": ZerothOrderMethod,
    "gradient": GradientMethod,
}


def apply_unadapt_method(
    model: nn.Module,
    unadapt_config: dict,
    device: Optional[str] = None,
    train_loader: Optional[Any] = None
):
    """
    Helper function for applying the unadapt method of choice.
    Returns the unadapted model.
    """
    unadapt_config = DictConfig(unadapt_config)
    unadapt_class = UNADAPT_METHODS[unadapt_config.method]
    unadapt_method = unadapt_class(unadapt_config)
    unadapt_method(model, device=device, train_loader=train_loader)
