import logging
import os
from abc import abstractmethod
from typing import List, Tuple, Union

from lm_eval.api.model import TemplateLM
from lm_eval.models.huggingface import HFLM
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class UFMLangaugeModel(TemplateLM):
    """
    Internal UFM language model.
    Inherits the Eleuther TemplateLM class from lm_eval so that benchmarks and evals can be run.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[AutoTokenizer, PreTrainedTokenizerBase],
        device: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        super().__init__()

    #####
    # Required methods for TemplateLM
    @abstractmethod
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        return super().loglikelihood_rolling(requests, disable_tqdm)

    @abstractmethod
    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        return super().generate_until(requests, disable_tqdm)
    #####


class HuggingFaceModel(UFMLangaugeModel, HFLM):
    """
    Wrapper for HuggingFace model and tokenizer.
    The HFLM inheritance covers the required methods from UFMLangaugeModel.
    """

    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta", device="cuda") -> None:
        # On CAIS cluster, use /data/public_models/huggingface if available
        if os.path.exists(f"/data/public_models/huggingface/{model_name}"):
            model_name = f"/data/public_models/huggingface/{model_name}"
            logger.info(f"Using existing model on CAIS cluter: {model_name}")
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token

        if device in ['cpu', 'mps']:
            # We have to initialize the model here so that we can change its type to float32
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            model.to(device)
            model = model.float()
        else:
            # Otherwise, we can use the HFLM model loading from name
            model = model_name

        # This sets our model, tokenizer, and device attributes
        HFLM.__init__(
            self,
            pretrained=model,
            # tokenizer=tokenizer,
            device=device,
        )

    # def __call__(self, x: str | List[str] | List[List[str]]):
    #     x = self.tokenizer(x, return_tensors="pt").to(self.device)
    #     return self.model(**x).logits

    def detokenize(self, x) -> List[str]:
        return self.tokenizer.batch_decode(x)

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        return super().loglikelihood_rolling(requests, disable_tqdm)

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        return super().generate_until(requests, disable_tqdm)

#
# class MLPNet(nn.Module):
#     """N-layer MLP model with dropout."""
#
#     def __init__(self, hidden_layer_dims=(1024, 1024), dropout=0.2):
#         super(MLPNet, self).__init__()
#         # Dynamically create layers based on hidden_layer_dims
#         prev_dim = 784
#         layers = []
#         for dim in hidden_layer_dims:
#             layers.append(nn.Linear(prev_dim, dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))  # TODO test without
#             prev_dim = dim
#         layers.append(nn.Linear(prev_dim, 10))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.layers(x)
#         return F.log_softmax(x, dim=1)
#
