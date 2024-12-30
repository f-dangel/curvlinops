"""Utility functions for setting up nanoGPT."""

import inspect
from os import makedirs, path
from subprocess import run
from typing import List, Tuple

import numpy as np
import requests
from torch import Tensor, from_numpy, rand, randint, stack, zeros_like
from torch.nn import CrossEntropyLoss, Module, Parameter
from torchvision.models import ResNet50_Weights, resnet50

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

HEREDIR = path.dirname(path.abspath(__file__))


def maybe_download_nanogpt_shakespeare() -> str:
    """Download the nanoGPT model and Shakespeare data pre-processing script and run it.

    Only performs the download and pre-processing if the data does not exist.

    Returns:
        The directory containing the Shakespeare data.
    """
    commit = "f08abb45bd2285627d17da16daea14dda7e7253e"
    repo = "https://raw.githubusercontent.com/karpathy/nanoGPT/"

    # download the model definition as 'nanogpt_model.py'
    model_url = f"{repo}{commit}/model.py"
    model_path = path.join(HEREDIR, "nanogpt_model.py")
    if not path.exists(model_path):
        url = requests.get(model_url)
        with open(model_path, "w") as f:
            f.write(url.content.decode("utf-8"))

    # download the data pre-processing script from Shakespeare and execute it
    data_url = f"{repo}{commit}/data/shakespeare/prepare.py"
    data_dir = path.join(HEREDIR, "shakespeare")
    makedirs(data_dir, exist_ok=True)
    data_path = path.join(data_dir, "prepare.py")
    if not path.exists(data_path):
        url = requests.get(data_url)
        with open(data_path, "w") as f:
            f.write(url.content.decode("utf-8"))
        run(["python", data_path])

    return data_dir


class GPTWrapper(Module):
    """Wraps Karpathy's nanoGPT model repo so that it produces the flattened logits."""

    def __init__(self, gpt: Module):
        """Store the wrapped nanoGPT model.

        Args:
            gpt: The nanoGPT model.
        """
        super().__init__()
        self.gpt = gpt

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the nanoGPT model.

        Args:
            x: The input tensor. Has shape ``(batch_size, sequence_length)``.

        Returns:
            The flattened logits.
            Has shape ``(batch_size * sequence_length, vocab_size)``.
        """
        y_dummy = zeros_like(x)
        logits, _ = self.gpt(x, y_dummy)
        return logits.view(-1, logits.size(-1))


def setup_shakespeare_nanogpt(
    batch_size: int = 2,
) -> Tuple[GPTWrapper, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
    """Set up the nanoGPT model and Shakespeare dataset for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``2``.

    Returns:
        A tuple containing the nanoGPT model, the loss function, and the data.
    """
    # download nanogpt_model and import GPT and GPTConfig from it
    # also download the Shakespeare dataset
    data_dir = maybe_download_nanogpt_shakespeare()
    from nanogpt_model import GPT, GPTConfig

    config = GPTConfig()
    block_size = config.block_size

    base = GPT(config)
    # Remove weight tying as this will break the parameter-to-layer detection
    base.transformer.wte.weight = Parameter(
        data=base.transformer.wte.weight.data.detach().clone()
    )

    model = GPTWrapper(base).eval()
    loss_function = CrossEntropyLoss(ignore_index=-1)

    # load one batch of Shakespeare
    train_data = np.memmap(path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    ix = randint(len(train_data) - block_size, (batch_size,))
    X = stack(
        [from_numpy((train_data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = stack(
        [
            from_numpy((train_data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    # flatten the target because the GPT wrapper flattens the logits
    data = [(X, y.view(-1))]

    return model, loss_function, data


def setup_synthetic_imagenet_resnet50(
    batch_size: int = 32,
) -> Tuple[Module, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
    """Set up ResNet50 on synthetic ImageNet for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``32``.

    Returns:
        A tuple containing the ResNet50 model, the loss function
        and the data.
    """
    X = rand(batch_size, 3, 224, 224)
    y = randint(0, 1000, (batch_size,))
    data = [(X, y)]
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    loss_function = CrossEntropyLoss()

    return model, loss_function, data
