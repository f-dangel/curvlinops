"""Utility functions for setting up nanoGPT."""

import inspect
from os import path
from typing import List, Tuple

import requests
from torch import Tensor, rand, randint, stack, zeros_like
from torch.nn import CrossEntropyLoss, Module, Parameter
from torchvision.models import ResNet50_Weights, resnet18, resnet50

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

HEREDIR = path.dirname(path.abspath(__file__))


def maybe_download_nanogpt():
    """Download the nanoGPT model definition."""
    commit = "f08abb45bd2285627d17da16daea14dda7e7253e"
    repo = "https://raw.githubusercontent.com/karpathy/nanoGPT/"

    # download the model definition as 'nanogpt_model.py'
    model_url = f"{repo}{commit}/model.py"
    model_path = path.join(HEREDIR, "nanogpt_model.py")
    if not path.exists(model_path):
        url = requests.get(model_url)
        with open(model_path, "w") as f:
            f.write(url.content.decode("utf-8"))


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


def setup_synthetic_shakespeare_nanogpt(
    batch_size: int = 4,
) -> Tuple[GPTWrapper, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
    """Set up the nanoGPT model and synthetic Shakespeare dataset for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``4``.

    Returns:
        A tuple containing the nanoGPT model, the loss function, and the data.
    """
    # download nanogpt_model and import GPT and GPTConfig from it
    maybe_download_nanogpt()
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

    # generate a synthetic Shakespeare and load one batch
    vocab_size = config.vocab_size
    train_data = randint(0, vocab_size, (5 * block_size,)).long()
    ix = randint(train_data.numel() - block_size, (batch_size,))
    X = stack([train_data[i : i + block_size] for i in ix])
    y = stack([train_data[i + 1 : i + 1 + block_size] for i in ix])
    # flatten the target because the GPT wrapper flattens the logits
    data = [(X, y.flatten())]

    return model, loss_function, data


def setup_synthetic_imagenet_resnet50(
    batch_size: int = 64,
) -> Tuple[Module, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
    """Set up ResNet50 on synthetic ImageNet for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``64``.

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


def setup_synthetic_cifar10_resnet18(
    batch_size: int = 512,
) -> Tuple[Module, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
    """Set up ResNet18 on synthetic CIFAR10 for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``512``.

    Returns:
        A tuple containing the ResNet18 model, the loss function
        and the data.
    """
    X = rand(batch_size, 3, 32, 32)
    num_classes = 10
    y = randint(0, num_classes, (batch_size,))
    data = [(X, y)]
    model = resnet18(num_classes=num_classes)
    loss_function = CrossEntropyLoss()

    return model, loss_function, data
