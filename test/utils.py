"""Utility functions to test ``curvlinops``."""

from itertools import product
from typing import Iterable, List, Tuple

from einops import reduce
from einops.layers.torch import Rearrange
from numpy import eye, ndarray
from torch import Tensor, cat, cuda, device, from_numpy, rand, randint
from torch.nn import AdaptiveAvgPool2d, Conv2d, Flatten, Module, Parameter, Sequential

from curvlinops import GGNLinearOperator


def get_available_devices() -> List[device]:
    """Return CPU and, if present, GPU device.

    Returns:
        devices: Available devices for ``torch``.
    """
    devices = [device("cpu")]

    if cuda.is_available():
        devices.append(device("cuda"))

    return devices


def classification_targets(size: Tuple[int], num_classes: int) -> Tensor:
    """Create random targets for classes ``0``, ..., ``num_classes - 1``.

    Args:
        size: Size of the targets to create.
        num_classes: Number of classes.

    Returns:
        Random targets.
    """
    return randint(size=size, low=0, high=num_classes)


def regression_targets(size: Tuple[int]) -> Tensor:
    """Create random targets for regression.

    Args:
        size: Size of the targets to create.

    Returns:
        Random targets.
    """
    return rand(*size)


def ggn_block_diagonal(
    model: Module,
    loss_func: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Tensor, Tensor]],
    separate_weight_and_bias: bool = True,
) -> ndarray:
    """Compute the block-diagonal GGN.

    Args:
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the GGN block-diagonals will be computed.
        data: A data loader.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal GGN. Default: ``True``.

    Returns:
        The block-diagonal GGN.
    """
    # compute the full GGN then zero out the off-diagonal blocks
    ggn = GGNLinearOperator(model, loss_func, params, data)
    ggn = from_numpy(ggn @ eye(ggn.shape[1]))
    sizes = [p.numel() for p in params]
    # ggn_blocks[i, j] corresponds to the block of (params[i], params[j])
    ggn_blocks = [list(block.split(sizes, dim=1)) for block in ggn.split(sizes, dim=0)]

    # find out which blocks to keep
    num_params = len(params)
    keep = [(i, i) for i in range(num_params)]
    param_ids = [p.data_ptr() for p in params]

    # keep blocks corresponding to jointly-treated weights and biases
    if not separate_weight_and_bias:
        # find all layers with weight and bias
        has_weight_and_bias = [
            mod
            for mod in model.modules()
            if hasattr(mod, "weight") and hasattr(mod, "bias") and mod.bias is not None
        ]
        # only keep those whose parameters are included
        has_weight_and_bias = [
            mod
            for mod in has_weight_and_bias
            if mod.weight.data_ptr() in param_ids and mod.bias.data_ptr() in param_ids
        ]
        for mod in has_weight_and_bias:
            w_pos = param_ids.index(mod.weight.data_ptr())
            b_pos = param_ids.index(mod.bias.data_ptr())
            keep.extend([(w_pos, b_pos), (b_pos, w_pos)])

    for i, j in product(range(num_params), range(num_params)):
        if (i, j) not in keep:
            ggn_blocks[i][j].zero_()

    # concatenate all blocks
    return cat([cat(row_blocks, dim=1) for row_blocks in ggn_blocks], dim=0).numpy()


class WeightShareModel(Sequential):
    """Sequential model with processing of the weight-sharing dimension.

    Wraps a ``Sequential`` model, but processes the weight-sharing dimension based
    on the ``setting`` before it returns the output of the sequential model.
    Assumes that the output of the sequential model is of shape
    ``(batch, ..., out_dim)``.
    """

    def __init__(self, *args: Module):
        """Initialize the model.

        Args:
            *args: Modules of the sequential model.
        """
        super().__init__(*args)
        self._setting = None

    @property
    def setting(self) -> str:
        """Return the setting of the model.

        Returns:
            The setting of the model.

        Raises:
            ValueError: If ``setting`` property has not been set.
        """
        if self._setting is None:
            raise ValueError("WeightShareModel.setting has not been set.")
        return self._setting

    @setting.setter
    def setting(self, setting: str):
        """Set the weight-sharing setting of the model.

        Args:
            setting: The weight-sharing setting of the model.

        Raises:
            ValueError: If ``setting`` is neither ``'expand'`` nor ``'reduce'``.
        """
        if setting not in {"expand", "reduce"}:
            raise ValueError(
                f"Expected 'setting' to be 'expand' or 'reduce', got {setting}."
            )
        self._setting = setting

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with processing of the weight-sharing dimension.

        Assumes MSELoss. The output would have to be transposed to be used with
        the CrossEntropyLoss.

        Args:
            x: Input to the forward pass.

        Returns:
            Output of the sequential model with processed weight-sharing dimension.
        """
        x = super().forward(x)
        if self.setting == "reduce":
            # Example: Vision transformer for image classification.
            # (batch, image_patches, c) -> (batch, c)
            return reduce(x, "batch ... c -> batch c", "mean")
        # if self.setting == "expand":
        # Example: Transformer for translation: (batch, sequence_length, c)
        # (although second and third dimension would have to be transposed for
        # classification)
        return x


class Conv2dModel(Module):
    """Sequential model with Conv2d module for expand and reduce setting."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self._setting = None
        self._models = {
            "expand": Sequential(
                Conv2d(3, 2, 4, padding=4 // 2),
                Rearrange("batch c h w -> batch h w c"),
            ),
            "reduce": Sequential(
                Conv2d(3, 2, 4, padding=4 // 2),
                AdaptiveAvgPool2d(1),
                Flatten(start_dim=1),
            ),
        }

    @property
    def setting(self) -> str:
        """Return the setting of the model.

        Returns:
            The setting of the model.

        Raises:
            ValueError: If `setting` property has not been set.
        """
        if self._setting is None:
            raise ValueError("Conv2dModel.setting has not been set.")
        return self._setting

    @setting.setter
    def setting(self, setting: str):
        """Set the weight-sharing setting of the model.

        Args:
            setting: The weight-sharing setting of the model.

        Raises:
            ValueError: If ``setting`` is neither ``'expand'`` nor ``'reduce'``.
        """
        if setting not in {"expand", "reduce"}:
            raise ValueError(
                f"Expected 'setting' to be 'expand' or 'reduce', got {setting}."
            )
        self._setting = setting
        self._model = self._models[setting]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sequential model based on the setting.

        Args:
            x: Input to the forward pass.

        Returns:
            Output of the sequential model.
        """
        return self._model(x)
