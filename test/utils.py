"""Utility functions to test ``curvlinops``."""

from collections.abc import MutableMapping
from itertools import product
from typing import Callable, Iterable, List, Optional, Tuple, Union

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from numpy import eye, ndarray
from torch import (
    Tensor,
    allclose,
    as_tensor,
    cat,
    cuda,
    device,
    dtype,
    from_numpy,
    rand,
    randint,
)
from torch.nn import (
    AdaptiveAvgPool2d,
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Identity,
    Module,
    MSELoss,
    Parameter,
    Sequential,
    Upsample,
)

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


def binary_classification_targets(size: Tuple[int]) -> Tensor:
    """Create random binary targets.

    Args:
        size: Size of the targets to create.

    Returns:
        Random targets (float).
    """
    return classification_targets(size, 2).float()


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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
    separate_weight_and_bias: bool = True,
) -> ndarray:
    """Compute the block-diagonal GGN.

    Args:
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the GGN block-diagonals will be computed.
        data: A data loader.
        batch_size_fn: A function that returns the batch size given a dict-like ``X``.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal GGN. Default: ``True``.

    Returns:
        The block-diagonal GGN.
    """
    # compute the full GGN then zero out the off-diagonal blocks
    ggn = GGNLinearOperator(model, loss_func, params, data, batch_size_fn=batch_size_fn)
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

    def __init__(self, *args: Module, setting: str = "expand", loss: str = "MSE"):
        """Initialize the model.

        Args:
            *args: Modules of the sequential model.
        """
        super().__init__(*args)
        self.setting = setting
        self.loss = loss

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

    @property
    def loss(self) -> str:
        """Return the type of loss function the model is used with.

        Returns:
            The type of loss function.
        """
        if self._loss is None:
            raise ValueError("WeightShareModel.loss has not been set.")
        return self._loss

    @loss.setter
    def loss(self, loss: str):
        """Set the type of loss function the model is used with.

        Args:
            loss: The type of loss function.

        Raises:
            ValueError: If ``loss`` is not one of ``MSE``, ``CE``, or ``BCE``.
        """
        if loss not in {"MSE", "CE", "BCE"}:
            raise ValueError(f"Expected loss to be 'MSE', 'CE', or 'BCE'. Got {loss}.")
        self._loss = loss

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
        if x.ndim > 2 and self.loss == "CE":
            x = rearrange(x, "batch ... c -> batch c ...")
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


class UnetModel(Module):
    """Simple Unet-like model where the number of spatial locations varies."""

    def __init__(self, loss: Module):
        """Initialize the model."""
        if loss not in {MSELoss, CrossEntropyLoss, BCEWithLogitsLoss}:
            raise ValueError(
                "Loss has to be one of MSELoss, CrossEntropyLoss, BCEWithLogitsLoss. "
                f"Got {loss}."
            )
        super().__init__()
        self._model = Sequential(
            Conv2d(3, 2, 3, padding=1, stride=2),
            Conv2d(2, 2, 3, padding=3 // 2),
            Upsample(scale_factor=2, mode="nearest"),
            Conv2d(2, 3, 3, padding=1),
            (
                Rearrange("batch c h w -> batch h w c")
                if issubclass(loss, (MSELoss, BCEWithLogitsLoss))
                else Identity()
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input to the forward pass.

        Returns:
            Output of the model.
        """
        return self._model(x)


def cast_input(
    X: Union[Tensor, MutableMapping], target_dtype: dtype
) -> Union[Tensor, MutableMapping]:
    """Cast an input tensor ``X`` (can be inside a dict-like object under the key "x")
        into ``target_dtype``.

    Args:
        X: The input tensor.
        target_dtype: Target ``torch`` data type.

    Returns:
        The casted tensor, preserved under the dict-like object, if applicable.
    """
    if isinstance(X, MutableMapping):
        X["x"] = X["x"].to(target_dtype)
    else:
        X = X.to(target_dtype)

    return X


def batch_size_fn(X: MutableMapping) -> int:
    """Get the batch size of a tensor wrapped in a dict-like object.

    Assumes that the key to that tensor is "x".

    Args:
        X: The dict-like object with key "x" and a corresponding tensor value.

    Returns:
        batch_size: The first dimension size of the tensor.
    """
    return X["x"].shape[0]


def compare_state_dicts(state_dict: dict, state_dict_new: dict):
    """Compare two state dicts recursively.

    Args:
        state_dict (dict): The first state dict to compare.
        state_dict_new (dict): The second state dict to compare.

    Raises:
        AssertionError: If the state dicts are not equal.
    """
    assert len(state_dict) == len(state_dict_new)
    for value, value_new in zip(state_dict.values(), state_dict_new.values()):
        if isinstance(value, Tensor):
            assert allclose(value, value_new)
        elif isinstance(value, dict):
            compare_state_dicts(value, value_new)
        elif isinstance(value, tuple):
            assert len(value) == len(value_new)
            assert all(isinstance(v, type(v2)) for v, v2 in zip(value, value_new))
            assert all(
                allclose(as_tensor(v), as_tensor(v2)) for v, v2 in zip(value, value_new)
            )
        else:
            assert value == value_new
