"""Utility functions to test ``curvlinops``."""

from collections.abc import MutableMapping
from contextlib import redirect_stdout, suppress
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
    zeros_like,
)
from torch import eye as torch_eye
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

from curvlinops import FisherMCLinearOperator, GGNLinearOperator
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.utils import allclose_report


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
    ggn = GGNLinearOperator(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    ).to_scipy()
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
            for v, v2 in zip(value, value_new):
                if v is None:
                    assert v2 is None
                else:
                    assert allclose(as_tensor(v), as_tensor(v2))
        else:
            assert value == value_new


def rand_accepted_formats(
    shapes: List[Tuple[int, ...]],
    is_vec: bool,
    dtype: dtype,
    device: device,
    num_vecs: int = 1,
) -> Tuple[List[Tensor], Tensor, ndarray]:
    """Generate a random vector/matrix in all accepted formats.

    Args:
        shapes: Sizes of the tensor product space.
        is_vec: Whether to generate representations of a vector or a matrix.
        dtype: Data type of the generated tensors.
        device: Device of the generated tensors.
        num_vecs: Number of vectors to generate. Ignored if ``is_vec`` is ``False``.
            Default: ``1``.

    Returns:
        M_tensor_list: Random vector/matrix in tensor list format.
        M_tensor: Random vector/matrix in tensor format.
        M_ndarray: Random vector/matrix in numpy format.
    """
    M_tensor_list = [
        rand(*shape, num_vecs, dtype=dtype, device=device) for shape in shapes
    ]
    M_tensor = cat([M.flatten(end_dim=-2) for M in M_tensor_list])

    if is_vec:
        M_tensor_list = [M.squeeze(-1) for M in M_tensor_list]
        M_tensor.squeeze(-1)

    M_ndarray = M_tensor.cpu().numpy()

    return M_tensor_list, M_tensor, M_ndarray


def compare_matmat(
    op: PyTorchLinearOperator,
    mat: Tensor,
    adjoint: bool,
    is_vec: bool,
    num_vecs: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Test the matrix-vector product of a PyTorch linear operator.

    Try all accepted formats for the input, as well as the SciPy-exported operator.

    Args:
        op: The operator to test.
        mat: The matrix representation of the linear operator.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
        num_vecs: Number of vectors to test (ignored if ``is_vec`` is ``True``).
            Default: ``2``.
        rtol: Relative tolerance for the comparison. Default: ``1e-5``.
        atol: Absolute tolerance for the comparison. Default: ``1e-8``.
    """
    if adjoint:
        op, mat = op.adjoint(), mat.conj().T

    num_vecs = 1 if is_vec else num_vecs
    dt = op._infer_dtype()
    dev = op._infer_device()
    x_list, x_tensor, x_numpy = rand_accepted_formats(
        [tuple(s) for s in op._in_shape], is_vec, dt, dev, num_vecs=num_vecs
    )

    tol = {"atol": atol, "rtol": rtol}

    # input in tensor format
    mat_x = mat @ x_tensor
    assert allclose_report(op @ x_tensor, mat_x, **tol)

    # input in numpy format
    op_scipy = op.to_scipy()
    op_x = op_scipy @ x_numpy
    assert type(op_x) is ndarray
    assert allclose_report(from_numpy(op_x).to(dev), mat_x, **tol)

    # input in tensor list format
    mat_x = [
        m_x.reshape(s if is_vec else (*s, num_vecs))
        for m_x, s in zip(mat_x.split(op._out_shape_flat), op._out_shape)
    ]
    op_x = op @ x_list
    assert len(op_x) == len(mat_x)
    for o_x, m_x in zip(op_x, mat_x):
        assert allclose_report(o_x, m_x, **tol)


def compare_consecutive_matmats(
    op: PyTorchLinearOperator,
    adjoint: bool,
    is_vec: bool,
    num_vecs: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Compare applying the linear operator to two identical vectors in sequence.

    Args:
        op: The operator to test.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
        num_vecs: Number of vectors to test (ignored if ``is_vec`` is ``True``).
            Default: ``2``.
        rtol: Relative tolerance for the comparison. Default: ``1e-5``.
        atol: Absolute tolerance for the comparison. Default: ``1e-8``.
    """
    if adjoint:
        op = op.adjoint()

    tol = {"atol": atol, "rtol": rtol}

    # Generate the vector using rand_accepted_formats
    dt = op._infer_dtype()
    dev = op._infer_device()
    _, X, _ = rand_accepted_formats(
        [tuple(s) for s in op._in_shape],
        is_vec=is_vec,
        dtype=dt,
        device=dev,
        num_vecs=num_vecs,
    )

    # Apply the operator twice to the same vector
    result1 = op @ X
    result2 = op @ X

    # Ensure the results are the same
    assert allclose_report(result1, result2, **tol)


def compare_matmat_expectation(
    op: FisherMCLinearOperator,
    mat: Tensor,
    adjoint: bool,
    is_vec: bool,
    max_repeats: int,
    check_every: int,
    num_vecs: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Test the matrix-vector product of a PyTorch linear operator in expectation.

    Args:
        op: The operator to test.
        mat: The matrix representation of the linear operator.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
        max_repeats: Maximum number of matrix-vector product within which the
            expectation must converge.
        check_every: Check the expectation every ``check_every`` iterations for
            convergence.
        num_vecs: Number of vectors to test (ignored if ``is_vec`` is ``True``).
            Default: ``2``.
        rtol: Relative tolerance for the comparison. Default: ``1e-5``.
        atol: Absolute tolerance for the comparison. Will be multiplied by the maximum
            absolute value of the ground truth. Default: ``1e-8``.
    """
    if adjoint:
        op, mat = op.adjoint(), mat.conj().T

    num_vecs = 1 if is_vec else num_vecs
    dt = op._infer_dtype()
    dev = op._infer_device()
    _, x, _ = rand_accepted_formats(
        [tuple(s) for s in op._in_shape], is_vec, dt, dev, num_vecs=num_vecs
    )

    op_x = zeros_like(x)
    mat_x = mat @ x

    atol *= mat_x.flatten().abs().max().item()
    tol = {"atol": atol, "rtol": rtol}

    for m in range(max_repeats):
        op_x += op @ x
        op._seed += 1

        total_samples = (m + 1) * op._mc_samples
        if total_samples % check_every == 0:
            with redirect_stdout(None), suppress(ValueError), suppress(AssertionError):
                assert allclose_report(op_x / (m + 1), mat_x, **tol)
                return

    assert allclose_report(op_x / max_repeats, mat_x, **tol)


def eye_like(A: Tensor) -> Tensor:
    """Create an identity matrix of same size as ``A``.

    Args:
        A: The tensor whose size determines the identity matrix.

    Returns:
        The identity matrix of ``A``'s size.
    """
    dim1, dim_2 = A.shape
    (dim,) = {dim1, dim_2}
    return torch_eye(dim, device=A.device, dtype=A.dtype)


def check_estimator_convergence(
    estimator: Callable[[], ndarray],
    num_matvecs: int,
    truth: float,
    max_total_matvecs: int = 100_000,
    check_every: int = 100,
    target_rel_error: float = 1e-3,
):
    """Test whether an estimator converges to the true value.

    Args:
        estimator: The estimator as function that accepts the number of matrix-vector
            products.
        num_matvecs: Number of matrix-vector products used per estimate.
        truth: True property of the linear operator.
        max_total_matvecs: Maximum number of matrix-vector products to perform.
            Default: ``100_000``. If convergence has not been reached by then, the test
            will fail.
        check_every: Check for convergence every ``check_every`` estimates.
            Default: ``100``.
        target_rel_error: Relative error for considering the estimator converged.
            Default: ``1e-3``.
    """
    used_matvecs, converged = 0, False

    def relative_l_inf_error(a_true: ndarray, a: ndarray) -> float:
        """Compute the relative infinity norm error.

        For scalars, this is simply | a - a_true | / | a_true |, the metric used by
        most trace estimation papers.

        For vector-/tensor-valued objects, this is the maximum relative error
        max(| a - a_true |) / max(| a_true |) where | . | denotes the element-wise
        absolute value. This metric is used by the XDiag paper to assess the
        quality of a diagonal estimator.

        Args:
            a_true: The true value.
            a: The estimated value.
        """
        assert a.shape == a_true.shape
        return abs(a - a_true).max() / abs(a_true).max()

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(estimator())
        used_matvecs += num_matvecs

        num_estimates = len(estimates)
        if num_estimates % check_every == 0:
            rel_error = relative_l_inf_error(truth, sum(estimates) / num_estimates)
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged
