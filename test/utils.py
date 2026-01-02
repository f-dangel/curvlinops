"""Utility functions to test ``curvlinops``."""

import os
from collections.abc import MutableMapping
from contextlib import redirect_stdout, suppress
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from numpy import ndarray
from pytest import raises, warns
from torch import (
    Tensor,
    allclose,
    as_tensor,
    cat,
    cuda,
    device,
    dtype,
    eye,
    from_numpy,
    linalg,
    load,
    logdet,
    manual_seed,
    rand,
    randint,
    randperm,
    save,
    trace,
    zeros_like,
)
from torch.nn import (
    AdaptiveAvgPool2d,
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Identity,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
    Upsample,
)

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    FisherMCLinearOperator,
    FisherType,
    GGNLinearOperator,
    KFACLinearOperator,
)
from curvlinops._torch_base import CurvatureLinearOperator, PyTorchLinearOperator
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


def maybe_exclude_or_shuffle_parameters(
    params: List[Parameter], model: Module, exclude: str, shuffle: bool
):
    """Maybe exclude or shuffle parameters.

    Args:
        params: List of parameters.
        model: The neural network.
        exclude: Parameter to exclude.
        shuffle: Whether to shuffle the parameters.

    Returns:
        List of parameters.
    """
    assert exclude in {None, "weight", "bias"}
    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]
    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]
    return params


def block_diagonal(
    linear_operator: Type[CurvatureLinearOperator],
    model: Module,
    loss_func: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
    separate_weight_and_bias: bool = True,
    optional_linop_args: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Compute the block-diagonal of the matrix induced by a linear operator.

    Args:
        linear_operator: The linear operator.
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the block-diagonal will be computed for.
        data: A data loader.
        batch_size_fn: A function that returns the batch size given a dict-like ``X``.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal. Default: ``True``.

    Returns:
        The block-diagonal matrix.
    """
    # compute the full matrix then zero out the off-diagonal blocks
    linop = linear_operator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        **(optional_linop_args or {}),
    )
    linop_mat = linop @ eye_like(linop)
    sizes = [p.numel() for p in params]
    # matrix_blocks[i, j] corresponds to the block of (params[i], params[j])
    matrix_blocks = [
        list(block.split(sizes, dim=1)) for block in linop_mat.split(sizes, dim=0)
    ]

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
            matrix_blocks[i][j].zero_()

    # concatenate all blocks
    return cat([cat(row_blocks, dim=1) for row_blocks in matrix_blocks], dim=0)


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
            ValueError: If ``setting`` is neither ``'expand'``,``'expand-flatten'``, nor
                ``'reduce'``.
        """
        if setting not in {"expand", "expand-flatten", "reduce"}:
            raise ValueError(
                "Expected 'setting' to be 'expand', 'expand-flatten', or 'reduce', got "
                f"{setting}."
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
        if self.setting == "expand-flatten":
            # Example: Pixel-wise MSE loss for diffusion model:
            # (batch, hight, width, c) -> (batch, hight * width * c)
            x = rearrange(x, "batch ... c -> batch (... c)")
        elif x.ndim > 2 and self.loss == "CE":
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
            "expand-flatten": Sequential(
                Conv2d(3, 2, 4, padding=4 // 2),
                Rearrange("batch c h w -> batch (h w c)"),
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
            ValueError: If ``setting`` is neither ``'expand'``, ``'expand-flatten'``,
                nor ``'reduce'``.
        """
        if setting not in {"expand", "expand-flatten", "reduce"}:
            raise ValueError(
                "Expected 'setting' to be 'expand', 'expand-flatten', or 'reduce', got "
                f"{setting}."
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

    def __init__(self, loss: Module, flatten: bool = False):
        """Initialize the model."""
        if loss not in {MSELoss, CrossEntropyLoss, BCEWithLogitsLoss}:
            raise ValueError(
                "Loss has to be one of MSELoss, CrossEntropyLoss, BCEWithLogitsLoss. "
                f"Got {loss}."
            )
        super().__init__()
        if issubclass(loss, (MSELoss, BCEWithLogitsLoss)):
            out_str = "batch (h w c)" if flatten else "batch h w c"
            rearrange_layer = Rearrange(f"batch c h w -> {out_str}")
        else:
            rearrange_layer = (
                Rearrange("batch c h w -> (batch h w) c") if flatten else Identity()
            )
        self._model = Sequential(
            Conv2d(3, 2, 3, padding=1, stride=2),
            Conv2d(2, 2, 3, padding=3 // 2),
            Upsample(scale_factor=2, mode="nearest"),
            Conv2d(2, 3, 3, padding=1),
            rearrange_layer,
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
        num_vecs: Number of vectors to generate. Will be overwritten to 1 if
            ``is_vec == True``. Default: ``1``.

    Returns:
        M_tensor_list: Random vector/matrix in tensor list format.
        M_tensor: Random vector/matrix in tensor format.
        M_ndarray: Random vector/matrix in numpy format.
    """
    num_vecs = 1 if is_vec else num_vecs

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
    dt = op.dtype
    dev = op.device
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
    dt, dev = op.dtype, op.device
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
    dt = op.dtype
    dev = op.device
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


def eye_like(A: Union[Tensor, PyTorchLinearOperator]) -> Tensor:
    """Create an identity matrix of same size as ``A``.

    Args:
        A: The tensor whose size determines the identity matrix.

    Returns:
        The identity matrix of ``A``'s size.
    """
    dim1, dim_2 = A.shape
    (dim,) = {dim1, dim_2}
    return eye(dim, dtype=A.dtype, device=A.device)


def check_estimator_convergence(
    estimator: Callable[[], Tensor],
    num_matvecs: int,
    truth: Tensor,
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

    def relative_l_inf_error(a_true: Tensor, a: Tensor) -> Tensor:
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
        return (a - a_true).abs().max() / a_true.abs().max()

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(estimator())
        used_matvecs += num_matvecs

        num_estimates = len(estimates)
        if num_estimates % check_every == 0:
            rel_error = relative_l_inf_error(truth, sum(estimates) / len(estimates))
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged


def _test_inplace_activations(
    linop_cls: Type[Union[KFACLinearOperator, EKFACLinearOperator]], dev: device
):
    """Test that (E)KFAC works if the network has in-place activations.

    We use a test case with a single datum as (E)KFAC becomes exact as the number of
    MC samples increases.

    Args:
        linop_cls: The linear operator class to test.
        dev: The device to run the test on.
    """
    manual_seed(0)
    model = Sequential(Linear(6, 3), ReLU(inplace=True), Linear(3, 2)).to(dev)
    loss_func = MSELoss().to(dev)
    batch_size = 1
    data = [(rand(batch_size, 6), regression_targets((batch_size, 2)))]
    params = list(model.parameters())

    # 1) compare (E)KFAC and GGN
    ggn = block_diagonal(GGNLinearOperator, model, loss_func, params, data)
    linop = linop_cls(model, loss_func, params, data, fisher_type=FisherType.TYPE2)
    linop_mat = linop @ eye_like(linop)
    assert allclose_report(ggn, linop_mat)

    # 2) Compare GGN (inplace=True) and GGN (inplace=False)
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    ggn_no_inplace = block_diagonal(GGNLinearOperator, model, loss_func, params, data)
    assert allclose_report(ggn, ggn_no_inplace)


def _test_property(  # noqa: C901
    linop_cls: Type[Union[KFACLinearOperator, EKFACLinearOperator]],
    property_name: str,
    model: Module,
    loss_func: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    batch_size_fn: Optional[Callable[[MutableMapping], int]],
    separate_weight_and_bias: bool,
    check_deterministic: bool,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Test a property of (E)KFAC.

    Args:
        linop_cls: The linear operator class to test.
        property_name: The property to test.
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the property will be computed.
        data: A data loader.
        batch_size_fn: A function that returns the batch size given a dict-like ``X``.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal.
        check_deterministic: Whether to check the property for a deterministic linear
            operator.
        rtol: Relative tolerance for the comparison. Default: ``1e-5``.
        atol: Absolute tolerance for the comparison. Default: ``1e-8``.
    """
    # Create instance of linear operator
    linop = linop_cls(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=check_deterministic,
    )

    # Add damping manually to avoid singular matrices for logdet
    if property_name == "logdet":
        DELTA = 1e-3
        if type(linop) is KFACLinearOperator:
            if not check_deterministic:
                linop.compute_kronecker_factors()
            assert linop._input_covariances or linop._gradient_covariances
            for aaT in linop._input_covariances.values():
                aaT.add_(eye_like(aaT), alpha=DELTA)
            for ggT in linop._gradient_covariances.values():
                ggT.add_(eye_like(ggT), alpha=DELTA)
        elif type(linop) is EKFACLinearOperator:
            if not check_deterministic:
                linop.compute_kronecker_factors()
                linop.compute_eigenvalue_correction()
            assert linop._corrected_eigenvalues
            for eigenvalues in linop._corrected_eigenvalues.values():
                if isinstance(eigenvalues, dict):
                    for eigenvals in eigenvalues.values():
                        eigenvals.add_(DELTA)
                else:
                    eigenvalues.add_(DELTA)

    # Mapping from the property name to the corresponding torch function
    torch_fn = {
        "trace": trace,
        "frobenius_norm": linalg.matrix_norm,
        "det": linalg.det,
        "logdet": logdet,
    }[property_name]

    # Check for equivalence of property and naive computation
    quantity = getattr(linop, property_name)
    linop_mat = linop @ eye_like(linop)
    quantity_naive = torch_fn(linop_mat)
    assert allclose_report(quantity, quantity_naive, rtol=rtol, atol=atol)

    # Check that the property is properly cached and reset
    assert getattr(linop, "_" + property_name) == quantity
    linop.compute_kronecker_factors()
    assert getattr(linop, "_" + property_name) is None


def _test_save_and_load_state_dict(
    linop_cls: Type[Union[KFACLinearOperator, EKFACLinearOperator]],
):
    """Test saving and loading state dict of (E)KFAC.

    Args:
        linop_cls: The linear operator class to test.
    """
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute linop
    linop = linop_cls(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # save state dict
    state_dict = linop.state_dict()
    PATH = "linop_state_dict.pt"
    save(state_dict, PATH)

    # create new linop with different loss function and try to load state dict
    linop_new = linop_cls(
        model,
        CrossEntropyLoss(),
        params,
        [(X, y)],
    )
    with raises(ValueError, match="loss"):
        linop_new.load_state_dict(load(PATH, weights_only=False))

    # create new linop with different loss reduction and try to load state dict
    linop_new = linop_cls(
        model,
        MSELoss(),
        params,
        [(X, y)],
    )
    with raises(ValueError, match="reduction"):
        linop_new.load_state_dict(load(PATH, weights_only=False))

    # create new linop with different model and try to load state dict
    wrong_model = Sequential(Linear(D_in, 10), ReLU(), Linear(10, D_out))
    wrong_params = list(wrong_model.parameters())
    linop_new = linop_cls(
        wrong_model,
        MSELoss(reduction="sum"),
        wrong_params,
        [(X, y)],
    )
    with raises(RuntimeError, match="loading state_dict"):
        linop_new.load_state_dict(load(PATH, weights_only=False))

    # create new linop and load state dict
    linop_new = linop_cls(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
        check_deterministic=False,  # turn off to avoid computing linop again
    )
    with warns(UserWarning, match="will overwrite the parameters"):
        linop_new.load_state_dict(load(PATH, weights_only=False))
    # clean up
    os.remove(PATH)

    # check that the two linops are equal
    compare_state_dicts(linop.state_dict(), linop_new.state_dict())
    test_vec = rand(linop.shape[1])
    assert allclose_report(linop @ test_vec, linop_new @ test_vec)


def _test_from_state_dict(
    linop_cls: Type[Union[KFACLinearOperator, EKFACLinearOperator]],
):
    """Test that (E)KFACLinearOperator can be created from state dict."""
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute (E)KFAC
    linop = linop_cls(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # save state dict
    state_dict = linop.state_dict()

    # create new linop from state dict
    linop_new = linop_cls.from_state_dict(state_dict, model, params, [(X, y)])

    # check that the two linops are equal
    compare_state_dicts(linop.state_dict(), linop_new.state_dict())
    test_vec = rand(linop.shape[1])
    assert allclose_report(linop @ test_vec, linop_new @ test_vec)


def _test_ekfac_closer_to_exact_than_kfac(
    model: Module,
    loss_func: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    batch_size_fn: Optional[Callable[[MutableMapping], int]],
    exclude: str,
    separate_weight_and_bias: bool,
    fisher_type: FisherType,
    kfac_approx: bool,
):
    """Test that EKFAC is closer in Frobenius norm to the exact quantity than KFAC.

    Args:
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the property will be computed.
        data: A data loader.
        batch_size_fn: A function that returns the batch size given a dict-like ``X``.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal.
        exclude: Parameter to exclude.
        fisher_type: The type of Fisher approximation.
        kfac_approx: THe type of KFAC approximation.
    """
    # Compute exact block-wise ground truth quantity.
    linop_cls = {
        FisherType.TYPE2: GGNLinearOperator,
        FisherType.MC: FisherMCLinearOperator,
        FisherType.EMPIRICAL: EFLinearOperator,
    }[fisher_type]
    optional_linop_args = {"seed": 1} if fisher_type == FisherType.MC else {}
    exact = block_diagonal(
        linop_cls,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        optional_linop_args=optional_linop_args,
    )

    # Compute KFAC and EKFAC.
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
        **optional_linop_args,
    )
    kfac_mat = kfac @ eye_like(kfac)
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
        **optional_linop_args,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    # Compute and compare (relative) distances to the exact quantity.
    exact_norm = linalg.matrix_norm(exact)
    exact_kfac_dist = linalg.matrix_norm(exact - kfac_mat) / exact_norm
    exact_ekfac_dist = linalg.matrix_norm(exact - ekfac_mat) / exact_norm
    assert exact_kfac_dist > exact_ekfac_dist or (
        allclose_report(exact_kfac_dist, exact_ekfac_dist, atol=1e-6)
        if exclude == "weight"
        else False
    )  # For no_weights the numerical error might dominate.
