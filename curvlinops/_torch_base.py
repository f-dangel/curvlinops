"""Defines a minimal ``LinearOperator`` interface in PyTorch."""

from __future__ import annotations

from typing import (
    Callable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import numpy
from scipy.sparse.linalg import LinearOperator
from torch import (
    Size,
    Tensor,
    as_tensor,
    bfloat16,
    cat,
    device,
    dtype,
    rand,
    zeros_like,
)
from torch.nn import Parameter

from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops.utils import allclose_report


class PyTorchLinearOperator:
    """Interface for linear operators in PyTorch.

    Heavily inspired by the Scipy interface
    (https://github.com/scipy/scipy/blob/v1.13.1/scipy/sparse/linalg/_interface.py),
    but only supports a sub-set of the functionality.

    One main difference is that the linear operators cannot only multiply
    vectors/matrices specified as single PyTorch tensors, but also
    vectors/matrices specified in tensor list format. This is common in
    PyTorch, where the space a linear operator acts on is a tensor product.

    Functions that need to be implemented are ``_matmat`` and ``_adjoint``.

    The interface also supports exporting the PyTorch linear operator to a SciPy linear
    operator, which can be useful for interfacing with SciPy routines. To achieve this,
    the properties ``.device`` and ``.dtype`` must be implemented.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. If ``True``,
            ``_adjoint`` does not need to be implemented. Default: ``False``.
    """

    SELF_ADJOINT: bool = False

    def __init__(
        self, in_shape: List[Tuple[int, ...]], out_shape: List[Tuple[int, ...]]
    ):
        """Store the linear operator's input and output space dimensions.

        Args:
            in_shape: A list of shapes specifying the linear operator's input space.
            out_shape: A list of shapes specifying the linear operator's output space.
        """
        self._in_shape = [Size(s) for s in in_shape]
        self._out_shape = [Size(s) for s in out_shape]

        self._in_shape_flat = [s.numel() for s in self._in_shape]
        self._out_shape_flat = [s.numel() for s in self._out_shape]
        self.shape = (sum(self._out_shape_flat), sum(self._in_shape_flat))

    def __rmatmul__(
        self, X: Union[List[Tensor], Tensor]
    ) -> Union[List[Tensor], Tensor]:
        """Multiply a tensor from the left onto the linear operator (``X @ A``).

        Args:
            X: A vector or matrix that left-multiplies the linear operator.
                Assume the linear operator has total shape ``[M, N]``:
                ``X`` can be of shape ``[M]`` (vector), or ``[K, M]`` (matrix).
                The result will have shape ``[N]`` or ``[K, N]``.
                ``X`` can also be a list of tensors representing vectors of shape
                ``[*M1], [*M2], ...`` or matrices of shape ``[K, *M1], [K, *M2], ...``,
                where ``M1, M2, ...`` are the output shapes of the operator.

        Returns:
            The result of the vector- or matrix-matrix multiplication in the same
            format as ``X``.
        """
        free_dim = "leading"  # position of axis in X to parallelize products over
        # Check and preprocess input, yielding a matrix in tensor list format with
        # leading matrix dimension (unsqueezed for the vector case)
        X, list_format, is_vec, num_vecs = self._check_input_and_preprocess(
            X, self._out_shape, free_dim
        )

        # Convert from leading to trailing dimension for compatibility with _matmat
        # Pre-processing always adds explicit matrix dimension, so always use movedim
        XH = [x.conj().movedim(0, -1) for x in X]

        # Use that X @ A = (A^H @ X^H)^H
        AH = self.adjoint()
        AH_XH = AH._matmat(XH)

        # Apply final conjugate to get (A^H @ X^H)^H
        AX = [r.conj().movedim(-1, 0) for r in AH_XH]

        # Postprocess output back to original format
        return self._check_output_and_postprocess(
            AX, list_format, is_vec, num_vecs, self._in_shape, free_dim
        )

    def __matmul__(
        self, X: Union[List[Tensor], Tensor, PyTorchLinearOperator]
    ) -> Union[List[Tensor], Tensor, _ChainPyTorchLinearOperator]:
        """Multiply onto a vector/matrix given as tensor/tensor list, or an operator.

        Args:
            X: A vector or matrix to multiply onto, represented as a single tensor or a
                tensor list, or another PyTorch linear operator.

                Assume the linear operator has total shape ``[M, N]``:
                If ``X`` is a single tensor, it can be of shape ``[N, K]`` (matrix), or
                ``[N]`` (vector). The result will have shape ``[M, K]`` or ``[M]``.

                Instead, we can also pass ``X`` as tensor list:
                Assume the linear operator's rows are formed by a list of shapes
                ``[M1, M2, ...]`` and the columns by ``[N1, N2, ...]``, such that
                ``M1.numel() + M2.numel() + ... = M`` and ``N1.numel() + N2.numel() +
                ... = N``. Then, ``X`` can also be a list of tensors with shape
                ``[*N1], [*N2], ...`` (vector) or ``[*N1, K], [*N2, K], ...`` (matrix).
                In this case, the output will be tensor list with shapes ``[*M1], [*M2],
                ...`` (vector) or ``[K, *M1], [K, *M2], ...`` (matrix).

        Returns:
            The result of the matrix-vector or matrix-matrix multiplication in the same
            format as ``X``, or a new linear operator representing the product of this
            and the passed linear operator.
        """
        free_dim = "trailing"  # position of axis in X to parallelize products over
        if isinstance(X, PyTorchLinearOperator):
            return _ChainPyTorchLinearOperator(self, X)

        # convert to tensor list format
        X, list_format, is_vec, num_vecs = self._check_input_and_preprocess(
            X, self._in_shape, free_dim
        )

        # matrix-matrix-multiply using tensor list format
        AX = self._matmat(X)

        # return same format as ``X`` passed by the user
        return self._check_output_and_postprocess(
            AX, list_format, is_vec, num_vecs, self._out_shape, free_dim
        )

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            X: A list of tensors representing the matrix to multiply onto.
                The list must contain tensors of shape ``[*N1, K], [*N2, K], ...``,
                where ``N1, N2, ...`` are the shapes of the linear operator's columns.

        Returns: # noqa: D402
            A list of tensors with shape ``[*M1, K], [*M2, K], ...``, where ``M1, M2,
            ...`` are the shapes of the linear operator's rows.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def adjoint(self) -> PyTorchLinearOperator:
        """Return the adjoint of the linear operator.

        Returns:
            The adjoint of the linear operator.
        """
        return self if self.SELF_ADJOINT else self._adjoint()

    def _adjoint(self) -> PyTorchLinearOperator:
        """Adjoint of the linear operator.

        Returns: # noqa: D402
            The adjoint of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def _check_input_and_preprocess(
        self,
        X: Union[List[Tensor], Tensor],
        expected_shapes: List[Size],
        free_dim: str,
    ) -> Tuple[List[Tensor], bool, bool, int]:
        """Check input format and pre-process it to a matrix in tensor list format.

        Args:
            X: The object onto which the linear operator is multiplied.
            expected_shapes: The expected shapes for tensor list format.
            free_dim: Whether the matrix dimension is ``"trailing"`` or
                ``"leading"``.

        Returns:
            X_tensor_list: The input object in tensor list format.
            list_format: Whether the input was specified in tensor list format.
                This is useful for post-processing the multiplication's result.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input format is invalid.
        """
        if isinstance(X, Tensor):
            list_format = False
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_and_preprocess(
                X, expected_shapes, free_dim
            )

        elif isinstance(X, list) and all(isinstance(x, Tensor) for x in X):
            list_format = True
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_list_and_preprocess(
                X, expected_shapes, free_dim
            )

        else:
            raise ValueError(f"Input must be tensor or list of tensors. Got {type(X)}.")

        return X_tensor_list, list_format, is_vec, num_vecs

    @staticmethod
    def __check_tensor_and_preprocess(
        X: Tensor, expected_shapes: List[Size], free_dim: str
    ) -> Tuple[List[Tensor], bool, int]:
        """Check single-tensor input format and process into a matrix tensor list.

        Args:
            X: The tensor onto which the linear operator is multiplied.
            expected_shapes: The expected shapes for the tensor list format.
            free_dim: Whether the matrix dimension is ``"trailing"`` or
                ``"leading"``.

        Returns:
            X_processed: The input tensor as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        expected_total_size = sum(s.numel() for s in expected_shapes)
        expected_flat_sizes = [s.numel() for s in expected_shapes]

        free_dim_pos = {"trailing": 1, "leading": 0}[free_dim]
        fixed_dim_pos = {"trailing": 0, "leading": -1}[free_dim]

        # Unified shape check using fixed_dim_pos
        if X.ndim > 2 or X.shape[fixed_dim_pos] != expected_total_size:
            shape_description = (
                f"({expected_total_size},) or "
                + {
                    "trailing": f"({expected_total_size}, K)",
                    "leading": f"(K, {expected_total_size})",
                }[free_dim]
            )
            raise ValueError(
                f"Input tensor must have shape {shape_description}, with "
                + f"K arbitrary. Got {X.shape}."
            )

        is_vec = X.ndim == 1
        num_vecs = 1 if is_vec else X.shape[free_dim_pos]

        X_processed = [
            x.reshape(*s, num_vecs)
            if free_dim == "trailing"
            else x.reshape(num_vecs, *s)
            for x, s in zip(
                X.split(expected_flat_sizes, dim=fixed_dim_pos),
                expected_shapes,
                strict=True,
            )
        ]

        return X_processed, is_vec, num_vecs

    @staticmethod
    def __check_tensor_list_and_preprocess(
        X: List[Tensor], expected_shapes: List[Size], free_dim: str
    ) -> Tuple[List[Tensor], bool, int]:
        """Check tensor list input format and process into a matrix tensor list.

        Args:
            X: The tensor list onto which the linear operator is multiplied.
            expected_shapes: The expected shapes for the tensor list format.
            free_dim: Whether the matrix dimension is ``"trailing"`` or
                ``"leading"``.

        Returns:
            X_processed: The input as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the tensor entries in the list have invalid shapes or there
                are too many/few entries.
        """
        free_dim_pos = {"trailing": -1, "leading": 0}[free_dim]

        if len(X) != len(expected_shapes):
            raise ValueError(
                f"Input list must have {len(expected_shapes)} tensors. "
                + f"Got {len(X)}."
            )

        # Check for vector format: [*N1], [*N2], ...
        if all(x.shape == s for x, s in zip(X, expected_shapes)):
            is_vec, num_vecs = True, 1

        # Check for matrix format: [*N1, K], [*N2, K], ... for trailing,
        # [K, *N1], [K, *N2], ... for leading
        elif (
            all(
                x.ndim == len(s) + 1
                and (x.shape[:-1] if free_dim == "trailing" else x.shape[1:]) == s
                for x, s in zip(X, expected_shapes)
            )
            and len({x.shape[free_dim_pos] for x in X}) == 1
        ):
            is_vec, (num_vecs,) = False, {x.shape[free_dim_pos] for x in X}

        else:
            raise ValueError(
                f"Input list must contain tensors with shapes {expected_shapes} "
                + f"and optional {free_dim} dimension for the matrix columns. "
                + f"Got {[x.shape for x in X]}."
            )

        X_processed = [x.unsqueeze(free_dim_pos) for x in X] if is_vec else X

        return X_processed, is_vec, num_vecs

    @staticmethod
    def _check_output_and_postprocess(
        AX: List[Tensor],
        list_format: bool,
        is_vec: bool,
        num_vecs: int,
        expected_shapes: List[Size],
        free_dim: str,
    ) -> Union[List[Tensor], Tensor]:
        """Check multiplication output and post-process it to the original format.

        Args:
            AX: The output of the multiplication as matrix in tensor list format.
                For ``free_dim="trailing"``, expects shapes
                ``[*S1, K], [*S2, K], ...``.
                For ``free_dim="leading"``, expects shapes
                ``[K, *S1], [K, *S2], ...`` (always with explicit matrix dimension).
            list_format: Whether the output should be in tensor list format.
            is_vec: Whether the output should be a vector or a matrix.
            num_vecs: The number of vectors represented by the output.
            expected_shapes: The expected shapes for the output.
            free_dim: Whether the matrix dimension is ``"trailing"`` or
                ``"leading"``.

        Returns:
            AX_processed: The output in the original format, either as single tensor
                or list of tensors.

        Raises:
            ValueError: If the output tensor list has an invalid shape or number
                of tensors.
        """
        free_dim_pos = {"trailing": -1, "leading": 0}[free_dim]
        fixed_dim_pos = {"trailing": 0, "leading": -1}[free_dim]
        expected_flat_sizes = [s.numel() for s in expected_shapes]

        if len(AX) != len(expected_shapes):
            raise ValueError(
                f"Output tensor list must have {len(expected_shapes)} tensors. "
                + f"Got {len(AX)}."
            )

        if any(
            Ax.shape != ((*s, num_vecs) if free_dim == "trailing" else (num_vecs, *s))
            for Ax, s in zip(AX, expected_shapes)
        ):
            raise ValueError(
                f"Output tensors must have shapes {expected_shapes} and additional "
                + f"{free_dim} dimension of {num_vecs}. "
                + f"Got {[Ax.shape for Ax in AX]}."
            )

        if list_format:
            AX_processed = [Ax.squeeze(free_dim_pos) for Ax in AX] if is_vec else AX

        else:
            AX_flattened = cat(
                [
                    Ax.reshape(s, num_vecs)
                    if free_dim == "trailing"
                    else Ax.reshape(num_vecs, s)
                    for Ax, s in zip(AX, expected_flat_sizes)
                ],
                dim=fixed_dim_pos,
            )
            AX_processed = (
                AX_flattened.squeeze(free_dim_pos) if is_vec else AX_flattened
            )

        return AX_processed

    ###########################################################################
    #                           OPERATOR COMPOSITION                          #
    ###########################################################################
    def __add__(self, other: PyTorchLinearOperator) -> _SumPyTorchLinearOperator:
        """Add another linear operator to the linear operator: A + B.

        Args:
            other: Another PyTorchLinearOperator to add.

        Returns:
            A new linear operator representing the sum A + B.
        """
        return _SumPyTorchLinearOperator(self, other)

    def __sub__(self, other: PyTorchLinearOperator) -> _SumPyTorchLinearOperator:
        """Subtract another linear operator from the linear operator: A - B.

        Args:
            other: Another PyTorchLinearOperator to subtract.

        Returns:
            A new linear operator representing the difference A - B.
        """
        return _SumPyTorchLinearOperator(self, -1.0 * other)

    def __mul__(self, scalar: Union[int, float]) -> _ScalePyTorchLinearOperator:
        """Multiply the linear operator by a scalar (A * scalar).

        Args:
            scalar: A scalar to multiply the linear operator with.

        Returns:
            A new linear operator representing the scaled linear operator.
        """
        return _ScalePyTorchLinearOperator(self, scalar)

    def __rmul__(self, scalar: Union[int, float]) -> _ScalePyTorchLinearOperator:
        """Right multiply the linear operator by a scalar (scalar * A).

        Args:
            scalar: A scalar to multiply the linear operator with.

        Returns:
            A new linear operator representing the scaled linear operator.
        """
        return self.__mul__(scalar)

    ###############################################################################
    #                                 SCIPY EXPORT                                #
    ###############################################################################

    def to_scipy(self, dtype: Optional[numpy.dtype] = None) -> LinearOperator:
        """Wrap the PyTorch linear operator with a SciPy linear operator.

        Args:
            dtype: The data type of the SciPy linear operator. If ``None``, uses
                NumPy's default data dtype.


        Returns:
            A SciPy linear operator that carries out the matrix-vector products
            in PyTorch.
        """
        dev, dt = self.device, self.dtype

        scipy_matmat = self._scipy_compatible(self.__matmul__, dev, dt)
        A_adjoint = self.adjoint()
        scipy_rmatmat = A_adjoint._scipy_compatible(A_adjoint.__matmul__, dev, dt)

        return LinearOperator(
            self.shape,
            matvec=scipy_matmat,
            rmatvec=scipy_rmatmat,
            matmat=scipy_matmat,
            rmatmat=scipy_rmatmat,
            dtype=numpy.dtype(dtype) if dtype is None else dtype,
        )

    @property
    def device(self) -> device:
        """Infer the linear operator's device.

        Returns:  # noqa: D402
            The device of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns: # noqa: D402
            The data type of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _check_deterministic_matvec(self, rtol: float = 1e-5, atol: float = 1e-8):
        """Probe whether the linear operator's matrix-vector product is deterministic.

        Performs two sequential matrix-vector products and compares them.

        Args:
            rtol: Relative tolerance for comparison. Defaults to ``1e-5``.
            atol: Absolute tolerance for comparison. Defaults to ``1e-8``.

        Raises:
            RuntimeError: If the two matrix-vector products yield different results.
        """
        v = rand(self.shape[1], device=self.device, dtype=self.dtype)
        Av1 = self @ v
        Av2 = self @ v
        if not allclose_report(Av1, Av2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic matvec failed.")

    @staticmethod
    def _scipy_compatible(
        f: Callable[[Tensor], Tensor], device: device, dtype: dtype
    ) -> Callable[[numpy.ndarray], numpy.ndarray]:
        """Wrap a PyTorch matrix multiplication function to be compatible with SciPy.

        Args:
            f: The PyTorch matrix multiplication function.
            device: The device on which the PyTorch linear operator is defined.
            dtype: The data type of the PyTorch linear operator.

        Returns:
            A function that takes a NumPy array and returns a NumPy array.
        """

        def f_scipy(X: numpy.ndarray) -> numpy.ndarray:
            """Scipy-compatible matrix multiplication function.

            Args:
                X: The input matrix in NumPy format.

            Returns:
                The output matrix in NumPy format.
            """
            X_torch = as_tensor(X, dtype=dtype, device=device)
            AX_torch = f(X_torch)
            # calling .numpy() on a BF-16 tensor is not supported, see
            # (https://github.com/pytorch/pytorch/issues/90574)
            if AX_torch.dtype == bfloat16:
                AX_torch = AX_torch.float()
            return AX_torch.detach().cpu().numpy().astype(X.dtype)

        return f_scipy


class _SumPyTorchLinearOperator(PyTorchLinearOperator):
    """Linear operator representing the sum of two linear operators A + B."""

    def __init__(self, A: PyTorchLinearOperator, B: PyTorchLinearOperator):
        """Initialize sum of two linear operators.

        Args:
            A: First linear operator.
            B: Second linear operator.

        Raises:
            ValueError: If the shapes, devices, or dtypes of the two linear
                operators do not match.
        """
        if A._in_shape != B._in_shape:
            raise ValueError(
                "Input shapes of linear operators must match:"
                + f"Got {A._in_shape} vs. {B._in_shape}."
            )
        if A._out_shape != B._out_shape:
            raise ValueError(
                "Output shapes of linear operators must match:"
                + f"Got {A._out_shape} vs. {B._out_shape}."
            )
        if A.device != B.device:
            raise ValueError(
                f"Devices of linear operators must match. Got {[A.device, B.device]}."
            )
        if A.dtype != B.dtype:
            raise ValueError(
                f"Dtypes of linear operators must match. Got {[A.dtype, B.dtype]}."
            )
        super().__init__(A._in_shape, A._out_shape)
        self._A, self._B = A, B

        # Sum is self-adjoint if both operands are self-adjoint
        self.SELF_ADJOINT = A.SELF_ADJOINT and B.SELF_ADJOINT

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply the linear operator onto a matrix in list format.

        Args:
            X: The matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        return [ax + bx for ax, bx in zip(self._A._matmat(X), self._B._matmat(X))]

    def _adjoint(self) -> _SumPyTorchLinearOperator:
        """Return the linear operator's adjoint: (A + B)* = A* + B*.

        Returns:
            A linear operator representing the adjoint.
        """
        return _SumPyTorchLinearOperator(self._A.adjoint(), self._B.adjoint())

    @property
    def device(self) -> device:
        """Determine the device the linear operators is defined on.

        Returns:
            The linear operator's device.
        """
        return self._A.device

    @property
    def dtype(self) -> dtype:
        """Determine the linear operator's data type.

        Returns:
            The linear operator's dtype.
        """
        return self._A.dtype


class _ScalePyTorchLinearOperator(PyTorchLinearOperator):
    """Linear operator representing the scaled version of a linear operator s * A."""

    def __init__(self, A: PyTorchLinearOperator, scalar: Union[float, int]):
        """Store the linear operator.

        Args:
            A: The linear operator.
            scalar: The scaling factor.
        """
        super().__init__(A._in_shape, A._out_shape)
        self._A = A
        self._scalar = scalar
        self.SELF_ADJOINT = A.SELF_ADJOINT

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply the linear operator onto a matrix in list format.

        Args:
            X: Matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        return [self._scalar * AX for AX in self._A._matmat(X)]

    def _adjoint(self) -> _ScalePyTorchLinearOperator:
        """Return a linear operator representing the adjoint: (cA)* = cA*.

        Returns:
            The linear operator's adjoint.
        """
        return _ScalePyTorchLinearOperator(self._A.adjoint(), self._scalar)

    @property
    def device(self) -> device:
        """Determine the device the linear operators is defined on.

        Returns:
            The linear operator's device.
        """
        return self._A.device

    @property
    def dtype(self) -> dtype:
        """Determine the linear operator's data type.

        Returns:
            The linear operator's dtype.
        """
        return self._A.dtype


class _ChainPyTorchLinearOperator(PyTorchLinearOperator):
    """Linear operator representing the product of two linear operators A @ B."""

    def __init__(self, A: PyTorchLinearOperator, B: PyTorchLinearOperator):
        """Initialize product of two linear operators.

        Args:
            A: First linear operator.
            B: Second linear operator.

        Raises:
            ValueError: If the shapes, devices, or dtypes of the two linear
                operators are incompatible.
        """
        if A._in_shape != B._out_shape:
            raise ValueError(f"{A._in_shape=} does not match {B._out_shape}.")
        if A.device != B.device:
            raise ValueError(
                f"Devices of linear operators must match. Got {[A.device, B.device]}."
            )
        if A.dtype != B.dtype:
            raise ValueError(
                f"Dtypes of linear operators must match. Got {[A.dtype, B.dtype]}."
            )
        self._A, self._B = A, B

        # Inherit shapes from the operands
        super().__init__(B._in_shape, A._out_shape)

    @property
    def dtype(self) -> dtype:
        """Determine the linear operator's data type.

        Returns:
            The linear operator's dtype.
        """
        return self._A.dtype

    @property
    def device(self) -> device:
        """Determine the device the linear operators is defined on.

        Returns:
            The linear operator's device.
        """
        return self._A.device

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply the linear operator onto a matrix in list format.

        Args:
            X: The matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        return self._A._matmat(self._B._matmat(X))

    def _adjoint(self) -> _ChainPyTorchLinearOperator:
        """Return the linear operator's adjoint: (AB)* = B*A*.

        Returns:
            A linear operator representing the adjoint.
        """
        return _ChainPyTorchLinearOperator(self._B.adjoint(), self._A.adjoint())


class CurvatureLinearOperator(_EmpiricalRiskMixin, PyTorchLinearOperator):
    """Base class for PyTorch linear operators of deep learning curvature matrices.

    To implement a new curvature linear operator, subclass this class and implement
    the ``_matmat_batch`` and ``_adjoint`` methods. If the linear operator is not
    defined as a map in the neural network's parameter space, you also need to
    implement ``_get_in_shape`` and ``_get_out_shape``.

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports multiplication with
            a block-diagonal approximation rather than the full matrix.
            Default: ``False``.
        FIXED_DATA_ORDER: Whether the data loader must return the same data
            for every iteration. Default: ``False``.
    """

    SUPPORTS_BLOCKS: bool = False
    FIXED_DATA_ORDER: bool = False

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        block_sizes: Optional[List[int]] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ):
        """Linear operator for curvature matrices of empirical risks.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
                Could be a PyTorch module representing a neural network.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value. If ``None``, there is no loss function and the
                represented matrix is independent of the loss function.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function.
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            block_sizes: This argument will be ignored if the linear operator does not
                support blocks. List of integers indicating the number of
                ``nn.Parameter``s forming a block. Entries must sum to ``len(params)``.
                For instance ``[len(params)]`` considers the full matrix, while
                ``[1, 1, ...]`` corresponds to a block diagonal approximation where
                each parameter forms its own block.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Raises:
            ValueError: If ``block_sizes`` is specified but the linear operator does not
                support blocks.
            ValueError: If the sum of blocks does not equal the number of parameters.
            ValueError: If any block size is not positive.
            ValueError: If ``X`` is not a tensor and ``batch_size_fn`` is not specified.
        """
        if block_sizes is not None:
            if not self.SUPPORTS_BLOCKS:
                raise ValueError(
                    "Block sizes were specified but operator does not support blocking."
                )
            if sum(block_sizes) != len(params):
                raise ValueError("Sum of blocks must equal the number of parameters.")
            if any(s <= 0 for s in block_sizes):
                raise ValueError("Block sizes must be positive.")
        self._block_sizes = [len(params)] if block_sizes is None else block_sizes

        _EmpiricalRiskMixin.__init__(
            self,
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            batch_size_fn=batch_size_fn,
            num_data=num_data,
            check_deterministic=check_deterministic,
        )
        PyTorchLinearOperator.__init__(
            self, self._get_in_shape(), self._get_out_shape()
        )

        if check_deterministic:
            self._check_deterministic_matvec()

    def _get_in_shape(self) -> List[Tuple[int, ...]]:
        """Return linear operator's input space dimensions.

        Returns:
            Shapes of the linear operator's input tensor product space.
        """
        return [tuple(p.shape) for p in self._params]

    def _get_out_shape(self) -> List[Tuple[int, ...]]:
        """Return linear operator's output space dimensions.

        Returns:
            Shapes of the linear operator's output tensor product space.
        """
        return [tuple(p.shape) for p in self._params]

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            M: Matrix for multiplication in tensor list format. Assume the linear
                operator's input tensor product space consists of shapes ``[*N1],
                [*N2], ...``. Then, ``M`` is a list of tensors with shapes
                ``[*N1, K], [*N2, K], ...`` with ``K`` the number of columns.

        Returns:
            Matrix-multiplication result ``mat @ M`` in tensor list format.
            Has same format as the input matrix, but lives in the linear operator's
            output tensor product space.
        """
        AM = [zeros_like(m) for m in M]

        for X, y in self._loop_over_data(desc="_matmat"):
            normalization_factor = self._get_normalization_factor(X, y)
            for AM_current, current in zip(AM, self._matmat_batch(X, y, M)):
                AM_current.add_(current, alpha=normalization_factor)

        return AM

    def _matmat_batch(
        self, X: Union[MutableMapping, Tensor], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix in list format (same shape as trainable model parameters with
                additional trailing dimension of size number of columns).

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError
