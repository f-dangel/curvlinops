"""Defines a minimal ``LinearOperator`` interface in PyTorch."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy
from scipy.sparse.linalg import LinearOperator
from torch import Size, Tensor, cat, device, dtype, from_numpy


class PyTorchLinearOperator:
    """Interface for linear operators in PyTorch.

    Heavily inspired by the Scipy interface
    (https://github.com/scipy/scipy/blob/v1.13.1/scipy/sparse/linalg/_interface.py),
    but only supports a sub-set of the functionality.

    One main difference is that the linear operators cannot only multiply
    vectors/matrices specified as single PyTorch tensors, but also
    vectors/matrices specified in tensor list format. This is common in
    PyTorch, where the space a linear operator acts on is a tensor product

    Functions that need to be implemented are ``_matmat`` and ``_adjoint``.

    The interface also supports exporting the PyTorch linear operator to a SciPy linear
    operator, which can be useful for interfacing with SciPy routines. To achieve this,
    the functions ``_infer_device`` and ``_infer_dtype`` must be implemented.

    """

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

    def __matmul__(self, X: Union[List[Tensor], Tensor]) -> Union[List[Tensor], Tensor]:
        """Multiply onto a vector or matrix given as PyTorch tensor or tensor list.

        Args:
            X: A vector or matrix to multiply onto, represented as a single tensor or a
                tensor list.

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
            format as ``X``.
        """
        # convert to tensor list format
        X, list_format, is_vec, num_vecs = self._check_input_and_preprocess(X)

        # matrix-matrix-multiply using tensor list format
        AX = self._matmat(X)

        # return same format as ``X`` passed by the user
        return self._check_output_and_postprocess(AX, list_format, is_vec, num_vecs)

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
        return self._adjoint()

    def _adjoint(self) -> PyTorchLinearOperator:
        """Adjoint of the linear operator.

        Returns: # noqa: D402
            The adjoint of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def _check_input_and_preprocess(
        self, X: Union[List[Tensor], Tensor]
    ) -> Tuple[List[Tensor], bool, bool, int]:
        """Check input format and pre-process it to a matrix in tensor list format.

        Args:
            X: The object onto which the linear operator is multiplied.

        Returns:
            X_tensor_list: The input object in tensor list format.
            list_format: Whether the input was specified in tensor list format.
                This is useful for post-processing the multiplication's result.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.
        """
        if isinstance(X, Tensor):
            list_format = False
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_and_preprocess(X)

        elif isinstance(X, list) and all(isinstance(x, Tensor) for x in X):
            list_format = True
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_list_and_preprocess(X)

        else:
            raise ValueError(f"Input must be tensor or list of tensors. Got {type(X)}.")

        return X_tensor_list, list_format, is_vec, num_vecs

    def __check_tensor_and_preprocess(
        self, X: Tensor
    ) -> Tuple[List[Tensor], bool, int]:
        """Check single-tensor input format and process into a matrix tensor list.

        Args:
            X: The tensor onto which the linear operator is multiplied.

        Returns:
            X_processed: The input tensor as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        if X.ndim > 2 or X.shape[0] != self.shape[1]:
            raise ValueError(
                f"Input tensor must have shape ({self.shape[1]},) or "
                + f"({self.shape[1]}, K), with K arbitrary. Got {X.shape}."
            )

        # determine whether the input is a vector or matrix
        is_vec = X.ndim == 1
        num_vecs = 1 if is_vec else X.shape[1]

        # convert to matrix in tensor list format
        X_processed = [
            x.reshape(*s, num_vecs)
            for x, s in zip(X.split(self._in_shape_flat), self._in_shape)
        ]

        return X_processed, is_vec, num_vecs

    def __check_tensor_list_and_preprocess(
        self, X: List[Tensor]
    ) -> Tuple[List[Tensor], bool, int]:
        """Check tensor list input format and process into a matrix tensor list.

        Args:
            X: The tensor list onto which the linear operator is multiplied.

        Returns:
            X_processed: The input as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the tensor entries in the list have invalid shapes.
        """
        if len(X) != len(self._in_shape):
            raise ValueError(
                f"List must contain {len(self._in_shape)} tensors. Got {len(X)}."
            )

        # check if input is a vector or a matrix
        if all(x.shape == s for x, s in zip(X, self._in_shape)):
            is_vec, num_vecs = True, 1
        elif (
            all(
                x.ndim == len(s) + 1 and x.shape[:-1] == s
                for x, s in zip(X, self._in_shape)
            )
            and len({x.shape[-1] for x in X}) == 1
        ):
            is_vec, (num_vecs,) = False, {x.shape[-1] for x in X}
        else:
            raise ValueError(
                f"Input list must contain tensors with shapes {self._in_shape} "
                + "and optional trailing dimension for the matrix columns. "
                + f"Got {[x.shape for x in X]}."
            )

        # convert to matrix in tensor list format
        X_processed = [x.unsqueeze(-1) for x in X] if is_vec else X

        return X_processed, is_vec, num_vecs

    def _check_output_and_postprocess(
        self, AX: List[Tensor], list_format: bool, is_vec: bool, num_vecs: int
    ) -> Union[List[Tensor], Tensor]:
        """Check multiplication output and post-process it to the original format.

        Args:
            AX: The output of the multiplication as matrix in tensor list format.
            list_format: Whether the output should be in tensor list format.
            is_vec: Whether the output should be a vector or a matrix.
            num_vecs: The number of vectors represented by the output.

        Returns:
            AX_processed: The output in the original format, either as single tensor
                or list of tensors.

        Raises:
            ValueError: If the output tensor list has an invalid length or shape.
        """
        # verify output tensor list format
        if len(AX) != len(self._out_shape):
            raise ValueError(
                f"Output list must contain {len(self._out_shape)} tensors. Got {len(AX)}."
            )
        if any(Ax.shape != (*s, num_vecs) for Ax, s in zip(AX, self._out_shape)):
            raise ValueError(
                f"Output tensors must have shapes {self._out_shape} and additional "
                + f"trailing dimension of {num_vecs}. "
                + f"Got {[Ax.shape for Ax in AX]}."
            )

        if list_format:
            AX_processed = [Ax.squeeze(-1) for Ax in AX] if is_vec else AX
        else:
            AX_processed = cat(
                [Ax.reshape(s, num_vecs) for Ax, s in zip(AX, self._out_shape_flat)]
            )
            AX_processed = AX_processed.squeeze(-1) if is_vec else AX_processed

        return AX_processed

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
        dev = self._infer_device()
        dt = self._infer_dtype()

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

    def _infer_device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The device of the linear operator.

        Raises: # noqa: D402
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _infer_dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The data type of the linear operator.

        Raises: # noqa: D402
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

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
            X_dtype = X.dtype
            X_torch = from_numpy(X).to(device, dtype)
            AX_torch = f(X_torch)
            return AX_torch.detach().cpu().numpy().astype(X_dtype)

        return f_scipy
