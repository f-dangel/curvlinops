"""Contains LinearOperator implementation of the GGN."""

from collections.abc import MutableMapping
from functools import cached_property, partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

from torch import Generator, Tensor, no_grad, vmap, zeros_like
from torch.func import jacrev, jvp, vjp
from torch.nn import BCEWithLogitsLoss, MSELoss, Module, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.kfac_utils import loss_hessian_matrix_sqrt, make_grad_output_sampler
from curvlinops.utils import make_functional_model_and_loss


def make_ggn_vector_product(
    f: Callable[..., Tensor], c: Callable[..., Tensor], num_c_extra_args: int = 0
) -> Callable[..., Tuple[Tensor, ...]]:
    """Create a function that computes GGN-vector products for given f and c functions.

    Args:
        f: Function that takes parameters and input, returns prediction.
            Signature: (*params, X) -> prediction
        c: Function that takes prediction, target, and optional additional args.
            Signature: (prediction, y, *args) -> loss
        num_c_extra_args: Number of additional arguments that the loss function c expects
            beyond prediction and target. Used to correctly split the input arguments
            between the vector to multiply and the additional loss function arguments.

    Returns:
        A function that computes GGN-vector products.
        Signature: (params, X, y, *c_args, *v) -> GGN @ v
        where c_args are additional arguments passed to the loss function c.
    """

    @no_grad()
    def ggn_vector_product(
        params: Tuple[Tensor, ...],
        X: Tensor,
        y: Tensor,
        *args_and_v: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, ...]:
        """Multiply the GGN on a vector in list format.

        Args:
            params: Parameters of the model.
            X: Input to the DNN.
            y: Ground truth.
            *args_and_v: Additional arguments for the loss function c,
                followed by vector to be multiplied with in tensor list format.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            the vector part of args_and_v.
        """
        # Split args_and_v into additional loss function arguments and vector v
        c_args, v = args_and_v[:num_c_extra_args], args_and_v[num_c_extra_args:]

        # Apply the Jacobian of f onto v: v → Jv
        f_val, f_jvp = jvp(lambda *params_inner: f(*params_inner, X), params, v)

        # Apply the criterion's Hessian onto Jv: Jv → HJv
        c_grad_func = jacrev(lambda pred: c(pred, y, *c_args))
        _, c_hvp = jvp(c_grad_func, (f_val,), (f_jvp,))

        # Apply the transposed Jacobian of f onto HJv: HJv → JᵀHJv
        # NOTE This re-evaluates the net's forward pass. [Unverified] It should be op-
        # timized away by common sub-expression elimination if you compile the function.
        _, f_vjp_func = vjp(lambda *params_inner: f(*params_inner, X), *params)
        return f_vjp_func(c_hvp)

    return ggn_vector_product


def make_batch_ggn_matrix_product(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Callable[
    [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
]:
    r"""Set up function that multiplies the mini-batch GGN onto a matrix in list format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the GGN is computed.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A function that takes inputs ``X``, ``y``, and a matrix ``M`` in list
        format, and returns the mini-batch GGN applied to ``M`` in list format.
    """
    # Create functional versions of the model (f: *params, X -> prediction) and
    # criterion function (c: prediction, y -> loss)
    f, c = make_functional_model_and_loss(model_func, loss_func, params)

    # Create the functional GGN-vector product
    ggn_vp = make_ggn_vector_product(f, c)  # params, X, y, *v -> *Gv

    # Fix the parameters
    ggnvp = partial(ggn_vp, params)  # X, y, *c_args, *v -> *Gv

    # Parallelize over vectors to multiply onto a matrix in list format
    list_format_vmap_dims = tuple(p.ndim for p in params)  # last axis
    return vmap(
        ggnvp,
        # No vmap in X, y, last-axis vmap over vector in list format
        in_dims=(None, None, *list_format_vmap_dims),
        # Vmapped output axis is last
        out_dims=list_format_vmap_dims,
        # We want each vector to be multiplied with the same mini-batch GGN
        randomness="same",
    )


class GGNLinearOperator(CurvatureLinearOperator):
    r"""Linear operator for the generalized Gauss-Newton matrix of an empirical risk.

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. The GGN matrix is

    .. math::
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            f_{\mathbf{\theta}}(\mathbf{x}_n)
        \right)^\top
        \left(
            \nabla_{f_\mathbf{\theta}(\mathbf{x}_n)}^2
            \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)
        \right)
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            f_{\mathbf{\theta}}(\mathbf{x}_n)
        \right)\,.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for GGNs.
    """

    SELF_ADJOINT: bool = True

    @cached_property
    def _mp(
        self,
    ) -> Callable[
        [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
    ]:
        """Lazy initialization of batch-GGN matrix product function.

        Returns:
            Function that computes mini-batch GGN-vector products, given inputs ``X``,
            labels ``y``, and the entries ``v1, v2, ...`` of the vector in list format.
            Produces a list of tensors with the same shape as the input vector that re-
            presents the result of the batch-GGN multiplication.
        """
        return make_batch_ggn_matrix_product(
            self._model_func, self._loss_func, tuple(self._params)
        )

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch GGN to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            ``M``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        return list(self._mp(X, y, *M))


class GGNDiagonalLinearOperator(CurvatureLinearOperator):
    """Linear operator for multiplication with the GGN diagonal.

    This operator represents multiplication by the diagonal of the Generalized
    Gauss-Newton matrix. The diagonal is computed by backpropagating the columns
    of the loss function's Hessian square root, then squaring and summing the
    results.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for
            diagonal matrices.
    """

    SELF_ADJOINT: bool = True
    FIXED_DATA_ORDER: bool = True

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        mode: str = "exact",
        seed: int = 2_147_483_647,
        mc_samples: int = 1,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ):
        """Linear operator for the GGN diagonal.

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
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.
        """
        assert mode in {"exact", "mc"}
        self._mode, self._seed, self._mc_samples = mode, seed, mc_samples
        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )

    @property
    def state(self) -> List[Tensor]:
        """Compute and cache the GGN diagonal.

        Returns:
            List of tensors representing the GGN diagonal, one for each parameter.
        """
        if not hasattr(self, "_state"):
            self._state = None

        if self._state is None:
            self._state = self._compute_ggn_diagonal()

        return self._state

    def refresh_state(self):
        """Refresh the internal state of the linear operator.

        Re-computes the GGN diagonal.
        """
        self._state = None
        # Accessing the property triggers the re-computation
        _ = self.state

    def _compute_ggn_diagonal(self) -> List[Tensor]:
        """Compute the GGN diagonal on the entire data set.

        Uses exact computation for mode='exact' and Monte Carlo approximation for
        mode='mc'.

        Returns:
            List of tensors containing the diagonal elements for each parameter.
        """
        # Create functional version of the model: (*params, x) -> prediction
        f, _ = make_functional_model_and_loss(
            self._model_func, self._loss_func, tuple(self._params)
        )

        # Set up vector generation and normalization based on mode
        if self._mode == "exact":

            def vector_generator_fn(f_x: Tensor, y: Tensor) -> Tensor:
                """Generate exact Hessian square root vectors."""
                hessian_sqrt = loss_hessian_matrix_sqrt(
                    f_x.unsqueeze(0), y.unsqueeze(0), self._loss_func
                )
                return hessian_sqrt.T

            normalization_factor = {"sum": 1.0, "mean": 1 / self._N_data}[
                self._loss_func.reduction
            ]
            use_generator = False
            generator = None
            desc = "GGN diagonal (exact)"

        elif self._mode == "mc":
            # Create the gradient output sampler for this loss function
            sample_grad_output = make_grad_output_sampler(self._loss_func)

            def vector_generator_fn(
                f_x: Tensor, y: Tensor, gen: Generator
            ) -> Tensor:
                """Generate MC sampled gradient vectors."""
                grad_output_samples = sample_grad_output(
                    f_x.unsqueeze(0), self._mc_samples, y.unsqueeze(0), gen
                ).squeeze(1)

                # Apply scaling for MSE with mean reduction
                scale = (
                    1.0 / f_x.numel()
                    if isinstance(self._loss_func, MSELoss)
                    and self._loss_func.reduction == "mean"
                    else 1.0
                )

                return grad_output_samples * scale**0.5

            normalization_factor = {
                "sum": 1.0 / self._mc_samples,
                "mean": 1.0 / self._N_data / self._mc_samples,
            }[self._loss_func.reduction]
            use_generator = True
            generator = Generator(self.device)
            generator.manual_seed(self._seed)
            desc = "GGN diagonal (MC)"
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        def ggn_diagonal_datum(
            x: Tensor, y: Tensor, gen: Optional[Generator] = None
        ) -> List[Tensor]:
            """Compute the GGN diagonal for a single datum.

            Args:
                x: Input datum.
                y: Label for the datum.
                gen: Generator for MC sampling (optional).

            Returns:
                List of tensors containing the diagonal elements for each parameter.
                Items have the same shape as the neural network's parameters.
            """
            f_x, f_vjp = vjp(lambda *params: f(*params, x), *self._params)
            if f_x.ndim != 1:
                raise RuntimeError("Sequence-valued predictions are unsupported.")

            # Generate vectors to backpropagate based on the strategy
            if use_generator:
                vectors = vector_generator_fn(f_x, y, gen)
            else:
                vectors = vector_generator_fn(f_x, y)

            if vectors.ndim != 2:
                raise RuntimeError("Expected 2d vectors for backpropagation.")

            gs = vmap(f_vjp)(vectors)
            return [(g**2).sum(0) for g in gs]

        if isinstance(self._loss_func, BCEWithLogitsLoss):
            raise RuntimeError("BCEWithLogitsLoss does not support vmap.")

        # Parallelize over data points
        if use_generator:
            ggn_diagonal_batched = vmap(
                ggn_diagonal_datum, in_dims=(0, 0, None), randomness="different"
            )
        else:
            ggn_diagonal_batched = vmap(ggn_diagonal_datum, in_dims=(0, 0))

        def ggn_diagonal(
            X: Union[MutableMapping, Tensor], y: Tensor
        ) -> List[Tensor]:
            """Compute the GGN diagonal on a batch.

            Args:
                X: Input batch.
                y: Labels for the batch.

            Returns:
                List of tensors containing the batch GGN's diagonal elements for each
                parameter. Items have the same shape as the neural network's
                parameters.
            """
            if not isinstance(X, Tensor):
                raise RuntimeError(
                    "Only tensor-valued inputs are supported by vmap."
                )

            if use_generator:
                return [res.sum(0) for res in ggn_diagonal_batched(X, y, generator)]
            else:
                return [res.sum(0) for res in ggn_diagonal_batched(X, y)]

        out = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc=desc):
            for out_p, batch_p in zip(out, ggn_diagonal(X, y), strict=True):
                out_p.add_(batch_p, alpha=normalization_factor)

        return out

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Multiply the GGN diagonal onto a matrix in list format.

        Args:
            M: Matrix to be multiplied in list format. Each item has the same shape as
                the corresponding neural network parameter, with an additional trailing
                dimension for the number of columns.

        Returns:
            Result of GGN diagonal multiplication in list format (same format as ``M``).
        """
        # Need to unsqueeze so that both tensors have the same number of dimensions
        # and the multiplication becomes broadcast-able
        return [M_p * G_p.unsqueeze(-1) for M_p, G_p in zip(M, self.state, strict=True)]
