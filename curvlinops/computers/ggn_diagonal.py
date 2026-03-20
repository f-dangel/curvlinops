"""Computer for the diagonal of the Generalized Gauss-Newton matrix."""

from collections import UserDict
from collections.abc import Callable, Iterable, MutableMapping
from functools import partial

from torch import Generator, Tensor, no_grad, zeros_like
from torch.func import vjp, vmap
from torch.nn import Module

from curvlinops._checks import (
    _check_supports_batched_and_unbatched_inputs,
    _register_userdict_as_pytree,
)
from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops.ggn_utils import make_grad_output_fn
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import _seed_generator


def make_batch_ggn_diagonal_func(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    loss_func: Module,
    mc_samples: int,
    batch_size_fn: Callable[[Tensor | MutableMapping], int],
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, Tensor, Generator | None],
    dict[str, Tensor],
]:
    """Create a function that computes the GGN diagonal for a batch.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.
        loss_func: Loss function module.
        mc_samples: Number of Monte Carlo samples. ``0`` uses the exact GGN diagonal
            via the loss Hessian's square root. Positive values use MC approximation.
        batch_size_fn: Function that returns the batch size given an input ``X``.
            If ``None``, defaults to using ``X.shape[0]`` for tensors or the first
            value's shape for MutableMapping inputs.

    Returns:
        Function with signature ``(params_dict, X, y, generator) -> dict[str, Tensor]``
        that computes the GGN diagonal on the batch ``(X, y)``.
    """
    fisher_type = FisherType.TYPE2 if mc_samples == 0 else FisherType.MC
    grad_output_fn = make_grad_output_fn(loss_func, fisher_type, mc_samples)
    reduction = loss_func.reduction

    def ggn_diagonal_datum(
        params: dict[str, Tensor],
        x: Tensor | MutableMapping,
        y: Tensor,
        generator: Generator | None = None,
    ) -> dict[str, Tensor]:
        """Compute the GGN diagonal for a single datum.

        Args:
            params: Parameters of the model as a dict.
            x: Input datum.
            y: Label for the datum.
            generator: Generator for MC sampling (optional).

        Returns:
            Dict mapping parameter names to diagonal elements. Each tensor has
            the same shape as the corresponding parameter.
        """
        f_x, f_vjp = vjp(lambda p: f(p, x), params)
        # Detach f_x: only values are needed for the grad output vectors;
        # actual parameter gradients are computed via f_vjp.
        grad_outputs = grad_output_fn(f_x.detach(), y, generator)
        (grad_params_dict,) = vmap(f_vjp)(grad_outputs)
        return {k: (grad_params_dict[k] ** 2).sum(0) for k in params}

    randomness = "different" if fisher_type == FisherType.MC else "same"
    # Parallelize over data points (vmap over x and y, not params or generator)
    ggn_diagonal_batched = vmap(
        ggn_diagonal_datum, in_dims=(None, 0, 0, None), randomness=randomness
    )

    @no_grad()
    def batch_ggn_diagonal(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        y: Tensor,
        generator: Generator | None = None,
    ) -> dict[str, Tensor]:
        """Compute the GGN diagonal on a batch.

        Args:
            params: Parameters of the model as a dict.
            X: Input batch.
            y: Labels for the batch.
            generator: Random generator (optional).

        Returns:
            Dict mapping parameter names to the batch GGN's diagonal elements.
            Each tensor has the same shape as the corresponding parameter.
        """
        # Register UserDict as PyTree if needed for vmap compatibility
        if isinstance(X, UserDict):
            _register_userdict_as_pytree()
        # For mean reduction, we have to divide by the batch size to obtain correct
        # scale
        scale = {"sum": 1.0, "mean": 1.0 / batch_size_fn(X)}[reduction]
        result_dict = ggn_diagonal_batched(params, X, y, generator)
        return {k: v.sum(0).mul_(scale) for k, v in result_dict.items()}

    return batch_ggn_diagonal


class GGNDiagonalComputer(_EmpiricalRiskMixin):
    """Computes the diagonal of the Generalized Gauss-Newton matrix.

    This class handles data iteration, deterministic checks, and the actual
    computation of the GGN diagonal. Call ``.compute()`` to obtain the
    diagonal as a list of tensors.

    Attributes:
        FIXED_DATA_ORDER: Whether the data loader must return the same data
            for every iteration. Set to ``True`` when ``mc_samples > 0``.
    """

    def __init__(
        self,
        model_func: Module
        | Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        params: dict[str, Tensor],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
        mc_samples: int = 0,
        seed: int = 2_147_483_647,
    ):
        """Set up the GGN diagonal computation.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: Either an ``nn.Module`` or a callable with signature
                ``(params_dict, X) -> prediction``.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value.
            params: Parameters for the model. Either a ``list[Parameter]`` (requires
                ``model_func`` to be a ``Module``) or a ``dict[str, Tensor]`` (requires
                ``model_func`` to be a callable).
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function. When using MC sampling, batches must be
                presented in the same deterministic order (no shuffling!).
            progressbar: Show a progressbar during computation.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: Function that computes the batch size from input data. For
                ``torch.Tensor`` inputs, this should typically return ``X.shape[0]``.
                For ``dict``/``UserDict`` inputs, this should return the batch size of
                the contained tensors.
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
                ``0`` (default) uses the exact GGN diagonal. Positive values activate
                the MC approximation.
            seed: Seed for the internal random number generator used for MC sampling.
                Only used when ``mc_samples > 0``. Default: ``2147483647``.
        """
        self._mc_samples = mc_samples
        if mc_samples > 0:
            self.FIXED_DATA_ORDER = True
        self._seed = seed

        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            batch_size_fn=batch_size_fn,
            num_data=num_data,
            check_deterministic=check_deterministic,
        )

    def _check_deterministic(self):
        """Check determinism and verify ``vmap`` compatibility.

        Extends the base class check by additionally verifying that the model
        supports both batched and un-batched inputs, which is required for
        ``vmap``.
        """
        super()._check_deterministic()
        X, _ = next(self._loop_over_data())
        _check_supports_batched_and_unbatched_inputs(
            X, partial(self._model_func, self._params)
        )

    def compute(self) -> dict[str, Tensor]:
        """Compute the GGN diagonal on the entire data set.

        Returns:
            Dict mapping parameter names to diagonal elements.
        """
        batch_ggn_diagonal_func = make_batch_ggn_diagonal_func(
            self._model_func, self._loss_func, self._mc_samples, self._batch_size_fn
        )

        generator = (
            None
            if self._mc_samples == 0
            else _seed_generator(None, self.device, self._seed)
        )

        result = {k: zeros_like(p) for k, p in self._params.items()}

        mode_str = "exact" if self._mc_samples == 0 else "mc"
        for X, y in self._loop_over_data(desc=f"GGN diagonal ({mode_str})"):
            batch_result = batch_ggn_diagonal_func(self._params, X, y, generator)
            normalization_factor = self._get_normalization_factor(X, y)
            for k in result:
                result[k].add_(batch_result[k], alpha=normalization_factor)

        return result
