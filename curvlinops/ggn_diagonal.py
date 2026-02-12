"""Contains a computer class for the diagonal of the GGN matrix."""

from collections import UserDict
from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union

from torch import Generator, Tensor, no_grad, zeros_like
from torch.func import vjp, vmap
from torch.nn import Module, Parameter

from curvlinops._checks import (
    _check_supports_batched_and_unbatched_inputs,
    _register_userdict_as_pytree,
)
from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops.kfac_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    make_grad_output_fn,
)
from curvlinops.utils import _seed_generator, make_functional_model_and_loss


def make_batch_ggn_diagonal_func(
    model_func: Module,
    loss_func: Module,
    params: Tuple[Parameter, ...],
    mode: str,
    mc_samples: int,
    batch_size_fn: Callable[[Union[Tensor, MutableMapping]], int],
) -> Callable[
    [Union[Tensor, MutableMapping], Tensor, Optional[Generator]],
    List[Tensor],
]:
    """Create a function that computes the GGN diagonal for a batch.

    Args:
        model_func: PyTorch module representing the neural network.
        loss_func: Loss function module.
        params: Tuple of model parameters.
        mode: Computation mode, either ``'exact'`` or ``'mc'``.
        mc_samples: Number of Monte Carlo samples (used when ``mode='mc'``).
        batch_size_fn: Function that returns the batch size given an input ``X``.
            If ``None``, defaults to using ``X.shape[0]`` for tensors or the first
            value's shape for MutableMapping inputs.

    Returns:
        Function with signature ``(X, y, generator) -> List[Tensor]``
        that computes the GGN diagonal on the batch ``(X, y)``.
    """
    # Create functional version of the model: (*params, x) -> prediction
    f, _ = make_functional_model_and_loss(model_func, loss_func, params)

    # Set up gradient output vector computation (binary target check is disabled
    # inside because it is incompatible with vmap; checked in batch_ggn_diagonal)
    grad_output_fn = make_grad_output_fn(loss_func, mode, mc_samples)
    reduction = loss_func.reduction

    def ggn_diagonal_datum(
        x: Union[Tensor, MutableMapping],
        y: Tensor,
        generator: Optional[Generator] = None,
    ) -> List[Tensor]:
        """Compute the GGN diagonal for a single datum.

        Args:
            x: Input datum.
            y: Label for the datum.
            generator: Generator for MC sampling (optional).

        Returns:
            List of tensors containing the diagonal elements for each parameter.
            Items have the same shape as the neural network's parameters.
        """
        f_x, f_vjp = vjp(lambda *p: f(*p, x), *params)
        # Detach f_x: only values are needed for the grad output vectors;
        # actual parameter gradients are computed via f_vjp.
        grad_outputs = grad_output_fn(f_x.detach(), y, generator)
        grad_params = vmap(f_vjp)(grad_outputs)
        return [(g**2).sum(0) for g in grad_params]

    randomness = {"mc": "different", "exact": "same"}[mode]
    # Parallelize over data points
    ggn_diagonal_batched = vmap(
        ggn_diagonal_datum, in_dims=(0, 0, None), randomness=randomness
    )

    @no_grad()
    def batch_ggn_diagonal(
        X: Union[Tensor, MutableMapping],
        y: Tensor,
        generator: Optional[Generator] = None,
    ) -> List[Tensor]:
        """Compute the GGN diagonal on a batch.

        Args:
            X: Input batch.
            y: Labels for the batch.
            generator: Random generator (optional).

        Returns:
            List of tensors containing the batch GGN's diagonal elements for each
            parameter. Items have the same shape as the neural network's
            parameters.
        """
        # Register UserDict as PyTree if needed for vmap compatibility
        if isinstance(X, UserDict):
            _register_userdict_as_pytree()
        # We turn off this check in the function that computes the GGN diagonal for a
        # single datum due to incompatibility with vmap. Therefore we need to re-introduce
        # this check here.
        _check_binary_if_BCEWithLogitsLoss(y, loss_func)

        # For mean reduction, we have to divide by the batch size to obtain correct
        # scale
        scale = {"sum": 1.0, "mean": 1.0 / batch_size_fn(X)}[reduction]
        return [res.sum(0).mul_(scale) for res in ggn_diagonal_batched(X, y, generator)]

    return batch_ggn_diagonal


class GGNDiagonalComputer(_EmpiricalRiskMixin):
    """Computes the diagonal of the Generalized Gauss-Newton matrix.

    This class handles data iteration, deterministic checks, and the actual
    computation of the GGN diagonal. Call ``.compute()`` to obtain the diagonal
    as a list of tensors.

    Attributes:
        SUPPORTED_MODES: Supported computation modes.
        FIXED_DATA_ORDER: Whether the data loader must return the same data
            for every iteration.
    """

    SUPPORTED_MODES: Tuple[str, ...] = ("exact", "mc")

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
        mode: str = "exact",
        seed: int = 2_147_483_647,
        mc_samples: int = 1,
    ):
        """Set up the GGN diagonal computation.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
                Could be a PyTorch module representing a neural network.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function.
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
            mode: Computation mode for the GGN diagonal. ``'exact'`` computes the
                exact diagonal using the loss Hessian's square root. ``'mc'`` uses
                Monte Carlo approximation with sampled gradients. Default: ``'exact'``.
            seed: Random seed for Monte Carlo sampling when ``mode='mc'``.
                Default: ``2147483647``.
            mc_samples: Number of Monte Carlo samples when ``mode='mc'``.
                Default: ``1``.

        Raises:
            ValueError: If mode is not one of the supported modes.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Must be one of {self.SUPPORTED_MODES}."
            )
        self.FIXED_DATA_ORDER = {"exact": False, "mc": True}[mode]
        self._mode = mode
        self._seed = seed
        self._mc_samples = mc_samples

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
        X, _ = next(iter(self._data))
        _check_supports_batched_and_unbatched_inputs(X, self._model_func)

    ###########################################################################
    #                        GGN DIAGONAL COMPUTATION                         #
    ###########################################################################

    def compute_ggn_diagonal(self) -> List[Tensor]:
        """Compute the GGN diagonal on the entire data set.

        Returns:
            List of tensors containing the diagonal elements for each parameter.
        """
        batch_ggn_diagonal_func = make_batch_ggn_diagonal_func(
            self._model_func,
            self._loss_func,
            tuple(self._params),
            self._mode,
            self._mc_samples,
            self._batch_size_fn,
        )

        generator = (
            None
            if self._mode == "exact"
            else _seed_generator(None, self.device, self._seed)
        )

        result = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc=f"GGN diagonal ({self._mode})"):
            batch_result = batch_ggn_diagonal_func(X, y, generator)
            normalization_factor = self._get_normalization_factor(X, y)
            for res_p, batch_p in zip(result, batch_result, strict=True):
                res_p.add_(batch_p, alpha=normalization_factor)

        return result
