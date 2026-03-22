"""Contains a linear operator class for the diagonal of the GGN matrix."""

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import Module

from curvlinops.computers.ggn_diagonal import GGNDiagonalComputer
from curvlinops.diag import DiagonalLinearOperator


class GGNDiagonalLinearOperator(DiagonalLinearOperator):
    r"""Diagonal linear operator representing the GGN diagonal.

    Computes :math:`\mathrm{diag}(\mathbf{G})` where :math:`\mathbf{G}` is the
    generalized Gauss-Newton matrix (see :class:`GGNLinearOperator` for the full
    definition). When ``mc_samples > 0``, the loss Hessian is approximated via
    Monte-Carlo sampling from the model's predictive distribution (see
    :class:`GGNLinearOperator` for details).

    Internally uses a :class:`GGNDiagonalComputer` to compute the diagonal,
    then initializes the parent :class:`DiagonalLinearOperator` with the result.
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
        """Initialize the GGN diagonal linear operator.

        Constructs a :class:`GGNDiagonalComputer` with the given arguments,
        computes the diagonal, and passes it to the parent class.

        Args:
            model_func: The neural network's forward pass, defining the functional
                relationship ``(params, X) -> prediction``. Either an ``nn.Module``
                (architecture) or a callable ``(params_dict, X) -> prediction``.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value.
            params: The parameter values at which the GGN diagonal is evaluated. A
                dictionary mapping parameter names to tensors.
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
        computer = GGNDiagonalComputer(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
            mc_samples=mc_samples,
            seed=seed,
        )
        diagonal = computer.compute()
        super().__init__([diagonal[k] for k in computer._params])
