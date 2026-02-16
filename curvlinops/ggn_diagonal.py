"""Contains a linear operator class for the diagonal of the GGN matrix."""

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import Parameter

from curvlinops.computers.ggn_diagonal import GGNDiagonalComputer
from curvlinops.diag import DiagonalLinearOperator


class GGNDiagonalLinearOperator(DiagonalLinearOperator):
    """Diagonal linear operator representing the GGN diagonal.

    Internally uses a :class:`GGNDiagonalComputer` to compute the diagonal,
    then initializes the parent :class:`DiagonalLinearOperator` with the result.
    """

    def __init__(
        self,
        model_func: Callable[[Tensor | MutableMapping], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        params: list[Parameter],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
        mode: str = "exact",
        seed: int = 2_147_483_647,
        mc_samples: int = 1,
    ):
        """Initialize the GGN diagonal linear operator.

        Constructs a :class:`GGNDiagonalComputer` with the given arguments,
        computes the diagonal, and passes it to the parent class.

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
            mode=mode,
            seed=seed,
            mc_samples=mc_samples,
        )
        diagonal = computer.compute()
        super().__init__(diagonal)
