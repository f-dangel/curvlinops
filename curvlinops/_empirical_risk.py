"""Mixin for classes that iterate over data to compute empirical risk quantities."""

from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

from torch import Tensor, device, dtype, tensor, zeros_like
from torch.autograd import grad
from torch.nn import Parameter
from tqdm import tqdm

from curvlinops.utils import _infer_device, _infer_dtype, allclose_report


class _EmpiricalRiskMixin:
    """Mixin for empirical risk computation over a data set."""

    FIXED_DATA_ORDER: bool = False

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
        num_data: Optional[int] = None,
        check_deterministic: bool = True,
    ):
        """Set up the shared state for empirical risk computation.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value. ``None`` means the represented quantity is independent
                of the loss function.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.
            progressbar: Show a progressbar during computation.
                Default: ``False``.
            batch_size_fn: Function that computes the batch size from input data. If
                ``None``, defaults to ``X.shape[0]``.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.

        Raises:
            ValueError: If ``X`` is a ``MutableMapping`` and ``batch_size_fn`` is not
                specified.
        """
        if isinstance(next(iter(data))[0], MutableMapping) and batch_size_fn is None:
            raise ValueError(
                "When using dict-like custom data, `batch_size_fn` is required."
            )

        self._model_func = model_func
        self._loss_func = loss_func
        self._params = params
        self._data = data
        self._progressbar = progressbar
        self._batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )

        self._N_data = (
            sum(
                self._batch_size_fn(X)
                for (X, _) in self._loop_over_data(desc="_N_data")
            )
            if num_data is None
            else num_data
        )

        if check_deterministic:
            self._check_deterministic()

    def _check_deterministic(self, rtol: float = 5e-5, atol: float = 1e-6):
        """Check that the data and model are deterministic.

        Two independent passes over the data must yield identical predictions,
        losses, and gradients. If ``FIXED_DATA_ORDER`` is ``True``, also checks
        that each mini-batch matches.

        Subclasses can override this method to add additional checks (e.g.
        verifying ``vmap`` compatibility). They should call ``super()`` first.

        Args:
            rtol: Relative tolerance for comparison. Default: ``5e-5``.
            atol: Absolute tolerance for comparison. Default: ``1e-6``.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        # Step 1: data determinism
        has_loss = self._loss_func is not None

        if has_loss:
            total_grad1 = [zeros_like(p) for p in self._params]
            total_grad2 = [zeros_like(p) for p in self._params]
            total_loss1 = tensor(0.0, device=self.device, dtype=self.dtype)
            total_loss2 = tensor(0.0, device=self.device, dtype=self.dtype)

        for (
            ((X1, y1), pred1, loss1, grad1),
            ((X2, y2), pred2, loss2, grad2),
        ) in zip(
            self._data_prediction_loss_gradient(),
            self._data_prediction_loss_gradient(),
        ):
            if self.FIXED_DATA_ORDER:
                self._check_deterministic_batch(
                    (X1, X2),
                    (y1, y2),
                    (pred1, pred2),
                    (loss1, loss2),
                    (grad1, grad2),
                    has_loss,
                    rtol=rtol,
                    atol=atol,
                )

            if has_loss:
                total_loss1.add_(loss1)
                total_loss2.add_(loss2)
                for tg1, g1 in zip(total_grad1, grad1):
                    tg1.add_(g1)
                for tg2, g2 in zip(total_grad2, grad2):
                    tg2.add_(g2)

        if has_loss:
            if not allclose_report(total_loss1, total_loss2, rtol=rtol, atol=atol):
                raise RuntimeError("Check for deterministic total loss failed.")
            if any(
                not allclose_report(g1, g2, atol=atol, rtol=rtol)
                for g1, g2 in zip(total_grad1, total_grad2)
            ):
                raise RuntimeError("Check for deterministic total gradient failed.")

    @staticmethod
    def _check_deterministic_batch(
        Xs,
        ys,
        predictions,
        losses,
        gradients,
        has_loss_func: bool,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """Compare two batch outputs of a data-prediction-loss-gradient pass.

        Args:
            Xs: The two data inputs to compare.
            ys: The two data targets to compare.
            predictions: The two predictions to compare.
            losses: The two losses to compare.
            gradients: The two gradients to compare.
            has_loss_func: Whether a loss function is present.
            rtol: Relative tolerance for comparison. Default: ``1e-5``.
            atol: Absolute tolerance for comparison. Default: ``1e-8``.

        Raises:
            RuntimeError: If any of the pairs mismatch.
        """
        X1, X2 = Xs
        if isinstance(X1, MutableMapping) and isinstance(X2, MutableMapping):
            for k in X1:
                v1, v2 = X1[k], X2[k]
                if isinstance(v1, Tensor) and not allclose_report(
                    v1, v2, rtol=rtol, atol=atol
                ):
                    raise RuntimeError("Check for deterministic X failed.")
        elif not allclose_report(X1, X2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic X failed.")

        y1, y2 = ys
        if not allclose_report(y1, y2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic y failed.")

        pred1, pred2 = predictions
        if not allclose_report(pred1, pred2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic batch prediction failed.")

        loss1, loss2 = losses
        grad1, grad2 = gradients
        if has_loss_func:
            if not allclose_report(loss1, loss2, rtol=rtol, atol=atol):
                raise RuntimeError("Check for deterministic batch loss failed.")
            if any(
                not allclose_report(g1, g2, rtol=rtol, atol=atol)
                for g1, g2 in zip(grad1, grad2)
            ):
                raise RuntimeError("Check for deterministic batch gradient failed.")

    @property
    def device(self) -> device:
        """Infer the device from model parameters.

        Returns:
            Inferred device.
        """
        return _infer_device(self._params)

    @property
    def dtype(self) -> dtype:
        """Infer the data type from model parameters.

        Returns:
            Inferred data type.
        """
        return _infer_dtype(self._params)

    def _loop_over_data(
        self,
        desc: Optional[str] = None,
    ) -> Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = self._data
        dev = self.device

        if self._progressbar:
            desc = (
                f"{self.__class__.__name__}"
                f"{'' if desc is None else f'.{desc}'}"
                f" (on {str(dev)})"
            )
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            if isinstance(X, Tensor):
                X = X.to(dev)
            y = y.to(dev)
            yield (X, y)

    def _get_normalization_factor(
        self, X: Union[MutableMapping, Tensor], y: Tensor
    ) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor.
        """
        return {"sum": 1.0, "mean": self._batch_size_fn(X) / self._N_data}[
            self._loss_func.reduction
        ]

    def _data_prediction_loss_gradient(
        self, desc: str = "batch_prediction_loss_gradient"
    ) -> Iterator[
        Tuple[
            Tuple[Union[Tensor, MutableMapping], Tensor],
            Tensor,
            Optional[Tensor],
            Optional[List[Tensor]],
        ]
    ]:
        """Yield (input, label), prediction, loss, and gradient for each batch.

        Args:
            desc: Description for the progress bar (if the progress bar is enabled).
                Default: ``'batch_prediction_loss_gradient'``.

        Yields:
            Tuple of ((input, label), prediction, loss, gradient) for each batch of
            the data.
        """
        for X, y in self._loop_over_data(desc=desc):
            prediction = self._model_func(X)
            if self._loss_func is None:
                loss, grad_params = None, None
            else:
                normalization_factor = self._get_normalization_factor(X, y)
                loss = self._loss_func(prediction, y).mul_(normalization_factor)
                grad_params = [g.detach() for g in grad(loss, self._params)]
                loss.detach_()

            yield (X, y), prediction, loss, grad_params

    def gradient_and_loss(self) -> Tuple[List[Tensor], Tensor]:
        """Evaluate the gradient and loss on the data.

        Returns:
            Gradient and loss on the data set.

        Raises:
            ValueError: If there is no loss function.
        """
        if self._loss_func is None:
            raise ValueError("No loss function specified.")

        total_loss = tensor([0.0], device=self.device, dtype=self.dtype).squeeze()
        total_grad = [zeros_like(p) for p in self._params]

        for _, _, loss, grad_params in self._data_prediction_loss_gradient(
            desc="gradient_and_loss"
        ):
            total_loss.add_(loss)
            for total_g, g in zip(total_grad, grad_params):
                total_g.add_(g)

        return total_grad, total_loss
