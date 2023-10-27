"""Linear operator for the Fisher/GGN's Kronecker-factored approximation.

Kronecker-factored approximate curvature was originally introduced for MLPs in

- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
  approximate curvature. International Conference on Machine Learning (ICML).

and extended to CNNs in

- Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
  convolution layers. International Conference on Machine Learning (ICML).
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List, Tuple, Union

from numpy import ndarray
from torch import Generator, Tensor, einsum, randn
from torch.nn import CrossEntropyLoss, Linear, Module, MSELoss, Parameter
from torch.utils.hooks import RemovableHandle

from curvlinops._base import _LinearOperator


class KFACLinearOperator(_LinearOperator):
    """Linear operator to multiply with Fisher/GGN's KFAC approximation."""

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss)

    def __init__(
        self,
        model_func: Module,
        loss_func: Union[MSELoss, CrossEntropyLoss],
        params: List[Parameter],
        data: Iterable[Tuple[Tensor, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        shape: Union[Tuple[int, int], None] = None,
        seed: int = 2147483647,
        mc_samples: int = 1,
    ):
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )

        # TODO Check for only linear layers
        idx = 0
        for mod in model_func.modules():
            if len(list(mod.modules())) == 1 and list(mod.parameters()):
                assert isinstance(mod, Linear)
                assert mod.bias is not None
                assert params[idx].data_ptr() == mod.weight.data_ptr()
                assert params[idx + 1].data_ptr() == mod.bias.data_ptr()
                idx += 2

        self._seed = seed
        self._generator: Union[None, Generator] = None
        self._mc_samples = mc_samples
        self._input_covariances: Dict[Tuple[int, ...], Tensor] = {}
        self._gradient_covariances: Dict[Tuple[int, ...], Tensor] = {}

        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            shape=shape,
        )

    def _matvec(self, x: ndarray) -> ndarray:
        """Loop over all batches in the data and apply the matrix to vector x.

        Create and seed the random number generator.

        Args:
            x: Vector for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ x``.
        """
        if not self._input_covariances and not self._gradient_covariances:
            self._compute_kfac()

        x_torch = super()._preprocess(x)
        assert len(x_torch) % 2 == 0

        for idx in range(len(x_torch) // 2):
            idx_weight, idx_bias = 2 * idx, 2 * idx + 1
            weight, bias = self._params[idx_weight], self._params[idx_bias]
            x_weight, x_bias = x_torch[idx_weight], x_torch[idx_bias]

            aaT = self._input_covariances[(weight.data_ptr(), bias.data_ptr())]
            ggT = self._gradient_covariances[(weight.data_ptr(), bias.data_ptr())]

            x_torch[idx_weight] = ggT @ x_weight @ aaT
            x_torch[idx_bias] = ggT @ x_bias

        return super()._postprocess(x_torch)

    def _adjoint(self) -> KFACLinearOperator:
        """Return the linear operator representing the adjoint.

        The KFAC approximation is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self

    def _compute_kfac(self):
        # install forward and backward hooks on layers
        hook_handles: List[RemovableHandle] = []

        modules = []
        for mod in self._model_func.modules():
            if len(list(mod.modules())) == 1 and list(mod.parameters()):
                assert isinstance(mod, Linear)
                modules.append(mod)
        hook_handles.extend(
            mod.register_forward_pre_hook(self._hook_accumulate_input_covariance)
            for mod in modules
        )
        hook_handles.extend(
            mod.register_full_backward_hook(self._hook_accumulate_gradient_covariance)
            for mod in modules
        )

        # loop over data set
        if self._generator is None:
            self._generator = Generator(device=self._device)
        self._generator.manual_seed(self._seed)

        for X, _ in self._loop_over_data(desc="Computing KFAC matrices"):
            output = self._model_func(X)

            for mc in range(self._mc_samples):
                y_sampled = self.draw_label(output)
                loss = self._loss_func(output, y_sampled)
                loss.backward(retain_graph=mc != self._mc_samples - 1)

        # remove hooks
        for handle in hook_handles:
            handle.remove()

    def draw_label(self, output: Tensor) -> Tensor:
        r"""Draw a sample from the model's predictive distribution.

        The model's distribution is implied by the (negative log likelihood) loss
        function. For instance, ``MSELoss`` implies a Gaussian distribution with
        constant variance, and ``CrossEntropyLoss`` implies a categorical distribution.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.

        Returns:
            A sample :math:`\{\mathbf{y}_n\}_{n=1}^N` drawn from the model's predictive
            distribution :math:`p(\mathbf{y} \mid \mathbf{x}, \mathbf{\theta})`. Has
            the same shape as the labels that would be fed into the loss function
            together with ``output``.
        """
        if isinstance(self._loss_func, MSELoss):
            std = {
                "sum": sqrt(1.0 / 2.0),
                "mean": sqrt(output.shape[1:].numel() / 2.0),
            }[self._loss_func.reduction]
            perturbation = std * randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator,
            )
            return output.clone().detach() + perturbation
        else:
            raise NotImplementedError

    def _hook_accumulate_gradient_covariance(
        self, module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
    ):
        """Backward hook that accumulates the output-gradient covariance of a layer.

        Updates ``self._gradient_covariances``.

        Args:
            module: The layer on which the hook is called.
            grad_input: The gradient of the loss w.r.t. the layer's inputs.
            grad_output: The gradient of the loss w.r.t. the layer's outputs.

        Raises:
            ValueError: If ``grad_output`` is not a 1-tuple.
            NotImplementedError: If a layer uses weight sharing.
            NotImplementedError: If the layer is not supported.
        """
        if len(grad_output) != 1:
            raise ValueError(
                f"Expected grad_output to be a 1-tuple, got {len(grad_output)}."
            )
        g = grad_output[0].data.detach()

        if isinstance(module, Linear):
            if g.ndim != 2:
                # TODO Support weight sharing
                raise NotImplementedError(
                    "Only 2d grad_outputs are supported for linear layers. "
                    + f"Got {g.ndim}d."
                )

            batch_size = g.shape[0]
            correction = {
                "sum": 1.0 / self._mc_samples,
                "mean": batch_size**2 / (self._N_data * self._mc_samples),
            }[self._loss_func.reduction]
            covariance = einsum("bi,bj->ij", g, g).mul_(correction)
        else:
            # TODO Support convolutions
            raise NotImplementedError(f"Layer of type {type(module)} is unsupported.")

        idx = tuple(p.data_ptr() for p in module.parameters())
        if idx not in self._gradient_covariances:
            self._gradient_covariances[idx] = covariance
        else:
            self._gradient_covariances[idx].add_(covariance)

    def _hook_accumulate_input_covariance(self, module: Module, inputs: Tuple[Tensor]):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Updates ``self._input_covariances``.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.

        Raises:
            ValueError: If the module has multiple inputs.
            NotImplementedError: If a layer uses weight sharing.
            NotImplementedError: If a module is not supported.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        x = inputs[0].data.detach()

        if isinstance(module, Linear):
            if x.ndim != 2:
                # TODO Support weight sharing
                raise NotImplementedError(
                    f"Only 2d inputs are supported for linear layers. Got {x.ndim}d."
                )

            covariance = einsum("bi,bj->ij", x, x).div_(self._N_data)
        else:
            # TODO Support convolutions
            raise NotImplementedError(f"Layer of type {type(module)} is unsupported.")

        idx = tuple(p.data_ptr() for p in module.parameters())
        if idx not in self._input_covariances:
            self._input_covariances[idx] = covariance
        else:
            self._input_covariances[idx].add_(covariance)
