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

from einops import rearrange
from numpy import ndarray
from torch import Generator, Tensor, einsum, randn
from torch.nn import CrossEntropyLoss, Linear, Module, MSELoss, Parameter
from torch.utils.hooks import RemovableHandle

from curvlinops._base import _LinearOperator


class KFACLinearOperator(_LinearOperator):
    r"""Linear operator to multiply with the Fisher/GGN's KFAC approximation.

    KFAC approximates the per-layer Fisher/GGN with a Kronecker product:
    Consider a weight matrix :math:`\mathbf{W}` and a bias vector :math:`\mathbf{b}`
    in a single layer. The layer's Fisher :math:`\mathbf{F}(\mathbf{\theta})` for

    .. math::
        \mathbf{\theta}
        =
        \begin{pmatrix}
        \mathrm{vec}(\mathbf{W}) \\ \mathbf{b}
        \end{pmatrix}

    where :math:`\mathrm{vec}` denotes column-stacking is approximated as

    .. math::
        \mathbf{F}(\mathbf{\theta})
        \approx
        \mathbf{A}_{(\text{KFAC})} \otimes \mathbf{B}_{(\text{KFAC})}

    (see :class:`curvlinops.FisherMCLinearOperator` for the Fisher's definition).
    Loosely speaking, the first Kronecker factor is the un-centered covariance of the
    inputs to a layer. The second Kronecker factor is the un-centered covariance of
    'would-be' gradients w.r.t. the layer's output. Those 'would-be' gradients result
    from sampling labels from the model's distribution and computing their gradients.

    The basic version of KFAC for MLPs was introduced in

    - Martens, J., & Grosse, R. (2015). Optimizing neural networks with
      Kronecker-factored approximate curvature. ICML.

    and later generalized to convolutions in

    - Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher
      matrix for convolution layers. ICML.

    Attributes:
        _SUPPORTED_LOSSES: Tuple of supported loss functions.
        _SUPPORTED_MODULES: Tuple of supported layers.
    """

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss)
    _SUPPORTED_MODULES = (Linear,)

    def __init__(
        self,
        model_func: Module,
        loss_func: MSELoss,
        params: List[Parameter],
        data: Iterable[Tuple[Tensor, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        shape: Union[Tuple[int, int], None] = None,
        seed: int = 2147483647,
        mc_samples: int = 1,
    ):
        """Kronecker-factored approximate curvature (KFAC) proxy of the Fisher/GGN.

        Warning:
            If the model's parameters change, e.g. during training, you need to
            create a fresh instance of this object. This is because, for performance
            reasons, the Kronecker factors are computed once and cached during the
            first matrix-vector product. They will thus become outdated if the model
            changes.

        Warning:
            This is an early proto-type with many limitations:

            - Parameters must be in the same order as the model's parameters.
            - Only linear layers with bias are supported.
            - Weights and biases are treated separately.
            - No weight sharing is supported.
            - Only the Monte-Carlo sampled version is supported.
            - Only the ``'expand'`` setting is supported.

        Args:
            model_func: The neural network. Must consist of modules.
            loss_func: The loss function.
            params: The parameters defining the Fisher/GGN that will be approximated
                through KFAC.
            data: A data loader containing the data of the Fisher/GGN.
            progressbar: Whether to show a progress bar when computing the Kronecker
                factors. Defaults to ``False``.
            check_deterministic: Whether to check that the linear operator is
                deterministic. Defaults to ``True``.
            shape: The shape of the linear operator. If ``None``, it will be inferred
                from the parameters. Defaults to ``None``.
            seed: The seed for the random number generator used to draw labels
                from the model's predictive distribution. Defaults to ``2147483647``.
            mc_samples: The number of Monte-Carlo samples to use per data point.
                Defaults to ``1``.

        Raises:
            ValueError: If the loss function is not supported.
            NotImplementedError: If the parameters are not in the same order as the
                model's parameters.
            NotImplementedError: If the model contains bias-free linear layers.
            NotImplementedError: If any parameter cannot be identified with a supported
                layer.
        """
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )

        self.hooked_modules: List[str] = []
        idx = 0
        for name, mod in model_func.named_modules():
            if isinstance(mod, self._SUPPORTED_MODULES):
                # TODO Support bias-free layers
                if mod.bias is None:
                    raise NotImplementedError(
                        "Bias-free linear layers are not yet supported."
                    )
                # TODO Support arbitrary orders and sub-sets of parameters
                if (
                    params[idx].data_ptr() != mod.weight.data_ptr()
                    or params[idx + 1].data_ptr() != mod.bias.data_ptr()
                ):
                    raise NotImplementedError(
                        "KFAC parameters must be in same order as model parameters "
                        + "for now."
                    )
                idx += 2
                self.hooked_modules.append(name)
        if idx != len(params):
            raise NotImplementedError(
                "Could not identify all parameters with supported layers."
            )

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
        """Compute and cache KFAC's Kronecker factors for future ``matvec``s."""
        # install forward and backward hooks
        hook_handles: List[RemovableHandle] = []
        hook_handles.extend(
            self._model_func.get_submodule(mod).register_forward_pre_hook(
                self._hook_accumulate_input_covariance
            )
            for mod in self.hooked_modules
        )
        hook_handles.extend(
            self._model_func.get_submodule(mod).register_full_backward_hook(
                self._hook_accumulate_gradient_covariance
            )
            for mod in self.hooked_modules
        )

        # loop over data set, computing the Kronecker factors
        if self._generator is None:
            self._generator = Generator(device=self._device)
        self._generator.manual_seed(self._seed)

        for X, _ in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_func(X)

            for mc in range(self._mc_samples):
                y_sampled = self.draw_label(output)
                loss = self._loss_func(output, y_sampled)
                loss.backward(retain_graph=mc != self._mc_samples - 1)

        # clean up
        self._model_func.zero_grad()
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
            A sample
            :math:`\{\mathbf{y}_n\}_{n=1}^N` drawn from the model's predictive
            distribution :math:`p(\mathbf{y} \mid \mathbf{x}, \mathbf{\theta})`. Has
            the same shape as the labels that would be fed into the loss function
            together with ``output``.

        Raises:
            NotImplementedError: If the loss function is not supported.
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
        elif isinstance(self._loss_func, CrossEntropyLoss):
            # TODO For output.ndim > 2, the scale of the 'would-be' gradient resulting
            # from these labels might be off
            probs = output.softmax(dim=1)
            # each row contains a vector describing a categorical
            probs_as_mat = rearrange(probs, "n c ... -> (n ...) c")
            labels = probs_as_mat.multinomial(
                num_samples=1, generator=self._generator
            ).squeeze(-1)
            label_shape = output.shape[:1] + output.shape[2:]
            return labels.reshape(label_shape)
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
