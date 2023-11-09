"""Linear operator for the Fisher/GGN's Kronecker-factored approximation.

Kronecker-factored approximate curvature was originally introduced for MLPs in

- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
  approximate curvature. International Conference on Machine Learning (ICML).

and extended to CNNs in

- Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
  convolution layers. International Conference on Machine Learning (ICML).
"""

from __future__ import annotations

from functools import partial
from math import sqrt
from typing import Dict, Iterable, List, Set, Tuple, Union

from einops import rearrange
from numpy import ndarray
from torch import Generator, Tensor, cat, einsum, randn, stack
from torch.nn import CrossEntropyLoss, Linear, Module, MSELoss, Parameter
from torch.utils.hooks import RemovableHandle

from curvlinops._base import _LinearOperator
from curvlinops.kfac_utils import loss_hessian_matrix_sqrt


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
        fisher_type: str = "mc",
        mc_samples: int = 1,
        separate_weight_and_bias: bool = True,
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

            - Only linear layers are supported.
            - No weight sharing is supported.
            - Only the ``'expand'`` approximation is supported.

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
            fisher_type: The type of Fisher/GGN to approximate. If 'type-2', the
                exact Hessian of the loss w.r.t. the model outputs is used. This
                requires as many backward passes as the output dimension, i.e.
                the number of classes for classification. This is sometimes also
                called type-2 Fisher. If ``'mc'``, the expectation is approximated
                by sampling ``mc_samples`` labels from the model's predictive
                distribution. If ``'empirical'``, the empirical gradients are
                used which corresponds to the uncentered gradient covariance, or
                the empirical Fisher. Defaults to ``'mc'``.
            mc_samples: The number of Monte-Carlo samples to use per data point.
                Has to be set to ``1`` when ``fisher_type != 'mc'``.
                Defaults to ``1``.
            separate_weight_and_bias: Whether to treat weights and biases separately.
                Defaults to ``True``.

        Raises:
            ValueError: If the loss function is not supported.
            NotImplementedError: If a parameter is in an unsupported layer.
        """
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )
        if fisher_type != "mc" and mc_samples != 1:
            raise ValueError(
                f"Invalid mc_samples: {mc_samples}. "
                "Only mc_samples=1 is supported for fisher_type != 'mc'."
            )

        self.param_ids = [p.data_ptr() for p in params]
        # mapping from tuples of parameter data pointers in a module to its name
        self.param_ids_to_hooked_modules: Dict[Tuple[int, ...], str] = {}

        hooked_param_ids: Set[int] = set()
        for name, mod in model_func.named_modules():
            p_ids = tuple(p.data_ptr() for p in mod.parameters())
            if isinstance(mod, self._SUPPORTED_MODULES) and any(
                p_id in self.param_ids for p_id in p_ids
            ):
                self.param_ids_to_hooked_modules[p_ids] = name
                hooked_param_ids.update(set(p_ids))

        # check that all parameters are in hooked modules
        if not set(self.param_ids).issubset(hooked_param_ids):
            raise NotImplementedError("Found parameters outside supported layers.")

        self._seed = seed
        self._generator: Union[None, Generator] = None
        self._separate_weight_and_bias = separate_weight_and_bias
        self._fisher_type = fisher_type
        self._mc_samples = mc_samples
        self._input_covariances: Dict[str, Tensor] = {}
        self._gradient_covariances: Dict[str, Tensor] = {}

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

        for name in self.param_ids_to_hooked_modules.values():
            mod = self._model_func.get_submodule(name)

            # bias and weights are treated jointly
            if not self._separate_weight_and_bias and self.in_params(
                mod.weight, mod.bias
            ):
                w_pos, b_pos = self.param_pos(mod.weight), self.param_pos(mod.bias)
                x_joint = cat([x_torch[w_pos], x_torch[b_pos].unsqueeze(-1)], dim=1)
                aaT = self._input_covariances[name]
                ggT = self._gradient_covariances[name]
                x_joint = ggT @ x_joint @ aaT

                w_cols = mod.weight.shape[1]
                x_torch[w_pos], x_torch[b_pos] = x_joint.split([w_cols, 1], dim=1)

            # for weights we need to multiply from the right with aaT
            # for weights and biases we need to multiply from the left with ggT
            else:
                for p_name in ["weight", "bias"]:
                    p = getattr(mod, p_name)
                    if self.in_params(p):
                        pos = self.param_pos(p)
                        x_torch[pos] = self._gradient_covariances[name] @ x_torch[pos]

                        if p_name == "weight":
                            x_torch[pos] = x_torch[pos] @ self._input_covariances[name]

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

        for name in self.param_ids_to_hooked_modules.values():
            module = self._model_func.get_submodule(name)

            # input covariance only required for weights
            if self.in_params(module.weight):
                hook_handles.append(
                    module.register_forward_pre_hook(
                        self._hook_accumulate_input_covariance
                    )
                )

            # gradient covariance required for weights and biases
            hook_handles.append(
                module.register_forward_hook(
                    self._register_tensor_hook_on_output_to_accumulate_gradient_covariance
                )
            )

        # loop over data set, computing the Kronecker factors
        if self._generator is None:
            self._generator = Generator(device=self._device)
        self._generator.manual_seed(self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_func(X)
            self._compute_loss_and_backward(output, y)

        # clean up
        self._model_func.zero_grad()
        for handle in hook_handles:
            handle.remove()

    def _compute_loss_and_backward(self, output: Tensor, y: Tensor):
        r"""Compute the loss and the backward pass(es) required for KFAC.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

        Raises:
            ValueError: If ``fisher_type`` is not ``'type-2'``, ``'mc'``, or
                ``'empirical'``.
            NotImplementedError: If ``fisher_type`` is ``'type-1'`` and the
                output is not 2d.
        """
        if self._fisher_type == "type-2":
            if output.ndim != 2:
                raise NotImplementedError(
                    "Type-2 Fisher not implemented for non-2d output."
                )
            # Compute per-sample Hessian square root, then concatenate over samples.
            # Result has shape `(batch_size, num_classes, num_classes)`
            hessian_sqrts = stack(
                [
                    loss_hessian_matrix_sqrt(out.detach(), self._loss_func)
                    for out in output.split(1)
                ]
            )

            # Fix scaling caused by the batch dimension
            batch_size = output.shape[0]
            reduction = self._loss_func.reduction
            scale = {"sum": 1.0, "mean": 1.0 / batch_size}[reduction]
            hessian_sqrts.mul_(scale)

            # For each column `c` of the matrix square root we need to backpropagate,
            # but we can do this for all samples in parallel
            num_cols = hessian_sqrts.shape[-1]
            for c in range(num_cols):
                batched_column = hessian_sqrts[:, :, c]
                (output * batched_column).sum().backward(retain_graph=c < num_cols - 1)

        elif self._fisher_type == "mc":
            for mc in range(self._mc_samples):
                y_sampled = self.draw_label(output)
                loss = self._loss_func(output, y_sampled)
                loss.backward(retain_graph=mc != self._mc_samples - 1)

        elif self._fisher_type == "empirical":
            loss = self._loss_func(output, y)
            loss.backward()

        else:
            raise ValueError(
                f"Invalid fisher_type: {self._fisher_type}. "
                + "Supported: 'type-2', 'mc', 'empirical'."
            )

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
            if output.ndim != 2:
                raise NotImplementedError(
                    "Only 2D output is supported for CrossEntropyLoss for now."
                )
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

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self, module: Module, inputs: Tuple[Tensor], output: Tensor
    ):
        """Register tensor hook on layer's output to accumulate the grad. covariance.

        Note:
            The easier way to compute the gradient covariance would be via a full
            backward hook on the module itself which performs the computation.
            However, this approach breaks down if the output of a layer feeds into an
            activation with `inplace=True` (see
            https://github.com/pytorch/pytorch/issues/61519). Hence we use the
            workaround
            https://github.com/pytorch/pytorch/issues/61519#issuecomment-883524237, and
            install a module hook which installs a tensor hook on the module's output
            tensor, which performs the accumulation of the gradient covariance.

        Args:
            module: Layer onto whose output a tensor hook to accumulate the gradient
                covariance will be installed.
            inputs: The layer's input tensors.
            output: The layer's output tensor.
        """
        tensor_hook = partial(self._accumulate_gradient_covariance, module)
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(self, module: Module, grad_output: Tensor):
        """Accumulate the gradient covariance for a layer's output.

        Updates ``self._gradient_covariances``.

        Args:
            module: The layer whose output's gradient covariance will be accumulated.
            grad_output: The gradient w.r.t. the output.

        Raises:
            NotImplementedError: If a layer uses weight sharing.
            NotImplementedError: If the layer is not supported.
        """
        g = grad_output.data.detach()

        if isinstance(module, Linear):
            if g.ndim != 2:
                # TODO Support weight sharing
                raise NotImplementedError(
                    "Only 2d grad_outputs are supported for linear layers. "
                    + f"Got {g.ndim}d."
                )

            batch_size = g.shape[0]
            # self._mc_samples will be 1 if fisher_type != "mc"
            correction = {
                "sum": 1.0 / self._mc_samples,
                "mean": batch_size**2 / (self._N_data * self._mc_samples),
            }[self._loss_func.reduction]
            covariance = einsum("bi,bj->ij", g, g).mul_(correction)
        else:
            # TODO Support convolutions
            raise NotImplementedError(
                f"Layer of type {type(module)} is unsupported. "
                + f"Supported layers: {self._SUPPORTED_MODULES}."
            )

        name = self.get_module_name(module)
        if name not in self._gradient_covariances:
            self._gradient_covariances[name] = covariance
        else:
            self._gradient_covariances[name].add_(covariance)

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

            if (
                self.in_params(module.weight, module.bias)
                and not self._separate_weight_and_bias
            ):
                x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

            covariance = einsum("bi,bj->ij", x, x).div_(self._N_data)
        else:
            # TODO Support convolutions
            raise NotImplementedError(f"Layer of type {type(module)} is unsupported.")

        name = self.get_module_name(module)
        if name not in self._input_covariances:
            self._input_covariances[name] = covariance
        else:
            self._input_covariances[name].add_(covariance)

    def get_module_name(self, module: Module) -> str:
        """Get the name of a module.

        Args:
            module: The module.

        Returns:
            The name of the module.
        """
        p_ids = tuple(p.data_ptr() for p in module.parameters())
        return self.param_ids_to_hooked_modules[p_ids]

    def in_params(self, *params: Union[Parameter, Tensor, None]) -> bool:
        """Check if all parameters are used in KFAC.

        Args:
            params: Parameters to check.

        Returns:
            Whether all parameters are used in KFAC.
        """
        return all(p is not None and p.data_ptr() in self.param_ids for p in params)

    def param_pos(self, param: Union[Parameter, Tensor]) -> int:
        """Get the position of a parameter in the list of parameters used in KFAC.

        Args:
            param: The parameter.

        Returns:
            The parameter's position in the parameter list.
        """
        return self.param_ids.index(param.data_ptr())
