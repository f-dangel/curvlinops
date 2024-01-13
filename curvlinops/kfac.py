"""Linear operator for the Fisher/GGN's Kronecker-factored approximation.

Kronecker-Factored Approximate Curvature (KFAC) was originally introduced for MLPs in

- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
  approximate curvature. International Conference on Machine Learning (ICML),

extended to CNNs in

- Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
  convolution layers. International Conference on Machine Learning (ICML),

and generalized to all linear layers with weight sharing in

- Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., Hennig, P. (2023).
  Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures (NeurIPS).
"""

from __future__ import annotations

from functools import partial
from math import sqrt
from typing import Dict, Iterable, List, Set, Tuple, Union

from einops import rearrange, reduce
from numpy import ndarray
from torch import Generator, Tensor, cat, einsum, randn, stack
from torch.nn import Conv2d, CrossEntropyLoss, Linear, Module, MSELoss, Parameter
from torch.utils.hooks import RemovableHandle

from curvlinops._base import _LinearOperator
from curvlinops.kfac_utils import (
    extract_averaged_patches,
    extract_patches,
    loss_hessian_matrix_sqrt,
)


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

    Kronecker-Factored Approximate Curvature (KFAC) was originally introduced for MLPs in

    - Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
    approximate curvature. International Conference on Machine Learning (ICML),

    extended to CNNs in

    - Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
    convolution layers. International Conference on Machine Learning (ICML),

    and generalized to all linear layers with weight sharing in

    - Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., Hennig, P. (2023).
    Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures (NeurIPS).

    Attributes:
        _SUPPORTED_LOSSES: Tuple of supported loss functions.
        _SUPPORTED_MODULES: Tuple of supported layers.
    """

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss)
    _SUPPORTED_MODULES = (Linear, Conv2d)
    _SUPPORTED_LOSS_AVERAGE: Tuple[Union[None, str], ...] = (
        None,
        "batch",
        "batch+sequence",
    )
    _SUPPORTED_KFAC_APPROX: Tuple[str, ...] = ("expand", "reduce")

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
        kfac_approx: str = "expand",
        loss_average: Union[None, str] = "batch",
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
            This is an early proto-type with limitations:
                - Only Linear and Conv2d modules are supported.

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
            kfac_approx: A string specifying the KFAC approximation that should
                be used for linear weight-sharing layers, e.g. ``Conv2d`` modules
                or ``Linear`` modules that process matrix- or higher-dimensional
                features.
                Possible values are ``'expand'`` and ``'reduce'``.
                See `Eschenhagen et al., 2023 <https://arxiv.org/abs/2311.00636>`_
                for an explanation of the two approximations.
            loss_average: Whether the loss function is a mean over per-sample
                losses and if yes, over which dimensions the mean is taken.
                If ``"batch"``, the loss function is a mean over as many terms as
                the size of the mini-batch. If ``"batch+sequence"``, the loss
                function is a mean over as many terms as the size of the
                mini-batch times the sequence length, e.g. in the case of
                language modeling. If ``None``, the loss function is a sum. This
                argument is used to ensure that the preconditioner is scaled
                consistently with the loss and the gradient. Default: ``"batch"``.
            separate_weight_and_bias: Whether to treat weights and biases separately.
                Defaults to ``True``.

        Raises:
            ValueError: If the loss function is not supported.
            ValueError: If the loss average is not supported.
            ValueError: If the loss average is ``None`` and the loss function's
                reduction is not ``'sum'``.
            ValueError: If the loss average is not ``None`` and the loss function's
                reduction is ``'sum'``.
            ValueError: If ``fisher_type != 'mc'`` and ``mc_samples != 1``.
            NotImplementedError: If a parameter is in an unsupported layer.
        """
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )
        if loss_average not in self._SUPPORTED_LOSS_AVERAGE:
            raise ValueError(
                f"Invalid loss_average: {loss_average}. "
                f"Supported: {self._SUPPORTED_LOSS_AVERAGE}."
            )
        if loss_average is None and loss_func.reduction != "sum":
            raise ValueError(
                f"Invalid loss_average: {loss_average}. "
                f"Must be 'batch' or 'batch+sequence' if loss_func.reduction != 'sum'."
            )
        if loss_func.reduction == "sum" and loss_average is not None:
            raise ValueError(
                f"Loss function uses reduction='sum', but loss_average={loss_average}."
                " Set loss_average to None if you want to use reduction='sum'."
            )
        if fisher_type != "mc" and mc_samples != 1:
            raise ValueError(
                f"Invalid mc_samples: {mc_samples}. "
                "Only mc_samples=1 is supported for fisher_type != 'mc'."
            )
        if kfac_approx not in self._SUPPORTED_KFAC_APPROX:
            raise ValueError(
                f"Invalid kfac_approx: {kfac_approx}. "
                f"Supported: {self._SUPPORTED_KFAC_APPROX}."
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
        self._kfac_approx = kfac_approx
        self._loss_average = loss_average
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
                x_w = rearrange(x_torch[w_pos], "c_out ... -> c_out (...)")
                x_joint = cat([x_w, x_torch[b_pos].unsqueeze(-1)], dim=1)
                aaT = self._input_covariances[name]
                ggT = self._gradient_covariances[name]
                x_joint = ggT @ x_joint @ aaT

                w_cols = x_w.shape[1]
                x_torch[w_pos], x_torch[b_pos] = x_joint.split([w_cols, 1], dim=1)

            # for weights we need to multiply from the right with aaT
            # for weights and biases we need to multiply from the left with ggT
            else:
                for p_name in ["weight", "bias"]:
                    p = getattr(mod, p_name)
                    if self.in_params(p):
                        pos = self.param_pos(p)

                        if p_name == "weight":
                            x_w = rearrange(x_torch[pos], "c_out ... -> c_out (...)")
                            x_torch[pos] = x_w @ self._input_covariances[name]

                        x_torch[pos] = self._gradient_covariances[name] @ x_torch[pos]

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
        """
        # if >2d output we convert to an equivalent 2d output
        if isinstance(self._loss_func, CrossEntropyLoss):
            output = rearrange(output, "batch c ... -> (batch ...) c")
            y = rearrange(y, "batch ... -> (batch ...)")
        else:
            output = rearrange(output, "batch ... c -> (batch ...) c")
            y = rearrange(y, "batch ... c -> (batch ...) c")

        if self._fisher_type == "type-2":
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
            ValueError: If the output is not 2d.
            NotImplementedError: If the loss function is not supported.
        """
        if output.ndim != 2:
            raise ValueError("Only a 2d output is supported.")

        if isinstance(self._loss_func, MSELoss):
            std = {
                "sum": sqrt(1.0 / 2.0),
                "mean": sqrt(output.shape[1] / 2.0),
            }[self._loss_func.reduction]
            perturbation = std * randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator,
            )
            return output.clone().detach() + perturbation

        elif isinstance(self._loss_func, CrossEntropyLoss):
            probs = output.softmax(dim=1)
            labels = probs.multinomial(
                num_samples=1, generator=self._generator
            ).squeeze(-1)
            return labels

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
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]
        if isinstance(module, Conv2d):
            g = rearrange(g, "batch c o1 o2 -> batch o1 o2 c")
        sequence_length = g.shape[1:-1].numel()
        num_loss_terms = {
            None: batch_size,
            "batch": batch_size,
            "batch+sequence": batch_size * sequence_length,
        }[self._loss_average]

        if self._kfac_approx == "expand":
            # KFAC-expand approximation
            g = rearrange(g, "batch ... d_out -> (batch ...) d_out")
        else:
            # KFAC-reduce approximation
            g = reduce(g, "batch ... d_out -> batch d_out", "sum")

        # self._mc_samples will be 1 if fisher_type != "mc"
        correction = {
            None: 1.0 / self._mc_samples,
            "batch": num_loss_terms**2 / (self._N_data * self._mc_samples),
            "batch+sequence": num_loss_terms**2
            / (self._N_data * self._mc_samples * sequence_length),
        }[self._loss_average]
        covariance = einsum("bi,bj->ij", g, g).mul_(correction)

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
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        x = inputs[0].data.detach()

        if isinstance(module, Conv2d):
            patch_extractor_fn = {
                "expand": extract_patches,
                "reduce": extract_averaged_patches,
            }[self._kfac_approx]
            x = patch_extractor_fn(
                x,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )

        if self._kfac_approx == "expand":
            # KFAC-expand approximation
            scale = x.shape[1:-1].numel()  # sequence_length
            x = rearrange(x, "batch ... d_in -> (batch ...) d_in")
        else:
            # KFAC-reduce approximation
            scale = 1.0  # since we use a mean reduction
            x = reduce(x, "batch ... d_in -> batch d_in", "mean")

        if (
            self.in_params(module.weight, module.bias)
            and not self._separate_weight_and_bias
        ):
            x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

        covariance = einsum("bi,bj->ij", x, x).div_(self._N_data * scale)

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
