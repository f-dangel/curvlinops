"""Hooks-based KFAC computer using forward/backward hooks.

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

from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator

from einops import einsum
from torch import Tensor, autograd
from torch.nn import Conv2d, Module
from torch.utils.hooks import RemovableHandle

from curvlinops.computers._base import (
    ParamGroup,
    ParamGroupKey,
    _BaseKFACComputer,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import _seed_generator


def _module_hyperparams(mod: Module) -> dict[str, Any]:
    """Extract KFAC-relevant hyperparameters from a module.

    Args:
        mod: A supported module (``Linear`` or ``Conv2d``).

    Returns:
        Empty dict for ``Linear``, convolution parameters for ``Conv2d``.
    """
    if isinstance(mod, Conv2d):
        return dict(
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
        )
    return {}


def _module_name_from_param(param_name: str) -> str:
    """Derive the module name from a fully qualified parameter name.

    Args:
        param_name: Full parameter name, e.g. ``"0.weight"`` or ``"weight"``.

    Returns:
        Module name, e.g. ``"0"`` or ``""`` (root module).
    """
    return param_name.rsplit(".", 1)[0] if "." in param_name else ""


@contextmanager
def _use_params(module: Module, params_dict: dict[str, Tensor]):
    """Temporarily replace a module's parameters with the given values.

    Restores the original parameter data after the context exits.
    Handles weight tying correctly since tied parameters share the same
    ``Parameter`` object — setting ``.data`` once affects all uses.

    Args:
        module: The module whose parameters to replace.
        params_dict: Dictionary mapping parameter names to replacement tensors.

    Yields:
        None.
    """
    originals = {}
    for name, param in module.named_parameters():
        if name in params_dict:
            originals[name] = param.data
            param.data = params_dict[name]
    try:
        yield
    finally:
        for name, param in module.named_parameters():
            if name in originals:
                param.data = originals[name]


class HooksKFACComputer(_BaseKFACComputer):
    r"""Computes KFAC's Kronecker factors using forward/backward hooks.

    KFAC approximates the per-layer Fisher/GGN with a Kronecker product:
    Consider a weight matrix :math:`\\mathbf{W}` and a bias vector :math:`\\mathbf{b}`
    in a single layer. The layer's Fisher :math:`\\mathbf{F}(\\mathbf{\\theta})` for

    .. math::
        \\mathbf{\\theta}
        =
        \\begin{pmatrix}
        \\mathrm{vec}(\\mathbf{W}) \\\\ \\mathbf{b}
        \\end{pmatrix}

    where :math:`\\mathrm{vec}` denotes column-stacking is approximated as

    .. math::
        \\mathbf{F}(\\mathbf{\\theta})
        \\approx
        \\mathbf{A}_{(\\text{KFAC})} \\otimes \\mathbf{B}_{(\\text{KFAC})}

    (see :class:`curvlinops.GGNLinearOperator` with ``mc_samples > 0``).
    """

    def _computation_context(self) -> Iterator[None]:
        """Set module parameters from ``self._params`` during computation.

        Returns:
            Context manager that temporarily replaces the module's parameters.
        """
        return _use_params(self._model_module, self._params)

    def _get_module(self, group: ParamGroup) -> Module:
        """Get the module corresponding to a parameter group.

        Derives the module name from the first parameter's full qualified name.

        Args:
            group: Parameter group (role → param name mapping).

        Returns:
            The module object.
        """
        mod_name = _module_name_from_param(next(iter(group.values())))
        return self._model_module.get_submodule(mod_name)

    def _compute_kronecker_factors(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        mapping = self.compute_parameter_groups(
            list(self._params.values()),
            self._model_module,
            self._separate_weight_and_bias,
        )

        # Create empty dictionaries to be populated by hooks
        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        # install forward and backward hooks
        hook_handles: list[RemovableHandle] = []

        for group in mapping:
            module = self._get_module(group)
            hparams = _module_hyperparams(module)

            fwd_hook = partial(
                self._hook_accumulate_input_covariance,
                group=group,
                layer_hyperparams=hparams,
                input_covariances=input_covariances,
            )
            bwd_hook = partial(
                self._register_tensor_hook_on_output_to_accumulate_gradient_covariance,
                group=group,
                layer_hyperparams=hparams,
                gradient_covariances=gradient_covariances,
            )

            if "W" in group:
                hook_handles.append(module.register_forward_pre_hook(fwd_hook))
            hook_handles.append(module.register_forward_hook(bwd_hook))

        # loop over data set, computing the Kronecker factors
        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_module(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y)

        # clean up
        for handle in hook_handles:
            handle.remove()

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances, mapping)

        return input_covariances, gradient_covariances, mapping

    def _compute_loss_and_backward(self, output: Tensor, y: Tensor):
        r"""Compute the loss and the backward pass(es) required for KFAC.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

        Raises:
            ValueError: If the output is not 2d and y is not 1d/2d.
        """
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported. "
                f"Got {output.ndim=} and {y.ndim=}."
            )

        # Compute the gradients w.r.t. the network's output that will be
        # backpropagated to compute the KFAC approximation.
        # Detach output: we only need values for the backpropagated vectors.
        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)

        # Fix scaling caused by the batch dimension
        num_loss_terms = output.shape[0]
        reduction = self._loss_func.reduction
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[reduction]
        grad_outputs.mul_(scale)

        # Backpropagate all vectors (0 for forward-only, 1 for empirical,
        # mc_samples for MC, and C (number of output features per datum) for TYPE2).
        num_vectors = grad_outputs.shape[0]
        for v in range(num_vectors):
            autograd.grad(
                output,
                list(self._params.values()),
                grad_outputs=grad_outputs[v],
                retain_graph=v < num_vectors - 1,
            )

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor],
        output: Tensor,
        group: ParamGroup,
        layer_hyperparams: dict[str, Any],
        gradient_covariances: dict[ParamGroupKey, Tensor],
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
            group: Parameter group for this layer.
            layer_hyperparams: Pre-computed hyperparameters for this layer.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        tensor_hook = partial(
            self._accumulate_gradient_covariance,
            group=group,
            layer_hyperparams=layer_hyperparams,
            gradient_covariances=gradient_covariances,
        )
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(
        self,
        grad_output: Tensor,
        group: ParamGroup,
        layer_hyperparams: dict[str, Any],
        gradient_covariances: dict[ParamGroupKey, Tensor],
    ):
        """Accumulate the gradient covariance for a layer's output.

        Updates the provided ``gradient_covariances`` dictionary.

        Args:
            grad_output: The gradient w.r.t. the output.
            group: Parameter group for this layer.
            layer_hyperparams: Pre-computed hyperparameters for this layer.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]

        g = grad_to_weight_sharing_format(
            g, self._kfac_approx, layer_hyperparams=layer_hyperparams
        )

        # Note: mc_samples scaling is already handled inside make_grad_output_fn.
        correction = compute_loss_correction(
            batch_size,
            self._num_per_example_loss_terms,
            self._loss_func.reduction,
            self._N_data,
        )

        covariance = einsum(g, g, "batch shared i, batch shared j -> i j").mul_(
            correction
        )
        self._set_or_add_(gradient_covariances, tuple(group.values()), covariance)

    def _hook_accumulate_input_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor, ...],
        group: ParamGroup,
        layer_hyperparams: dict[str, Any],
        input_covariances: dict[ParamGroupKey, Tensor],
    ):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Updates the provided ``input_covariances`` dictionary.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            group: Parameter group for this layer.
            layer_hyperparams: Pre-computed hyperparameters for this layer.
            input_covariances: Dictionary to store input covariances.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        x = inputs[0].data.detach()

        has_joint_wb = "b" in group and "W" in group

        x = input_to_weight_sharing_format(
            x,
            self._kfac_approx,
            layer_hyperparams=layer_hyperparams,
            bias_pad=1 if has_joint_wb else None,
        )
        scale = x.shape[1]
        covariance = einsum(x, x, "batch shared i, batch shared j -> i j").div_(
            self._N_data * scale
        )
        self._set_or_add_(input_covariances, tuple(group.values()), covariance)

    @classmethod
    def compute_parameter_groups(
        cls,
        params: list[Tensor],
        model_func: Module,
        separate_weight_and_bias: bool = True,
    ) -> list[ParamGroup]:
        """Construct parameter groups for the model's layers.

        Each supported module produces one group (joint treatment) or two
        groups (separate treatment). Joint treatment (``separate_weight_and_bias
        =False``) stores fewer Kronecker factors and is recommended for
        performance.

        Args:
            params: List of parameter tensors.
            model_func: The model function.
            separate_weight_and_bias: Whether to treat weight and bias as
                separate parameter groups.

        Returns:
            List of parameter groups (``dict[str, str]``), one per group.

        Raises:
            NotImplementedError: If parameters are found outside supported layers.
        """
        _role = {"weight": "W", "bias": "b"}

        param_ids = {p.data_ptr() for p in params}
        ptr_to_name = {
            p.data_ptr(): name
            for name, p in model_func.named_parameters()
            if p.data_ptr() in param_ids
        }
        groups: list[ParamGroup] = []
        processed = set()

        for _, mod in model_func.named_modules():
            if isinstance(mod, cls._SUPPORTED_MODULES) and any(
                p.data_ptr() in param_ids for p in mod.parameters()
            ):
                param_roles: ParamGroup = {}
                for p_name, p in mod.named_parameters(recurse=False):
                    p_id = p.data_ptr()
                    if p_id in param_ids:
                        param_roles[_role[p_name]] = ptr_to_name[p_id]
                        processed.add(p_id)
                param_dicts = (
                    [{r: n} for r, n in param_roles.items()]
                    if separate_weight_and_bias
                    else [param_roles]
                )
                groups.extend(param_dicts)

        # check that all parameters are in known modules
        if len(processed) != len(param_ids):
            raise NotImplementedError("Found parameters in un-supported layers.")

        return groups
