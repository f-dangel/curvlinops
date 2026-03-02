r"""Inverses (natural gradient)
===============================

This example demonstrates how to work with inverses of linear operators.

:code:`curvlinops` offers multiple ways to compute the inverse of a linear operator:
conjugate gradient (CG) and Neumann inversion. We will demonstrate CG inversion
first and conclude with a comparison to Neumann inversion.

Concretely, we will compute the natural gradient :math:`\mathbf{\tilde{g}} =
\mathbf{F}^{-1} \mathbf{g}`,  defined by the inverse Fisher information
matrix :math:`\mathbf{F}^{-1}` and the gradient :math:`\mathbf{g}`. We can use
the GGN, as it corresponds to the Fisher for common loss functions like square
and cross-entropy loss.

.. note::
    The GGN is positive semi-definite, i.e. not full-rank. But we need a
    full-rank matrix to form the inverse. This is why we will add a damping term
    :math:`\delta \mathbf{I}` before inverting.

As always, let's first import the required functionality.
"""

import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from torch import Tensor, cat, cuda, device, eye, float64, manual_seed, rand, tensor
from torch.linalg import eigvalsh, inv, norm
from torch.nn import Linear, MSELoss, ReLU, Sequential, Sigmoid
from torch.nn.utils import parameters_to_vector

from curvlinops import (
    CGInverseLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.examples import IdentityLinearOperator, TensorLinearOperator, gradient_and_loss
from curvlinops.examples.functorch import (
    functorch_ggn,
    functorch_gradient_and_loss,
    functorch_hessian,
)
from curvlinops.utils import allclose_report

# make deterministic
manual_seed(0)


class DenseParameterSpaceLinearOperator(PyTorchLinearOperator):
    """Dense matrix linear operator acting on parameter-list shaped vectors."""

    def __init__(self, A: Tensor, param_shapes: list[tuple[int, ...]]):
        super().__init__(param_shapes, param_shapes)
        self._A = A
        self.SELF_ADJOINT = A.shape[0] == A.shape[1] and A.allclose(A.T)

    @property
    def device(self):
        return self._A.device

    @property
    def dtype(self):
        return self._A.dtype

    def _adjoint(self):
        return DenseParameterSpaceLinearOperator(self._A.conj().T, self._in_shape)

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        X_flat = cat([x.flatten(end_dim=-2) for x in X])
        _, num_vecs = X_flat.shape
        AX_flat = self._A @ X_flat
        return [
            block.reshape(*shape, num_vecs)
            for block, shape in zip(AX_flat.split(self._out_shape_flat), self._out_shape)
        ]

# %%
#
# Setup
# -----
#
# We will use synthetic data, consisting of two mini-batches, a small MLP, and
# mean-squared error as loss function.

N = 64
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = device("cuda" if cuda.is_available() else "cpu")
DTYPE = float64  # double precision for better stability when computing inverse

X1, y1 = rand(N, D_in).to(DEVICE, DTYPE), rand(N, D_out).to(DEVICE, DTYPE)
X2, y2 = rand(N, D_in).to(DEVICE, DTYPE), rand(N, D_out).to(DEVICE, DTYPE)

model = Sequential(
    Linear(D_in, D_hidden),
    ReLU(),
    Linear(D_hidden, D_hidden),
    Sigmoid(),
    Linear(D_hidden, D_out),
).to(DEVICE, DTYPE)
params = [p for p in model.parameters() if p.requires_grad]

loss_function = MSELoss(reduction="mean").to(DEVICE, DTYPE)


# %%
#
# Next, let's compute the ingredients for the natural gradient.
#
# Inverse GGN/Fisher
# ------------------
#
# First,  we set up a linear operator for the damped GGN/Fisher

data = [(X1, y1), (X2, y2)]
GGN = GGNLinearOperator(model, loss_function, params, data)
shapes = [p.shape for p in params]
delta = 1e-2
damping = delta * IdentityLinearOperator(shapes, GGN.device, DTYPE)
damped_GGN = GGN + damping

# %%
#
# and the linear operator of its inverse:

inverse_damped_GGN = CGInverseLinearOperator(
    damped_GGN,
    eps=0,  # do not add CG-internal damping
    # use a small number of iterations for a rough solution
    max_iter=5,
    max_tridiag_iter=5,
)

# %%
#
# Gradient
# --------
#
# We can obtain the gradient via a convenience function from :code:`curvlinops.examples`:

gradient, _ = gradient_and_loss(model, loss_function, params, data)
# flatten and concatenate
gradient = parameters_to_vector(gradient).detach()

# %%
#
# Natural gradient
# ----------------
#
# Now we have all components together to compute the natural gradient with a
# simple matrix-vector product:

natural_gradient = inverse_damped_GGN @ gradient

# %%
#
# As a first sanity check, let's compare if the natural gradient satisfies
# :math:`\mathbf{F} \mathbf{\tilde{g}} = \mathbf{g}`

approx_gradient = damped_GGN @ natural_gradient

print("Comparing gradient with Fisher @ natural gradient.")
assert allclose_report(approx_gradient, gradient, rtol=1e-4, atol=1e-5)

# %%
#
# Verifying results
# -----------------
#
# To check if the code works, let's compute the GGN with :code:`functorch`,
# using a utility function of :code:`curvlinops.examples`; then damp it, invert
# it, and multiply it onto the gradient.

GGN_mat_functorch = functorch_ggn(model, loss_function, params, data).detach()

# %%
#
#  then damp it and invert it.

damping_mat = delta * eye(GGN_mat_functorch.shape[0], device=DEVICE, dtype=DTYPE)
damped_GGN_mat = GGN_mat_functorch + damping_mat
inv_damped_GGN_mat = inv(damped_GGN_mat)


# %%
#
#  Next, let's compute the gradient with :code:`functorch`, using a utility
#  function from :code:`curvlinops.examples`:

gradient_functorch, _ = functorch_gradient_and_loss(model, loss_function, params, data)
# flatten and concatenate
gradient_functorch = parameters_to_vector(gradient_functorch).detach()

print("Comparing gradient with functorch's gradient.")
assert allclose_report(gradient, gradient_functorch)

# %%
#
#  We can now compute the natural gradient from the :code:`functorch`
#  quantities. This should yield approximately the same result:

natural_gradient_functorch = inv_damped_GGN_mat @ gradient_functorch

print("Comparing natural gradient with functorch's natural gradient.")
rtol, atol = 5e-3, 5e-5
assert allclose_report(
    natural_gradient, natural_gradient_functorch, rtol=rtol, atol=atol
)

# %%
#
#  You might have noticed the rather small tolerances required to achieve
#  approximate equality. We can use stricter convergence hyperparameters for CG
#  to achieve a more accurate inversion

inverse_damped_GGN = CGInverseLinearOperator(
    damped_GGN,
    eps=0,  # do not add CG-internal damping
    # increase number of iterations to get an better approximation
    max_iter=10,
    max_tridiag_iter=10,
)
natural_gradient_more_accurate = inverse_damped_GGN @ gradient

smaller_rtol, smaller_atol = rtol / 10, atol / 10
print("Comparing more accurate natural gradient with functorch's natural gradient.")
assert allclose_report(
    natural_gradient_more_accurate,
    natural_gradient_functorch,
    rtol=smaller_rtol,
    atol=smaller_atol,
)

# %%
#
#  whereas the less accurate inversion does not pass this check:

print(
    "Comparing natural gradient with functorch's natural gradient (smaller tolerances)."
)
try:
    assert allclose_report(
        natural_gradient,
        natural_gradient_functorch,
        rtol=smaller_rtol,
        atol=smaller_atol,
    )
    raise RuntimeError("This comparison should not pass")
except AssertionError as e:
    print(e)

# %%
#
# Visual comparison
# -----------------
#
# Finally, let's visualize the damped Fisher/GGN and its inverse. For improved
# visibility, we take the logarithm of the absolute value of each element
# (blank pixels correspond to zeros).

fig, ax = plt.subplots(ncols=2)
plt.suptitle("Logarithm of absolute values")

ax[0].set_title("Damped GGN/Fisher")
image = ax[0].imshow(damped_GGN_mat.detach().cpu().abs().log10())
plt.colorbar(image, ax=ax[0], shrink=0.5)

ax[1].set_title("Inv. damped GGN/Fisher")
image = ax[1].imshow(inv_damped_GGN_mat.detach().cpu().abs().log10())
plt.colorbar(image, ax=ax[1], shrink=0.5)

# %%
#
# Neumann inverse (CG alternative)
# --------------------------------
#
# So far, we used CG to solve the linear system :math:`\mathbf{F}
# \mathbf{\tilde{g}} = \mathbf{g}` for the natural gradient
# :math:`\mathbf{\tilde{g}}` (i.e. the result of the inverse Fisher-gradient
# product). Alternatively, we can use the truncated `Neumann series
# <https://en.wikipedia.org/wiki/Neumann_series>`_ to approximate the inverse,
# using :py:class:`NeumannLinearOperator`.
#
# .. note::
#     The Neumann series does not always converge. But we can use a re-scaling
#     trick to make it converge if we know the matrix is PSD and are given its
#     largest eigenvalue. More information can be found in the docstring.
#
# To make the Neumann series converge, we need to know the largest eigenvalue
# of the matrix to be inverted:
max_eigval = eigsh(damped_GGN.to_scipy(), k=1, which="LM", return_eigenvectors=False)[0]
# eigenvalues (scale * damped_GGN_mat) are in [0; 2)
scale = 1.0 if max_eigval < 2.0 else 1.99 / max_eigval

# %%
#
# Let's compute the inverse approximation for different truncation numbers:

num_terms = [10]
neumann_inverses = []

for n in num_terms:
    inv_neumann = NeumannInverseLinearOperator(damped_GGN, scale=scale, num_terms=n)
    neumann_inverses.append(
        inv_neumann @ eye(inv_neumann.shape[1], device=DEVICE, dtype=DTYPE)
    )

# %%
#
# Here are their visualizations:

fig, axes = plt.subplots(ncols=len(num_terms) + 1)
plt.suptitle("Inverse damped Fisher (logarithm of absolute values)")

for i, (n, inv_mat) in enumerate(zip(num_terms, neumann_inverses)):
    ax = axes.flat[i]
    ax.set_title(f"Neumann, {n} terms")
    image = ax.imshow(inv_mat.detach().cpu().abs().log10())
    plt.colorbar(image, ax=ax, shrink=0.5)

ax = axes.flat[-1]
ax.set_title("Exact inverse")
image = ax.imshow(inv_damped_GGN_mat.detach().cpu().abs().log10())
plt.colorbar(image, ax=ax, shrink=0.5)

# %%
#
# The Neumann inversion is usually more inaccurate than CG inversion. But it
# might sometimes be preferred if only a rough approximation of the inverse
# matrix product is needed.

# %%
#
# Small-scale matrix examples (exact error comparison)
# ----------------------------------------------------
#
# Before moving to preconditioned Hessian examples, let's compare CG and Neumann
# on a few tiny SPD matrices where we can form the exact inverse directly. We
# also test the ``preconditioner=...`` argument on these toy problems.
#
# For Neumann, note that passing ``preconditioner=...`` switches to a
# preconditioned Richardson/Neumann iteration and the ``scale`` argument is
# ignored. Therefore, ``preconditioner=Identity`` is generally *not* equivalent
# to the plain Neumann solver unless we explicitly use a scaled identity.

toy_matrices = {
    "3x3 SPD": tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    ),
    "4x4 ill-conditioned SPD": (
        tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            device=DEVICE,
            dtype=DTYPE,
        ).T
        @ tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            device=DEVICE,
            dtype=DTYPE,
        )
        + 1e-2 * eye(4, device=DEVICE, dtype=DTYPE)
    ),
}

toy_errors = {}
for name, A_toy in toy_matrices.items():
    A_toy_op = TensorLinearOperator(A_toy)
    identity_pre_toy = IdentityLinearOperator([(A_toy.shape[0],)], DEVICE, DTYPE)
    exact_pre_toy = TensorLinearOperator(inv(A_toy))
    rhs_toy = rand(A_toy.shape[0], 2, device=DEVICE, dtype=DTYPE)
    exact_toy = inv(A_toy) @ rhs_toy

    cg_toy = CGInverseLinearOperator(
        A_toy_op, eps=0, tolerance=0, max_iter=2, max_tridiag_iter=2
    )
    cg_identity_toy = CGInverseLinearOperator(
        A_toy_op,
        preconditioner=identity_pre_toy,
        eps=0,
        tolerance=0,
        max_iter=2,
        max_tridiag_iter=2,
    )
    cg_exact_pre_toy = CGInverseLinearOperator(
        A_toy_op,
        preconditioner=exact_pre_toy,
        eps=0,
        tolerance=0,
        max_iter=2,
        max_tridiag_iter=2,
    )
    cg_toy_sol = cg_toy @ rhs_toy
    cg_identity_toy_sol = cg_identity_toy @ rhs_toy
    cg_exact_pre_toy_sol = cg_exact_pre_toy @ rhs_toy

    max_eigval_toy = eigvalsh(A_toy).max().item()
    scale_toy = 1.0 if max_eigval_toy < 2.0 else 1.99 / max_eigval_toy
    neumann_toy = NeumannInverseLinearOperator(A_toy_op, num_terms=10, scale=scale_toy)
    neumann_identity_toy = NeumannInverseLinearOperator(
        A_toy_op, num_terms=10, preconditioner=identity_pre_toy, scale=scale_toy
    )
    neumann_exact_pre_toy = NeumannInverseLinearOperator(
        A_toy_op,
        num_terms=10,  # exact inverse preconditioner should solve in one term
        preconditioner=exact_pre_toy,
        scale=scale_toy,
    )
    neumann_toy_sol = neumann_toy @ rhs_toy
    neumann_identity_toy_sol = neumann_identity_toy @ rhs_toy
    neumann_exact_pre_toy_sol = neumann_exact_pre_toy @ rhs_toy

    toy_errors[name] = {
        "CG plain": (norm(cg_toy_sol - exact_toy) / norm(exact_toy)).item(),
        "CG identity pre": (
            norm(cg_identity_toy_sol - exact_toy) / norm(exact_toy)
        ).item(),
        "CG exact pre": (norm(cg_exact_pre_toy_sol - exact_toy) / norm(exact_toy)).item(),
        "Neumann plain": (norm(neumann_toy_sol - exact_toy) / norm(exact_toy)).item(),
        "Neumann identity pre": (
            norm(neumann_identity_toy_sol - exact_toy) / norm(exact_toy)
        ).item(),
        "Neumann exact pre": (
            norm(neumann_exact_pre_toy_sol - exact_toy) / norm(exact_toy)
        ).item(),
    }

print("Small-matrix relative errors vs exact inverse action:")
for name, vals in toy_errors.items():
    print(f"  {name}")
    print(
        "    "
        + f"CG plain={vals['CG plain']:.3e}, "
        + f"CG identity-pre={vals['CG identity pre']:.3e}, "
        + f"CG exact-pre={vals['CG exact pre']:.3e}"
    )
    print(
        "    "
        + f"Neumann plain={vals['Neumann plain']:.3e}, "
        + f"Neumann identity-pre={vals['Neumann identity pre']:.3e}, "
        + f"Neumann exact-pre={vals['Neumann exact pre']:.3e}"
    )

# %%
#
# Preconditioning in an over-parameterized regime
# -----------------------------------------------
#
# Next, we demonstrate how to use the new ``preconditioner=...`` argument for
# CG and Neumann inversion.
#
# We build an over-parameterized linear regression problem (many parameters, few
# data points), form the Hessian, and compare inverse-Hessian-vector products
# against a dense ground truth. We use KFAC and EKFAC inverse operators as
# preconditioners.
#
# A linear model with mean-squared error has a PSD Hessian and, in this setup,
# Hessian and GGN coincide. This makes KFAC/EKFAC natural preconditioners.

N_small = 2000
D_in_small = 20
D_out_small = 10

X_small = rand(N_small, D_in_small, device=DEVICE, dtype=DTYPE)
y_small = rand(N_small, D_out_small, device=DEVICE, dtype=DTYPE)
small_data = [(X_small, y_small)]

small_model = Linear(D_in_small, D_out_small).to(DEVICE, DTYPE)
small_params = [p for p in small_model.parameters() if p.requires_grad]
small_loss = MSELoss(reduction="mean").to(DEVICE, DTYPE)

num_params = sum(p.numel() for p in small_params)
print(
    "Over-parameterized setup:",
    f"{num_params} parameters for {N_small} data points.",
)

H = HessianLinearOperator(small_model, small_loss, small_params, small_data)
H_mat = functorch_hessian(small_model, small_loss, small_params, small_data).detach()

delta_h = 1e-3 * H_mat.diag().mean().item()
H_damped = H + delta_h * IdentityLinearOperator(
    [p.shape for p in small_params], H.device, DTYPE
)
H_damped_mat = H_mat + delta_h * eye(H_mat.shape[0], device=DEVICE, dtype=DTYPE)
H_damped_inv_mat = inv(H_damped_mat)

rhs = rand(H.shape[1], 3, device=DEVICE, dtype=DTYPE)
exact_solution = H_damped_inv_mat @ rhs

# Build preconditioners (inverse linear operators in parameter space).
identity_preconditioner = IdentityLinearOperator(
    [p.shape for p in small_params], H.device, DTYPE
)
exact_hessian_preconditioner = DenseParameterSpaceLinearOperator(
    H_damped_inv_mat, [p.shape for p in small_params]
)
kfac = KFACLinearOperator(small_model, small_loss, small_params, small_data)
ekfac = EKFACLinearOperator(small_model, small_loss, small_params, small_data)
kfac_preconditioner = kfac.inverse(damping=1e-3, use_heuristic_damping=True)
ekfac_preconditioner = ekfac.inverse(damping=1e-3)

# CG with and without preconditioning (few iterations on purpose).
cg_plain = CGInverseLinearOperator(
    H_damped, eps=1e-3, tolerance=0, max_iter=3, max_tridiag_iter=3
)
cg_identity = CGInverseLinearOperator(
    H_damped,
    preconditioner=identity_preconditioner,
    eps=1e-3,
    tolerance=0,
    max_iter=3,
    max_tridiag_iter=3,
)
cg_exact_hessian = CGInverseLinearOperator(
    H_damped,
    preconditioner=exact_hessian_preconditioner,
    eps=1e-3,
    tolerance=0,
    max_iter=3,
    max_tridiag_iter=3,
)
cg_kfac = CGInverseLinearOperator(
    H_damped,
    preconditioner=kfac_preconditioner,
    eps=1e-3,
    tolerance=0,
    max_iter=3,
    max_tridiag_iter=3,
)
cg_ekfac = CGInverseLinearOperator(
    H_damped,
    preconditioner=ekfac_preconditioner,
    eps=1e-3,
    tolerance=0,
    max_iter=3,
    max_tridiag_iter=3,
)

cg_plain_sol = cg_plain @ rhs
cg_identity_sol = cg_identity @ rhs
cg_exact_hessian_sol = cg_exact_hessian @ rhs
cg_kfac_sol = cg_kfac @ rhs
cg_ekfac_sol = cg_ekfac @ rhs

# Neumann with and without preconditioning (few terms on purpose).
max_eigval_h = eigvalsh(H_damped_mat).max().item()
neumann_scale = 1.0 if max_eigval_h < 2.0 else 1.99 / max_eigval_h

neumann_plain = NeumannInverseLinearOperator(
    H_damped, num_terms=10, scale=neumann_scale
)
neumann_identity = NeumannInverseLinearOperator(
    H_damped, num_terms=10, preconditioner=identity_preconditioner, scale=neumann_scale
)
neumann_exact_hessian = NeumannInverseLinearOperator(
    H_damped,
    num_terms=0,
    preconditioner=exact_hessian_preconditioner,
    scale=1.0,
)
neumann_kfac = NeumannInverseLinearOperator(
    H_damped, num_terms=10, preconditioner=kfac_preconditioner, scale=neumann_scale
)
neumann_ekfac = NeumannInverseLinearOperator(
    H_damped, num_terms=10, preconditioner=ekfac_preconditioner, scale=neumann_scale
)

neumann_plain_sol = neumann_plain @ rhs
neumann_identity_sol = neumann_identity @ rhs
neumann_exact_hessian_sol = neumann_exact_hessian @ rhs
neumann_kfac_sol = neumann_kfac @ rhs
neumann_ekfac_sol = neumann_ekfac @ rhs


def relative_inverse_error(approx, exact):
    """Relative error of inverse-operator applications."""
    return (norm(approx - exact) / norm(exact)).item()


errors = {
    "CG": {
        "plain": relative_inverse_error(cg_plain_sol, exact_solution),
        "Identity pre": relative_inverse_error(cg_identity_sol, exact_solution),
        "Exact-H pre": relative_inverse_error(cg_exact_hessian_sol, exact_solution),
        "KFAC pre": relative_inverse_error(cg_kfac_sol, exact_solution),
        "EKFAC pre": relative_inverse_error(cg_ekfac_sol, exact_solution),
    },
    "Neumann": {
        "plain": relative_inverse_error(neumann_plain_sol, exact_solution),
        "Identity pre": relative_inverse_error(neumann_identity_sol, exact_solution),
        "Exact-H pre": relative_inverse_error(
            neumann_exact_hessian_sol, exact_solution
        ),
        "KFAC pre": relative_inverse_error(neumann_kfac_sol, exact_solution),
        "EKFAC pre": relative_inverse_error(neumann_ekfac_sol, exact_solution),
    },
}

print("Relative error to exact dense inverse-Hessian action (lower is better):")
for method, vals in errors.items():
    print(
        f"  {method:7s} plain={vals['plain']:.3e}  "
        f"I-pre={vals['Identity pre']:.3e}  "
        f"ExactH-pre={vals['Exact-H pre']:.3e}  "
        f"KFAC-pre={vals['KFAC pre']:.3e}  EKFAC-pre={vals['EKFAC pre']:.3e}"
    )
