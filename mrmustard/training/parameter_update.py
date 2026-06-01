# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Custom optax ``GradientTransformation``\ s for non-euclidean parameter updates."""

from collections.abc import Callable

import jax
import optax

from mrmustard import math

__all__ = [
    "riemannian_gradient",
    "riemannian_retraction",
    "siegel_retract",
    "siegel_retraction",
    "update_orthogonal",
    "update_siegel",
    "update_symplectic",
    "update_unitary",
]


def _hpsd_eigh(H):
    r"""Eigendecomposition of a batched Hermitian positive-semidefinite matrix
    with eigenvalues clipped to ``1e-12`` to absorb tiny numerical negatives
    and to keep negative powers well-defined.
    """
    evals, V = math.eigh(H)
    evals = math.maximum(math.real(evals), 1e-12)
    return evals, V


def _hpsd_power_from_eigh(evals, V, power: float):
    r"""Reassemble ``V diag(evals**power) V^H`` from a precomputed clipped
    eigendecomposition (see :func:`_hpsd_eigh`).
    """
    pow_evals = math.cast(evals**power, V.dtype)
    return math.einsum("...ij,...j,...kj->...ik", V, pow_evals, math.conj(V))


def _hpsd_power(H, power: float):
    r"""Compute ``H ** power`` for a batched Hermitian positive-semidefinite matrix
    via eigendecomposition.
    """
    evals, V = _hpsd_eigh(H)
    return _hpsd_power_from_eigh(evals, V, power)


def siegel_retract(Z, eta):
    r"""Bergman-geodesic retraction on the Siegel disk :math:`\mathcal{D}_g`.

    Follows the Bergman geodesic for unit time from :math:`Z` in direction ``eta``
    (complex symmetric) via three steps: pull ``eta`` back to the origin to obtain
    :math:`\xi_0 = A^{-1/2}\,\eta\,B^{-1/2}` with :math:`A = I - ZZ^*`,
    :math:`B = I - Z^*Z`; apply the origin geodesic
    :math:`V = U\,\mathrm{diag}(\tanh\sigma)\,U^T` where
    :math:`\xi_0 = U\,\mathrm{diag}(\sigma)\,U^T` is the Takagi decomposition;
    transvect the result back to :math:`Z` via the Möbius map

    .. math::
        \phi_Z(V) = A^{-1/2}\,(V + Z)\,(I + Z^* V)^{-1}\,B^{1/2}.

    The origin geodesic is computed without an explicit Takagi factorization via
    the identity :math:`U\,\tanh(S)\,U^T = h(\xi_0\xi_0^*)\,\xi_0` with
    :math:`h(x) = \tanh(\sqrt{x})/\sqrt{x}`. The Hermitian function
    :math:`h(\xi_0\xi_0^*)` is invariant under any unitary rotation within an
    eigenspace of :math:`\xi_0\xi_0^*`, so the result is correct even when the
    Takagi singular values of :math:`\xi_0` are degenerate.

    Args:
        Z: A point in :math:`\mathcal{D}_g` (complex symmetric).
        eta: A tangent vector at ``Z`` (complex symmetric).

    Returns:
        A complex symmetric matrix in :math:`\mathcal{D}_g`.
    """
    Z_H = math.conj(math.swapaxes(Z, -1, -2))
    eye = math.eye(Z.shape[-1], dtype=Z.dtype)
    A_neg_half = _hpsd_power(eye - math.matmul(Z, Z_H), -0.5)
    B_evals, B_V = _hpsd_eigh(eye - math.matmul(Z_H, Z))
    B_half = _hpsd_power_from_eigh(B_evals, B_V, 0.5)
    B_neg_half = _hpsd_power_from_eigh(B_evals, B_V, -0.5)
    xi0 = math.matmul(A_neg_half, math.matmul(eta, B_neg_half))
    xi0 = 0.5 * (xi0 + math.swapaxes(xi0, -1, -2))

    # Origin geodesic V = U tanh(S) U^T computed as h(xi0 xi0^*) @ xi0,
    # where h(x) = tanh(sqrt(x)) / sqrt(x). The eigenvectors of xi0 xi0^*
    # at eigenvalue 0 are orthogonal to range(xi0), so any finite value
    # assigned to h at sqrt(x) = 0 contributes exactly 0 to h_mat @ xi0;
    # we therefore clamp the denominator with a tiny constant rather than
    # branching on the Takagi limit.
    xi0_H = math.conj(math.swapaxes(xi0, -1, -2))
    sigma_sq, M_V = math.eigh(math.matmul(xi0, xi0_H))
    sigma = math.sqrt(math.maximum(math.real(sigma_sq), 0.0))
    # Clip tanh below 1 so the next iteration's (I - ZZ*)^{-1/2} stays
    # well-defined in float64 when the optimizer produces a very long step.
    tanh_sigma = math.minimum(math.tanh(sigma), 1.0 - 1e-7)
    h_diag = math.cast(tanh_sigma / math.maximum(sigma, 1e-30), M_V.dtype)
    h_mat = math.einsum("...ij,...j,...kj->...ik", M_V, h_diag, math.conj(M_V))
    V = math.matmul(h_mat, xi0)
    V = 0.5 * (V + math.swapaxes(V, -1, -2))

    right = math.matmul(math.inv(eye + math.matmul(Z_H, V)), B_half)
    out = math.matmul(math.matmul(A_neg_half, V + Z), right)
    return 0.5 * (out + math.swapaxes(out, -1, -2))


def riemannian_gradient(method: str):
    r"""Transforms Euclidean gradients to Riemannian gradients (Lie Algebra elements)."""

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(grads, state, params):
        if params is None:
            return grads, state

        if method == "symplectic":
            updates = jax.tree_util.tree_map(
                math.euclidean_to_symplectic,
                params,
                grads,
            )
        elif method == "unitary":
            updates = jax.tree_util.tree_map(
                math.euclidean_to_unitary,
                params,
                grads,
            )
        elif method == "orthogonal":
            updates = jax.tree_util.tree_map(
                lambda p, g: math.euclidean_to_unitary(p, math.real(g)),
                params,
                grads,
            )
        elif method == "siegel":
            updates = jax.tree_util.tree_map(
                math.euclidean_to_siegel,
                params,
                grads,
            )
        else:
            updates = grads

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _retraction(step_fn: Callable):
    r"""Builds an optax ``GradientTransformation`` that applies a per-leaf retraction.

    ``step_fn(p, u)`` should return the new point on the manifold given the current
    point ``p`` and the tangent-space step ``u``; this transformation converts it to
    the additive update ``p_new - p`` that optax composes downstream.
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        new_updates = jax.tree_util.tree_map(lambda p, u: step_fn(p, u) - p, params, updates)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def riemannian_retraction():
    r"""Lie-group retraction ``S_new = S @ expm(update)`` (used for U(N), O(N), Sp)."""
    return _retraction(lambda p, u: math.matmul(p, math.expm(u)))


def siegel_retraction():
    r"""Bergman-geodesic retraction on the Siegel disk."""
    return _retraction(siegel_retract)


def _riemannian_update(
    method: str,
    lr: float | Callable[[int], float],
    optimizer_cls: Callable,
    retraction: Callable[[], optax.GradientTransformation],
) -> optax.GradientTransformation:
    r"""Compose ``riemannian_gradient(method) → optimizer_cls(lr) → retraction()``."""
    return optax.chain(
        riemannian_gradient(method),
        optimizer_cls(learning_rate=lr),
        retraction(),
    )


def update_orthogonal(
    orthogonal_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""Creates an optax GradientTransformation for orthogonal parameter updates using Riemannian optimization.

    Args:
        orthogonal_lr: The learning rate for orthogonal updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for orthogonal updates.
    """
    return _riemannian_update("orthogonal", orthogonal_lr, optimizer_cls, riemannian_retraction)


def update_symplectic(
    symplectic_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""Creates an optax GradientTransformation for symplectic parameter updates using Riemannian optimization.

    Args:
        symplectic_lr: The learning rate for symplectic updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for symplectic updates.
    """
    return _riemannian_update("symplectic", symplectic_lr, optimizer_cls, riemannian_retraction)


def update_unitary(
    unitary_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""Creates an optax GradientTransformation for unitary parameter updates using Riemannian optimization.

    Args:
        unitary_lr: The learning rate for unitary updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for unitary updates.
    """
    return _riemannian_update("unitary", unitary_lr, optimizer_cls, riemannian_retraction)


def update_siegel(
    siegel_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""Creates an optax GradientTransformation for Siegel-disk parameter updates using
    Riemannian optimization with the Bergman metric.

    Args:
        siegel_lr: The learning rate for Siegel-disk updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for Siegel-disk updates.
    """
    return _riemannian_update("siegel", siegel_lr, optimizer_cls, siegel_retraction)
