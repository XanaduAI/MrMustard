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

"""
Stellar representation computations.

Includes the stellar decomposition (factoring states into core states and
Gaussian operations/unitaries) and stellar root finding (computing the zeros
of the Bargmann polynomial from Fock amplitudes).
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial
from scipy.special import gammaln

from mrmustard import math
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

__all__ = [
    "formal_stellar_triples",
    "physical_stellar_triples_dm",
    "physical_stellar_triples_ket",
    "plot_stellar_roots",
    "stellar_roots",
]


def _unitary_normalization(Au: ComplexMatrix, bu: ComplexVector) -> ComplexTensor:
    """Compute U @ U.dual normalization via Gaussian integral."""
    M = Au.shape[-1] // 2
    idx_u = list(range(M))
    idx_dual = list(range(M, 2 * M))
    perm = idx_dual + idx_u
    Au_dual = math.conj(math.gather(math.gather(Au, perm, axis=-1), perm, axis=-2))
    bu_dual = math.conj(math.gather(bu, perm, axis=-1))
    _, _, log_c = math.complex_gaussian_integral_2(Au, bu, Au_dual, bu_dual, idx_u, idx_dual)
    return math.exp(log_c)


def formal_stellar_triples(
    triple: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    M: int,
) -> tuple[
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
]:
    """Returns the core and operator triples for the formal stellar decomposition.
    It decomposes any block Bargmann triple into two triples where the first has the core
    property on the first M indices and the second is an operator that acts on the first M indices
    to reconstruct the original triple.

    Args:
        triple: The Bargmann triple (A, b, c) of the original state.
        M: The number of core indices.

    Returns:
        core_triple: The Bargmann triple (A, b, c) of the core state.
        op_triple: The Bargmann triple (A, b, c) of the operator in out-in ordering.
    """
    A, b, c = triple
    batch_shape = A.shape[:-2]
    Am, An = A[..., :M, :M], A[..., M:, M:]
    R, R_T = A[..., M:, :M], A[..., :M, M:]
    bm, bn = b[..., :M], b[..., M:]

    Om = math.zeros_like(Am)
    A_core = math.block([[Om, R_T], [R, An]])
    b_core = math.concat([math.zeros_like(bm), bn], axis=-1)

    Im = math.eye_like(Am)
    if batch_shape:
        Im = math.broadcast_to(Im, (*batch_shape, M, M))
    A_Op = math.block([[Am, Im], [Im, Om]])  # in out-in ordering
    b_Op = math.concat([bm, math.zeros_like(bm)], axis=-1)

    return (A_core, b_core, c), (A_Op, b_Op, math.ones_like(c))


def physical_stellar_triples_ket(
    triple: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    M: int,
) -> tuple[
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
]:
    """Returns ``core`` and ``U`` from the physical stellar decomposition ``psi = core >> U``.
    Here ``core`` is a Gaussian ket on M+N modes with the core property on the first M modes,
    and ``U`` is an M-mode Gaussian unitary.

    Args:
        triple: The Bargmann triple (A, b, c) of the original state.
        M: The number of core modes.

    Returns:
        core_triple: The Bargmann triple (A, b, c) of the core state.
        U_triple: The Bargmann triple (A, b, c) of the unitary.
    """
    A, b, c = triple
    batch_shape = A.shape[:-2]
    Am, An = A[..., :M, :M], A[..., M:, M:]
    R, R_T = A[..., M:, :M], A[..., :M, M:]
    bm, bn = b[..., :M], b[..., M:]

    gamma_sq = math.eye(M, dtype=math.complex128) - Am @ math.conj(Am)
    evals, evecs = math.eigh(gamma_sq)
    gamma = math.einsum(
        "...ij,...j,...kj->...ik",
        evecs,
        math.sqrt(math.cast(evals, math.complex128)),
        math.conj(evecs),
    )
    gamma_T = math.swapaxes(gamma, -1, -2)
    gamma_inv = math.inv(gamma)
    gamma_inv_T = math.swapaxes(gamma_inv, -1, -2)

    # Cache repeated computations
    R_gamma_inv_T = R @ gamma_inv_T
    gamma_sq_inv = (
        gamma_inv @ gamma_inv
    )  # gamma_sq = gamma @ gamma, so inv is gamma_inv @ gamma_inv

    Au = math.block([[Am, gamma], [gamma_T, -math.conj(Am)]])
    bu_in = -math.einsum("...ij,...jk,...k->...i", math.conj(Am), gamma_inv_T, bm) - math.einsum(
        "...ij,...j->...i", gamma_inv, math.conj(bm)
    )
    bu = math.concat([bm, bu_in], -1)

    sqrt_renorm = math.sqrt(_unitary_normalization(Au, bu))
    c_u = math.ones(batch_shape, dtype=math.complex128) / sqrt_renorm

    A_core = math.block(
        [
            [math.zeros((*batch_shape, M, M), dtype=math.complex128), gamma_inv @ R_T],
            [R_gamma_inv_T, An + R @ math.conj(Am) @ gamma_sq_inv @ R_T],
        ]
    )
    b_core = math.concat(
        [math.zeros_like(bm), bn - math.einsum("...ij,...j->...i", R_gamma_inv_T, bu_in)], axis=-1
    )

    return (A_core, b_core, c * sqrt_renorm), (Au, bu, c_u)


def physical_stellar_triples_dm(
    triple: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    M: int,
) -> tuple[
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    tuple[ComplexMatrix, ComplexVector, ComplexTensor],
]:
    r"""Physical stellar decomposition rho = rho_core >> channel.
    It decomposes a triple parametrizing a DM into a Gaussian core DM and a Gaussian channel
    that acts on the core modes to reconstruct the original DM.
    This is always possible if M is at least half of the total number of modes.
    If M is fewer than half, the following rank condition must be satisfied:

    .. math::

        \mathrm{rank}(R^\top R + \sigma^\top \sigma) \leq M

    where

    .. math::

        A =
        \begin{bmatrix}
            A_m & R \\
            R^\top & A_n
        \end{bmatrix}

    Args:
        triple: The Bargmann triple (A, b, c) of the original state.
        M: The number of core modes.

    Returns:
        core_triple: The Bargmann triple (A, b, c) of the core state.
        phi_triple: The Bargmann triple (A, b, c) of the channel.

    Raises:
        ValueError: If the rank condition is not satisfied.
    """
    A, b, c = triple
    N = (A.shape[-1] - 2 * M) // 2
    batch_shape = A.shape[:-2]
    Am, An = A[..., : 2 * M, : 2 * M], A[..., 2 * M :, 2 * M :]
    R, R_T = A[..., 2 * M :, : 2 * M], A[..., : 2 * M, 2 * M :]
    bm, bn = b[..., : 2 * M], b[..., 2 * M :]

    sigma, r = R[..., N:, :M], R[..., N:, M:]
    alpha_m, alpha_n, a_n = Am[..., M:, :M], An[..., N:, :N], An[..., N:, N:]
    r_T, sigma_T, R_T = (math.swapaxes(x, -1, -2) for x in (r, sigma, R))

    rank = np.linalg.matrix_rank(r @ math.conj(r_T) + sigma @ math.conj(sigma_T))
    if math.any(rank > M):
        raise ValueError(
            f"Rank {rank} exceeds number of core modes {M}; physical stellar decomposition not possible. Try formal stellar decomposition instead."
        )

    # Cache repeated computations
    X_M = math.Xmat(M)
    alpha_m_inv = math.inv(alpha_m)
    sigma_alpha_m_inv_sigma_T = sigma @ alpha_m_inv @ math.conj(sigma_T)

    I2M = math.eye(2 * M, dtype=math.complex128)
    if batch_shape:
        I2M = math.broadcast_to(I2M, (*batch_shape, 2 * M, 2 * M))
    reduced_A = R @ math.inv(I2M - X_M @ Am) @ math.conj(R_T)
    r_c_sq = reduced_A[..., N:, N:] + sigma_alpha_m_inv_sigma_T
    evals, evecs = math.eigh(r_c_sq)

    if N >= M:
        r_c = math.einsum(
            "...ij,...j->...ij",
            evecs[..., -M:],
            math.sqrt(evals[..., -M:], dtype=math.complex128),
        )
    else:
        r_c_small = math.einsum(
            "...ij,...j->...ij",
            evecs,
            math.sqrt(evals, dtype=math.complex128),
        )
        padding = math.zeros((*batch_shape, N, M - N), dtype=math.complex128)
        r_c = math.concat([padding, r_c_small], axis=-1)

    Os_NM = math.zeros((*batch_shape, N, M), dtype=math.complex128)
    R_c = math.block([[math.conj(r_c), Os_NM], [Os_NM, r_c]])
    R_c_T = math.swapaxes(R_c, -1, -2)

    gamma = math.pinv(R_c) @ R
    gamma_T = math.swapaxes(gamma, -1, -2)
    Am_minus_X_inv = math.inv(Am - X_M)
    gamma_Am_minus_X_inv = gamma @ Am_minus_X_inv
    Aphi_in = gamma_Am_minus_X_inv @ gamma_T + X_M
    A_phi_out_in = math.block([[Am, gamma_T], [gamma, Aphi_in]])

    # Standard order for Map: (bra_out, bra_in, ket_out, ket_in)
    # A_phi_out_in indices: (bra_out, ket_out, bra_in, ket_in)
    idx_phi = math.concat(
        [
            math.arange(0, M),
            math.arange(2 * M, 3 * M),
            math.arange(M, 2 * M),
            math.arange(3 * M, 4 * M),
        ],
        axis=0,
    )
    A_phi = math.gather(math.gather(A_phi_out_in, idx_phi, axis=-1), idx_phi, axis=-2)

    # Displacements for phi and core
    b_phi_in = math.einsum("...ij,...j->...i", gamma_Am_minus_X_inv, bm)
    b_phi_out_in = math.concat([bm, b_phi_in], axis=-1)
    b_phi = math.gather(b_phi_out_in, idx_phi, axis=-1)

    alpha_core_n = alpha_n - sigma_alpha_m_inv_sigma_T
    a_core_n = a_n + reduced_A[..., N:, :N]
    A_core_n = math.block(
        [[math.conj(a_core_n), math.conj(alpha_core_n)], [alpha_core_n, a_core_n]]
    )
    A_core = math.block(
        [
            [math.zeros((*batch_shape, 2 * M, 2 * M), dtype=math.complex128), R_c_T],
            [R_c, A_core_n],
        ]
    )

    b_core_m = math.zeros((*batch_shape, 2 * M), dtype=b.dtype)
    b_core_n = bn - math.einsum("...ij,...j->...i", R_c, b_phi_in)
    b_core = math.concat([b_core_m, b_core_n], -1)

    # c_phi such that phi is trace-preserving
    c_phi = math.sqrt(math.det(I2M - Am @ X_M))
    # c_core such that rho = phi(rho_core)
    c_core = c / c_phi

    return (A_core, b_core, c_core), (A_phi, b_phi, c_phi)


def stellar_roots(fock_amplitudes: np.ndarray) -> np.ndarray:
    r"""Compute the stellar roots of a single-mode quantum state.

    The stellar roots are the zeros of the Bargmann polynomial

    .. math::

        F(z) = \sum_n  c_n z^n / \sqrt{n!}

    where :math:`c_{n}` are the Fock-basis amplitudes.

    Uses ``gammaln`` for numerical stability when normalizing by the factorial.

    Args:
        fock_amplitudes: 1D complex array of Fock coefficients ``c_n``.

    Returns:
        Complex 1D array of stellar roots (length ``len(fock_amplitudes) - 1``).
    """
    amplitudes = np.asarray(fock_amplitudes, dtype=np.complex128).ravel()
    if len(amplitudes) < 2:
        return np.array([], dtype=np.complex128)

    indices = np.arange(len(amplitudes))
    sqrt_factorial = np.exp(0.5 * gammaln(indices + 1))
    bargmann_coefficients = amplitudes / sqrt_factorial

    # numpy.polynomial.Polynomial expects coefficients in ascending order
    # (lowest degree first), which matches bargmann_coefficients
    poly = Polynomial(bargmann_coefficients)
    return poly.roots()


def plot_stellar_roots(
    roots: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    max_limit: float = 10.0,
    title: str = "Stellar Roots",
) -> plt.Axes:  # pragma: no cover
    """Plot stellar roots in the complex plane, colored by phase angle.

    Each root is colored according to ``np.angle(root)`` mapped through the
    HSV colormap, producing a natural hue wheel on the complex plane.

    When *ax* is ``None`` a new figure is created and ``plt.show()`` is
    called.  When an existing ``Axes`` is passed the roots are drawn on it
    and the caller is responsible for display (useful with the ipympl
    backend for flicker-free interactive updates).

    Args:
        roots: Complex array of stellar roots.
        ax: Optional matplotlib ``Axes`` to draw on.  If ``None``, a new
            figure is created.
        max_limit: Maximum absolute value for axis limits (default 10).
        title: Plot title.

    Returns:
        The ``Axes`` instance used for the plot.
    """
    roots = np.asarray(roots, dtype=np.complex128).ravel()

    # -- colors from phase angle --
    angles = np.angle(roots)
    hue_normalized = (angles + np.pi) / (2 * np.pi)  # [0, 1]
    colors = plt.cm.hsv(hue_normalized)

    # -- axis limits: include all roots with padding, capped at max_limit --
    if len(roots) > 0:
        extent = max(np.abs(roots.real).max(), np.abs(roots.imag).max())
        limit = min(extent * 1.15 + 0.3, max_limit)
    else:
        limit = 2.0

    # -- figure: create new or use existing axes --
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")

    ax.scatter(
        roots.real,
        roots.imag,
        c=colors,
        s=36,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )

    # -- reference lines through origin --
    ax.axhline(0, color="#bbbbbb", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#bbbbbb", linewidth=0.5, zorder=1)

    # -- sqrt(pi) grid --
    spacing = np.sqrt(np.pi)
    tick_vals = []
    v = spacing
    while v <= limit:
        tick_vals.extend([v, -v])
        v += spacing
    for v in tick_vals:
        ax.axhline(v, color="#eeeeee", linewidth=0.4, zorder=0)
        ax.axvline(v, color="#eeeeee", linewidth=0.4, zorder=0)

    # -- axis styling --
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)", fontsize=11)
    ax.set_ylabel("Im(z)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)

    ax.tick_params(labelsize=9, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#999999")

    if created_figure:
        fig.tight_layout()
        plt.show()

    return ax
