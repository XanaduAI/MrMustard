# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=redefined-outer-name

"""
This module contains functions for performing calculations on objects in the Fock representations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Iterable

import numpy as np
from scipy.special import comb, factorial

from mrmustard import math, settings
from mrmustard.math.lattice import strategies
from mrmustard.math.caching import tensor_int_cache

from mrmustard.utils.typing import Scalar, Tensor, Vector, Batch

SQRT = np.sqrt(np.arange(1e6))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fock_state(n: int | Sequence[int], cutoffs: int | Sequence[int] | None = None) -> Tensor:
    r"""
    The Fock array of a tensor product of one-mode ``Number`` states.

    Args:
        n: The photon numbers of the number states.
        cutoffs: The cutoffs of the arrays for the number states. If it is given as
            an ``int``, it is broadcasted to all the states. If ``None``, it
            defaults to ``[n1+1, n2+1, ...]``, where ``ni`` is the photon number
            of the ``i``th mode.

    Returns:
        The Fock array of a tensor product of one-mode ``Number`` states.
    """
    n = math.atleast_1d(n)

    if cutoffs is None:
        cutoffs = list(n)
    elif isinstance(cutoffs, int):
        cutoffs = [cutoffs] * len(n)
    else:
        cutoffs = math.atleast_1d(cutoffs)

    if len(cutoffs) != len(n):
        msg = f"Expected ``len(cutoffs)={len(n)}`` but found ``{len(cutoffs)}``."
        raise ValueError(msg)

    shape = tuple(c + 1 for c in cutoffs)
    array = np.zeros(shape, dtype=np.complex128)

    try:
        array[tuple(n)] = 1
    except IndexError as e:
        msg = "Photon numbers cannot be larger than the corresponding cutoffs."
        raise ValueError(msg) from e

    return math.astensor(array)


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""Maps a ket to a density matrix.

    Args:
        ket: the ket

    Returns:
        Tensor: the density matrix
    """
    return math.outer(ket, math.conj(ket))


def ket_to_probs(ket: Tensor) -> Tensor:
    r"""Maps a ket to probabilities.

    Args:
        ket: the ket

    Returns:
        Tensor: the probabilities vector
    """
    return math.abs(ket) ** 2


def dm_to_probs(dm: Tensor) -> Tensor:
    r"""Extracts the diagonals of a density matrix.

    Args:
        dm: the density matrix

    Returns:
        Tensor: the probabilities vector
    """
    return math.all_diagonals(dm, real=True)


def U_to_choi(U: Tensor, Udual: Tensor | None = None) -> Tensor:
    r"""Converts a unitary transformation to a Choi tensor.

    Args:
        U: the unitary transformation
        Udual: the dual unitary transformation (optional, will use conj U if not provided)

    Returns:
        Tensor: the Choi tensor. The index order is going to be :math:`[\mathrm{out}_l, \mathrm{in}_l, \mathrm{out}_r, \mathrm{in}_r]`
        where :math:`\mathrm{in}_l` and :math:`\mathrm{in}_r` are to be contracted with the left and right indices of the density matrix.
    """
    return math.outer(U, math.conj(U) if Udual is None else Udual)


def fidelity(dm_a, dm_b) -> Scalar:
    r"""Computes the fidelity between two states in Fock representation."""
    # Richard Jozsa (1994) Fidelity for Mixed Quantum States,
    # Journal of Modern Optics, 41:12, 2315-2323, DOI: 10.1080/09500349414552171
    sqrt_dm_a = math.sqrtm(dm_a)
    return math.abs(math.trace(math.sqrtm(math.matmul(sqrt_dm_a, dm_b, sqrt_dm_a))) ** 2)


def number_means(tensor, is_dm: bool):
    r"""Returns the mean of the number operator in each mode."""
    probs = math.all_diagonals(tensor, real=True) if is_dm else math.abs(tensor) ** 2
    modes = list(range(len(probs.shape)))
    marginals = [math.sum(probs, axis=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    return math.astensor(
        [
            math.sum(marginal * math.arange(len(marginal), dtype=math.float64))
            for marginal in marginals
        ]
    )


def number_variances(tensor, is_dm: bool):
    r"""Returns the variance of the number operator in each mode."""
    probs = math.all_diagonals(tensor, real=True) if is_dm else math.abs(tensor) ** 2
    modes = list(range(len(probs.shape)))
    marginals = [math.sum(probs, axis=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    return math.astensor(
        [
            (
                math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype) ** 2)
                - math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype)) ** 2
            )
            for marginal in marginals
        ]
    )


def validate_contraction_indices(in_idx, out_idx, M, name):
    r"""Validates the indices used for the contraction of a tensor."""
    if len(set(in_idx)) != len(in_idx):
        raise ValueError(f"{name}_in_idx should not contain repeated indices.")
    if len(set(out_idx)) != len(out_idx):
        raise ValueError(f"{name}_out_idx should not contain repeated indices.")
    if not set(range(M)).intersection(out_idx).issubset(set(in_idx)):
        wrong_indices = set(range(M)).intersection(out_idx) - set(in_idx)
        raise ValueError(
            f"Indices {wrong_indices} in {name}_out_idx are trying to replace uncontracted indices."
        )


@tensor_int_cache
def oscillator_eigenstate(q: Vector, cutoff: int) -> Tensor:
    r"""Harmonic oscillator eigenstate wavefunction `\psi_n(q) = <n|q>`.

    Args:
        q (Vector): a vector containing the q points at which the function is evaluated (units of \sqrt{\hbar})
        cutoff (int): maximum number of photons

    Returns:
        Tensor: a tensor of size ``len(q)*cutoff``. Each entry with index ``[i, j]`` represents the eigenstate evaluated
            with number of photons ``i`` evaluated at position ``q[j]``, i.e., `\psi_i(q_j)`.

    .. details::

        .. admonition:: Definition
            :class: defn

        The q-quadrature eigenstates are defined as

        .. math::

            \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
                \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

        where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
    """
    hbar = settings.HBAR
    x = math.cast(q / np.sqrt(hbar), math.complex128)  # unit-less vector

    # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
    prefactor = math.cast(
        (np.pi * hbar) ** (-0.25) * math.pow(0.5, math.arange(0, cutoff) / 2),
        math.complex128,
    )

    # Renormalized physicist hermite polys: Hn / sqrt(n!)
    R = -np.array([[2 + 0j]])  # to get the physicist polys

    def f_hermite_polys(xi):  # pragma: no cover
        return math.hermite_renormalized(R, math.astensor([2 * xi]), 1 + 0j, (cutoff,))

    hermite_polys = math.map_fn(f_hermite_polys, x)

    # (real) wavefunction
    psi = math.exp(-(x**2 / 2)) * math.transpose(prefactor * hermite_polys)
    return psi


@lru_cache
def estimate_dx(cutoff, period_resolution=20):
    r"""Estimates a suitable quadrature discretization interval `dx`. Uses the fact
    that Fock state `n` oscillates with angular frequency :math:`\sqrt{2(n + 1)}`,
    which follows from the relation

    .. math::

            \psi^{[n]}'(q) = q - sqrt(2*(n + 1))*\psi^{[n+1]}(q)

    by setting q = 0, and approximating the oscillation amplitude by `\psi^{[n+1]}(0)

    Ref: https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions

    Args
        cutoff (int): Fock cutoff
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller `dx`.

    Returns
        (float): discretization value of quadrature
    """
    fock_cutoff_frequency = np.sqrt(2 * (cutoff + 1))
    fock_cutoff_period = 2 * np.pi / fock_cutoff_frequency
    dx_estimate = fock_cutoff_period / period_resolution
    return dx_estimate


@lru_cache
def estimate_xmax(cutoff, minimum=5):
    r"""Estimates a suitable quadrature axis length

    Args
        cutoff (int): Fock cutoff
        minimum (float): Minimum value of the returned xmax

    Returns
        (float): maximum quadrature value
    """
    if cutoff == 0:
        xmax_estimate = 3
    else:
        # maximum q for a classical particle with energy n=cutoff
        classical_endpoint = np.sqrt(2 * cutoff)
        # approximate probability of finding particle outside classical region
        excess_probability = 1 / (7.464 * cutoff ** (1 / 3))
        # Emperical factor that yields reasonable results
        A = 5
        xmax_estimate = classical_endpoint * (1 + A * excess_probability)
    return max(minimum, xmax_estimate)


@lru_cache
def estimate_quadrature_axis(cutoff, minimum=5, period_resolution=20):
    """Generates a suitable quadrature axis.

    Args
        cutoff (int): Fock cutoff
        minimum (float): Minimum value of the returned xmax
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller dx.

    Returns
        (array): quadrature axis
    """
    xmax = estimate_xmax(cutoff, minimum=minimum)
    dx = estimate_dx(cutoff, period_resolution=period_resolution)
    xaxis = np.arange(-xmax, xmax, dx)
    xaxis = np.append(xaxis, xaxis[-1] + dx)
    xaxis = xaxis - np.mean(xaxis)  # center around 0
    return xaxis


def quadrature_basis(
    fock_array: Tensor,
    quad: Batch[Vector],
    conjugates: bool | list[bool] = False,
    phi: Scalar = 0.0,
):
    r"""Given the Fock basis representation return the quadrature basis representation.

    Args:
        fock_array (Tensor): fock tensor amplitudes
        quad (Batch[Vector]): points at which the quadrature basis is evaluated
        conjugates (list[bool]): which dimensions of the array to conjugate based on
            whether it is a bra or a ket
        phi (float): angle of the quadrature basis vector

    Returns:
        tuple(Tensor): quadrature basis representation at the points in quad
    """
    dims = len(fock_array.shape)

    if quad.shape[-1] != dims:
        raise ValueError(
            f"Input fock array has dimension {dims} whereas ``quad`` has {quad.shape[-1]}."
        )

    conjugates = conjugates if isinstance(conjugates, Iterable) else [conjugates] * dims

    # construct quadrature basis vectors
    shapes = fock_array.shape
    quad_basis_vecs = []
    for dim in range(dims):
        q_to_n = oscillator_eigenstate(quad[..., dim], shapes[dim])
        if not np.isclose(phi, 0.0):
            theta = -math.arange(shapes[dim]) * phi
            Ur = math.make_complex(math.cos(theta), math.sin(theta))
            q_to_n = math.einsum("n,nq->nq", Ur, q_to_n)
        if conjugates[dim]:
            q_to_n = math.conj(q_to_n)
        quad_basis_vecs += [math.cast(q_to_n, "complex128")]

    # Convert each dimension to quadrature
    fock_string = "".join([chr(i) for i in range(98, 98 + dims)])  #'bcd....'
    q_string = "".join([fock_string[i] + "a," for i in range(dims - 1)] + [fock_string[-1] + "a"])
    quad_array = math.einsum(
        fock_string + "," + q_string + "->" + "a", fock_array, *quad_basis_vecs
    )

    return quad_array


def quadrature_distribution(
    state: Tensor,
    quadrature_angle: float = 0.0,
    x: Vector | None = None,
):
    r"""Given the ket or density matrix of a single-mode state, it generates the probability
    density distribution :math:`\tr [ \rho |x_\phi><x_\phi| ]`  where `\rho` is the
    density matrix of the state and |x_\phi> the quadrature eigenvector with angle `\phi`
    equal to ``quadrature_angle``.

    Args:
        state (Tensor): single mode state ket or density matrix
        quadrature_angle (float): angle of the quadrature basis vector
        x (Vector): points at which the quadrature distribution is evaluated

    Returns:
        tuple(Vector, Vector): coordinates at which the pdf is evaluated and the probability distribution
    """
    cutoff = state.shape[0]
    if x is None:
        x = np.sqrt(settings.HBAR) * math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")

    dims = len(state.shape)
    is_dm = dims == 2

    quad = math.transpose(math.astensor([x] * dims))
    conjugates = [True, False] if is_dm else [False]
    quad_basis = quadrature_basis(state, quad, conjugates, quadrature_angle)
    pdf = quad_basis if is_dm else math.abs(quad_basis) ** 2

    return x, math.real(pdf)


def sample_homodyne(state: Tensor, quadrature_angle: float = 0.0) -> tuple[float, float]:
    r"""Given a single-mode state, it generates the pdf of :math:`\tr [ \rho |x_\phi><x_\phi| ]`
    where `\rho` is the reduced density matrix of the state.

    Args:
        state (Tensor): ket or density matrix of the state being measured
        quadrature_angle (float): angle of the quadrature distribution

    Returns:
        tuple(float, float): outcome and probability of the outcome
    """
    dims = len(state.shape)
    if dims > 2:
        raise ValueError(
            "Input state has dimension {state.shape}. Make sure is either a single-mode ket or dm."
        )

    x, pdf = quadrature_distribution(state, quadrature_angle)
    probs = pdf * (x[1] - x[0])

    # draw a sample from the distribution
    pdf = math.Categorical(probs=probs, name="homodyne_dist")
    sample_idx = pdf.sample()
    homodyne_sample = math.gather(x, sample_idx)
    probability_sample = math.gather(probs, sample_idx)

    return homodyne_sample, probability_sample


@math.custom_gradient
def displacement(alpha, shape, tol=1e-15):
    r"""creates a single mode displacement matrix"""
    alpha = math.asnumpy(alpha)
    alpha = np.array(alpha)
    if math.abs(alpha) > tol:
        gate = strategies.displacement(tuple(shape), complex(alpha))
    else:
        gate = math.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]
    ret = math.astensor(gate, dtype=gate.dtype.name)
    if math.backend_name in ["numpy", "jax"]:
        return ret

    def grad(dL_dDc):
        dD_da, dD_dac = strategies.jacobian_displacement(math.asnumpy(gate), alpha)
        dL_dac = np.sum(np.conj(dL_dDc) * dD_dac + dL_dDc * np.conj(dD_da))
        return math.astensor(dL_dac)

    return ret, grad


@math.custom_gradient
def beamsplitter(theta: float, phi: float, shape: Sequence[int], method: str):
    r"""Creates a beamsplitter tensor with given cutoffs using a numba-based fock lattice strategy.

    Args:
        theta (float): transmittivity angle of the beamsplitter
        phi (float): phase angle of the beamsplitter
        cutoffs (int,int): cutoff dimensions of the two modes
        method (str): method to compute the beamsplitter ("vanilla", "schwinger" or "stable")
    """
    t, s = math.asnumpy(theta), math.asnumpy(phi)
    if method == "vanilla":
        bs_unitary = strategies.beamsplitter(shape, t, s)
    elif method == "schwinger":
        bs_unitary = strategies.beamsplitter_schwinger(shape, t, s)
    elif method == "stable":
        bs_unitary = strategies.stable_beamsplitter(shape, t, s)
    else:
        raise ValueError(f"Unknown method {method}. Use 'vanilla', 'schwinger' or 'stable'.")

    ret = math.astensor(bs_unitary, dtype=bs_unitary.dtype.name)
    if math.backend_name in ["numpy", "jax"]:
        return ret

    def vjp(dLdGc):
        dtheta, dphi = strategies.beamsplitter_vjp(
            math.asnumpy(bs_unitary),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(theta),
            math.asnumpy(phi),
        )
        return math.astensor(dtheta, dtype=theta.dtype), math.astensor(dphi, dtype=phi.dtype)

    return ret, vjp


@math.custom_gradient
def squeezer(r, phi, shape):
    r"""creates a single mode squeezer matrix using a numba-based fock lattice strategy"""
    sq_unitary = strategies.squeezer(shape, math.asnumpy(r), math.asnumpy(phi))

    ret = math.astensor(sq_unitary, dtype=sq_unitary.dtype.name)
    if math.backend_name in ["numpy", "jax"]:
        return ret

    def vjp(dLdGc):
        dr, dphi = strategies.squeezer_vjp(
            math.asnumpy(sq_unitary),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(r),
            math.asnumpy(phi),
        )
        return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

    return ret, vjp


@math.custom_gradient
def squeezed(r, phi, shape):
    r"""creates a single mode squeezed state using a numba-based fock lattice strategy"""
    sq_ket = strategies.squeezed(shape, math.asnumpy(r), math.asnumpy(phi))

    ret = math.astensor(sq_ket, dtype=sq_ket.dtype.name)
    if math.backend_name in ["numpy", "jax"]:  # pragma: no cover
        return ret

    def vjp(dLdGc):
        dr, dphi = strategies.squeezed_vjp(
            math.asnumpy(sq_ket),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(r),
            math.asnumpy(phi),
        )
        return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

    return ret, vjp


def c_ps_matrix(m, n, alpha):
    """
    helper function for ``c_in_PS``.
    """
    mu_range = range(max(0, alpha - n), min(m, alpha) + 1)
    tmp = [comb(m, mu) * comb(n, alpha - mu) * (1j) ** (m - n - 2 * mu + alpha) for mu in mu_range]
    return np.sum(tmp)


def gamma_matrix(c):
    """
    helper function for ``c_in_PS`.
    constructs the matrix transformation that helps transforming ``c``.
    ``c`` here must be 2-dimensional.
    """
    M = c.shape[0] + c.shape[1] - 1
    Gamma = np.zeros((M**2, c.shape[0] * c.shape[1]), dtype=np.complex128)

    for m in range(c.shape[0]):
        for n in range(c.shape[1]):
            for alpha in range(m + n + 1):
                factor = math.sqrt(
                    factorial(m) * factorial(n) / (factorial(alpha) * factorial(m + n - alpha))
                )
                value = c_ps_matrix(m, n, alpha) * math.sqrt(settings.HBAR / 2) ** (m + n)
                row = alpha * M + (m + n - alpha)
                col = m * c.shape[0] + n
                Gamma[row, col] = value / factor
    return Gamma


def c_in_PS(c):
    """
    Transforms the ``c`` matrix of a ``DM`` object from bargmann to phase-space.
    It is a helper function used in

    Args:
        c (Tensor): the 2-dimensional ``c`` matrix of the ``DM`` object
    """
    M = c.shape[0] + c.shape[1] - 1
    return np.reshape(gamma_matrix(c) @ np.reshape(c, (c.shape[0] * c.shape[1], 1)), (M, M))
