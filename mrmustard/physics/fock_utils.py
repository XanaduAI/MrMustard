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

"""
This module contains functions for performing calculations on objects in the Fock representations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

import jax
import numpy as np
from scipy.special import comb, factorial

from mrmustard import math, settings
from mrmustard.math.caching import tensor_int_cache
from mrmustard.utils.typing import Batch, Scalar, Tensor, Vector

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fock_state(n: int | Sequence[int], cutoff: int | None = None) -> Tensor:
    r"""
    The Fock array of a batchable single-mode ``Number`` state.

    Args:
        n: The photon number of the number state. Can be a single integer or a batch of integers.
        cutoff: The cutoff of the Fock array. This acts as the core dimension of the returned array.
            If ``None``, it defaults to ``math.max(n)+1``.

    Returns:
        The Fock array of a batchable single-mode ``Number`` state.

    Raises:
        ValueError: If the photon numbers are larger than the corresponding cutoffs.
    """
    n = math.astensor(n, dtype=math.int64)
    if cutoff is None:
        cutoff = int(math.max(n) + 1)

    def check_photon_numbers(n, cutoff):
        if math.any(n >= cutoff):
            raise ValueError("Photon numbers cannot be larger than the corresponding cutoff.")

    if math.backend_name == "jax":  # pragma: no cover
        jax.debug.callback(check_photon_numbers, n, cutoff)
    else:
        check_photon_numbers(n, cutoff)

    return math.eye(cutoff)[n]


def fidelity(dm_a, dm_b) -> Scalar:
    r"""Computes the fidelity between two states in Fock representation."""
    # Richard Jozsa (1994) Fidelity for Mixed Quantum States,
    # Journal of Modern Optics, 41:12, 2315-2323, DOI: 10.1080/09500349414552171
    sqrt_dm_a = math.sqrtm(dm_a)
    return math.abs(math.trace(math.sqrtm(math.matmul(sqrt_dm_a, dm_b, sqrt_dm_a))) ** 2)


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
    return math.exp(-(x**2 / 2)) * math.transpose(prefactor * hermite_polys)


@lru_cache
def estimate_dx(cutoff, period_resolution=20):
    r"""Estimates a suitable quadrature discretization interval `dx`. Uses the fact
    that Fock state `n` oscillates with angular frequency :math:`\sqrt{2(n + 1)}`,
    which follows from the relation

    .. math::

        \psi^{[n]}'(q) = q - sqrt(2*(n + 1))*\psi^{[n+1]}(q)

    by setting q = 0, and approximating the oscillation amplitude by `\psi^{[n+1]}(0)`.

    Ref: https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions

    Args:
        cutoff (int): Fock cutoff
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller `dx`.

    Returns:
        (float): discretization value of quadrature
    """
    fock_cutoff_frequency = np.sqrt(2 * (cutoff + 1))
    fock_cutoff_period = 2 * np.pi / fock_cutoff_frequency
    return fock_cutoff_period / period_resolution


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
    r"""Generates a suitable quadrature axis.

    Args:
        cutoff (int): Fock cutoff
        minimum (float): Minimum value of the returned xmax
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller dx.

    Returns:
        (array): quadrature axis
    """
    xmax = estimate_xmax(cutoff, minimum=minimum)
    dx = estimate_dx(cutoff, period_resolution=period_resolution)
    xaxis = np.arange(-xmax, xmax, dx)
    xaxis = np.append(xaxis, xaxis[-1] + dx)
    return xaxis - np.mean(xaxis)  # center around 0


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
            f"Input fock array has dimension {dims} whereas ``quad`` has {quad.shape[-1]}.",
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
    return math.einsum(fock_string + "," + q_string + "->" + "a", fock_array, *quad_basis_vecs)


def quadrature_distribution(
    state: Tensor,
    quadrature_angle: float = 0.0,
    x: Vector | None = None,
):
    r"""
    Given the ket or density matrix of a single-mode state, it generates the probability
    density distribution :math:`\tr [ \rho |x_\phi><x_\phi| ]` where ``\rho`` is the
    density matrix of the state and ``|x_\phi>`` the quadrature eigenvector with angle ``\phi``
    equal to ``quadrature_angle``.

    Args:
        state: A single mode state ket or density matrix.
        quadrature_angle: The angle of the quadrature basis vector.
        x: The points at which the quadrature distribution is evaluated.

    Returns:
        The coordinates at which the pdf is evaluated and the probability distribution.
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
                    factorial(m) * factorial(n) / (factorial(alpha) * factorial(m + n - alpha)),
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
