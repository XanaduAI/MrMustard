# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions related to homodyne sampling in Fock representation"""
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import lru_cache, wraps

from numba import njit
import numpy as np

from mrmustard.types import Tuple, Tensor
from mrmustard.math import Math

if TYPE_CHECKING:
    from mrmustard.lab.abstract import State

math = Math()


def hermite_cache(fn):
    """Decorator function to cache outcomes of the physicist_hermite_polys
    function. To do so the input tensor (non-hashable) is converted into a
    numpy array (non-hashable) and then a tuple (hashable) is used in conjuntion
    with ``functools.lru_cache``."""

    @lru_cache
    def cached_wrapper(hashable_array, cutoff):
        array = np.array(hashable_array)
        return fn(array, cutoff)

    @wraps(fn)
    def wrapper(tensor, cutoff):
        return cached_wrapper(tuple(tensor.numpy()), cutoff)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@hermite_cache
def physicist_hermite_polys(x: Tensor, cutoff: int):
    r"""Reduction of the multidimensional hermite polynomials into the one-dimensional
    renormalized physicist polys.

    Args:
        x (Tensor): argument values of the Hermite polynomial
        cutoff (int): maximum size of the subindices in the Hermite polynomial

    Returns:
        Tensor: the evaluated renormalized Hermite polynomials
    """
    R = math.astensor(2 * np.ones([1, 1]))  # to get the physicist polys

    def f_hermite_polys(xi):
        return math.hermite_renormalized(R, math.astensor([xi]), 1, cutoff, modified=False)

    return math.map_fn(f_hermite_polys, x)


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


def sample_homodyne_fock(state: State, hbar: float) -> Tuple[float, float, State]:
    r"""Given a single-mode state, it generates the pdf of :math:`\tr [ \rho |x><x| ]`
    where `\rho` is the reduced density matrix of the state on the
    measured mode.

    Here the following quadrature wavefunction for the Fock states are used:

    .. math::

        \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
            \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

    where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial. Hence, the
    probability density function is

    .. math ::

        p(\rho|x) = \tr [ \rho |x><x| ] = \sum_{n,m} \rho_{n,m} \psi_n(x) \psi_m(x)

    Args:
        state (array): ket or dm of the state being measured
        hbar: value of hbar

    Returns:
        tuple(float, float): outcome and probability of the outcome
    """
    if len(state.shape) == 1:
        x, probs = _probs_homodyne_pure(state, hbar)
    elif len(state.shape) == 2:
        x, probs = _probs_homodyne_mixed(state, hbar)
    else:
        raise ValueError(
            "Input state has dimension {state.shape}. Make sure is either a single mode ket or dm."
        )

    # draw a sample from the distribution
    pdf = math.Categorical(probs=probs, name="homodyne_dist")
    sample_idx = pdf.sample()
    homodyne_sample = math.gather(x, sample_idx)
    probability_sample = math.gather(probs, sample_idx)

    return homodyne_sample, probability_sample


def _probs_homodyne_pure(state_ket, hbar):

    cutoff = state_ket.shape[0]

    # calculate prefactors of the PDF
    omega_over_hbar = 1 / hbar
    q_tensor = math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")
    x = np.sqrt(hbar) * q_tensor

    # Hn / sqrt(n!)
    hermite_polys = math.cast(physicist_hermite_polys(q_tensor, cutoff), "complex128")
    # prefactor term 1 / 2**n
    prefactor = math.cast(2 ** (-math.arange(0, cutoff) / 2), "complex128")
    # build terms inside the sum: `\ket_{n} Hn / ( 2**n n! )`
    sum_terms = math.squeeze(prefactor * state_ket * math.expand_dims(hermite_polys, 0))
    # calculate the pdf and multiply by factors outside the sum
    out_factor = (omega_over_hbar / np.pi) ** 0.5 * math.exp(-(q_tensor**2)) * (x[1] - x[0])
    probs = out_factor * math.abs(math.sum(sum_terms, axes=[1])) ** 2

    return x, probs


def _probs_homodyne_mixed(state_dm, hbar):

    cutoff = state_dm.shape[0]

    # calculate prefactors of the PDF
    omega_over_hbar = 1 / hbar
    q_tensor = math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")
    x = np.sqrt(hbar) * q_tensor
    hermite_polys = physicist_hermite_polys(q_tensor, cutoff)

    # build matrix of terms Hn Hm / sqrt(n! m!)
    hermite_polys = math.expand_dims(hermite_polys, axis=-1)
    hermite_matrix = math.matmul(hermite_polys, hermite_polys, transpose_b=True)
    hermite_matrix = math.cast(hermite_matrix, "complex128")

    # build matrix of terms 1 / sqrt( 2**(n+m) )
    prefactor = math.expand_dims(2 ** (-math.arange(0, cutoff) / 2), axis=-1)
    prefactor = math.matmul(prefactor, prefactor, transpose_b=True)
    prefactor = math.cast(prefactor, "complex128")

    # build terms inside the sum: `\rho_{n,m} Hn Hm / sqrt( 2**(n+m) n! m!)`
    sum_terms = math.expand_dims(prefactor, 0) * math.expand_dims(state_dm, 0) * hermite_matrix

    # calculate the pdf and multiply by factors outside the sum
    out_factor = (omega_over_hbar / np.pi) ** 0.5 * math.exp(-(q_tensor**2)) * (x[1] - x[0])
    probs = out_factor * math.real(math.sum(sum_terms, axes=[1, 2]))

    return x, probs


def sample_heterodyne_fock(state: State, hbar: float) -> Tuple[float, float, State]:
    r"""Given a single-mode state, it samples the Husimi function :math:`\pi^{-1} \tr [ \rho |\alpha><\alpha| ]`,
    where `\rho` is the reduced density matrix of the state on the measured mode and `|\alpha>` is a coherent state.

    Args:
        state (array): ket or dm of the state being measured
        hbar: value of hbar

    Returns:
        tuple(float, float): outcome and probability of the outcome
    """
    if len(state.shape) == 1:
        Q, P, probs = _probs_heterodyne_pure(state, hbar)
    elif len(state.shape) == 2:
        Q, P, probs = _probs_heterodyne_mixed(state, hbar)
    else:
        raise ValueError(
            "Input state has dimension {state.shape}. Make sure is either a single mode ket or dm."
        )

    # draw a sample from the distribution
    probs = probs.flatten()
    Q = Q.flatten()
    P = P.flatten()
    pdf = math.Categorical(probs=probs, name="heterodyne_dist")
    sample_idx = pdf.sample()
    Q_sample = math.gather(Q, sample_idx)
    P_sample = math.gather(P, sample_idx)
    probability_sample = math.gather(probs, sample_idx)

    return Q_sample, P_sample, probability_sample


def _probs_heterodyne_pure(state_ket, hbar):

    cutoff = state_ket.shape[0]

    # calculate prefactors of the PDF
    q_tensor = math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")
    x = np.sqrt(hbar) * q_tensor
    p = hbar * q_tensor

    Q, P, probs = husimi_pure(state_ket.numpy(), x.numpy(), p.numpy(), hbar)

    return Q, P, probs


def _probs_heterodyne_mixed(state_dm, hbar):

    cutoff = state_dm.shape[0]

    # calculate prefactors of the PDF
    q_tensor = math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")
    x = np.sqrt(hbar) * q_tensor
    p = hbar * q_tensor

    Q, P, probs = husimi_mixed(state_dm.numpy(), x.numpy(), p.numpy(), hbar)

    return Q, P, probs


@njit
def husimi_pure(ket, xvec, pvec, hbar=1.0):
    r"""Calculates the discretized Wigner marginal on the X axis

    Adapted from `strawberryfields <https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L725>`

    Args:
        rho (complex array): the state in Fock representation (can be pure or mixed)
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        hbar (float): the value of ``\hbar``
    """

    Q = np.outer(pvec, np.ones_like(xvec))
    P = np.outer(np.ones_like(pvec), xvec)
    cutoff = ket.shape[-1]
    A2 = (Q**2 + P**2) / (2 * hbar)

    Hvec = np.zeros((1, cutoff) + A2.shape, dtype=np.complex128)

    # Husimi function for |0>
    Hvec[0] = np.exp(-A2) / np.pi
    H = np.real(ket[0]) * np.real(Hvec[0])

    # Husimi function for |n>
    for n in range(1, cutoff):
        Hvec[n] = (A2 * Hvec[n - 1]) / n
        H += np.real(ket[n] * Hvec[n])

    return Q, P, H / hbar


@njit
def husimi_mixed(rho, xvec, pvec, hbar=1.0):
    r"""Calculates the discretized Wigner marginal on the X axis

    Adapted from `strawberryfields <https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L725>`

    Args:
        rho (complex array): the state in Fock representation (can be pure or mixed)
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        hbar (float): the value of ``\hbar``
    """

    Q = np.outer(pvec, np.ones_like(xvec))
    P = np.outer(np.ones_like(pvec), xvec)
    cutoff = rho.shape[-1]
    A = (Q + P * 1.0j) / (2 * np.sqrt(hbar / 2))

    Hmat = np.zeros((2, cutoff) + A.shape, dtype=np.complex128)

    # Husimi function for |0><0|
    Hmat[0, 0] = np.exp(-np.abs(A) ** 2) / np.pi
    H = np.real(rho[0, 0]) * np.real(Hmat[0, 0])

    for n in range(1, cutoff):
        Hmat[0, n] = (A * Hmat[0, n - 1]) / np.sqrt(n)
        H += np.real(rho[0, n] * Hmat[0, n])

    for m in range(1, cutoff):
        # Husimi function for |m><m|
        Hmat[1, m] = (np.conj(A) * Hmat[0, m] - np.sqrt(m) * Hmat[0, m - 1]) / np.sqrt(m)
        H += np.real(rho[m, m] * Hmat[1, m])

        for n in range(m + 1, cutoff):
            # Husimi function for |m><n|
            Hmat[1, n] = (2 * A * Hmat[1, n - 1] - np.sqrt(m) * Hmat[0, n - 1]) / np.sqrt(n)
            H += 2 * np.real(rho[m, n] * Hmat[1, n])
        Hmat[0] = Hmat[1]

    return Q, P, H / hbar
