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

"""Utility functions related to homodyne sampling on Fock representation"""

from __future__ import annotations
from typing import TYPE_CHECKING
from functools import lru_cache, wraps

import numpy as np
from scipy.special import factorial

from mrmustard import settings
from mrmustard.types import Tuple, List, Union, Iterable, Tensor
import mrmustard.lab as lab
from mrmustard.math import Math

if TYPE_CHECKING:
    from mrmustard.lab.abstract import State

math = Math()


def hermite_cache(fn):
    """Decorator function to cache outcomes of the physicist_hermite_polys
    function. To do so the input tensor (non-hashable) is converted into a
    numpy array (non-hashable) and then a tuple (hashable) is used in conjuntion
    with ``functools.lru_cache``."""

    @lru_cache()
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
    physicist polys.

    Args:
        x (Tensor): argument values of the Hermite polynomial
        cutoff (int): maximum size of the subindices in the Hermite polynomial

    Returns:
        Tensor: the evaluated Hermite polynomials
    """
    R = math.astensor(2 * np.ones([1, 1]))  # to get the physicist polys

    def f_hermite_polys(xi):
        return math.hermite_renormalized(R, math.astensor([xi]), 1, cutoff)

    return math.map_fn(f_hermite_polys, x)


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
        # approximate probability of finding particle outside classical region (see Wikipedia)
        excess_probability = 1 / (7.464 * cutoff ** (1 / 3))
        # Emperical factor that yields reasonable results
        A = 5
        xmax_estimate = classical_endpoint * (1 + A * excess_probability)
    return max(minimum, xmax_estimate)


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


def sample_homodyne_fock(
    state: State, quadrature_angle: float, mode: Union[int, List[int]]
) -> Tuple[float, State]:
    r"""Given a state, it generates the pdf of :math:`\tr [ \rho |x><x| ]`
    where `\rho` is the reduced density matrix of the ``other`` state on the
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
        state (State): state being measured
        quadrature_angle (float): measurement quadrature angle
        mode: the modes of the state being measured

    Returns:
        tuple(float, State): homodyne outcome and projector state
    """

    if isinstance(mode, int):
        mode = [mode]
    elif isinstance(mode, Iterable) and len(mode) != 1:
        raise ValueError(
            f"Requested homodyne sampling for {len(mode)} modes. \
                Homodyne sampling is supported on a single mode."
        )

    # create reduced state of mode to be measured on the homodyne basis
    reduced_state = state.get_modes(mode) >> lab.Rgate(-quadrature_angle, modes=mode)
    cutoff = reduced_state.cutoffs[0]

    omega_over_hbar = 1 / settings.HBAR
    q_tensor = math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")
    x = np.sqrt(omega_over_hbar) * q_tensor
    hermite_polys = physicist_hermite_polys(x, cutoff)

    if reduced_state.is_pure:

        reduced_ket = math.real(reduced_state.ket())

        prefactor = 2 ** (-math.arange(0, cutoff) / 2)

        sum_terms = math.squeeze(prefactor * reduced_ket * math.expand_dims(hermite_polys, 0))

        rho_dist = (
            math.sum(sum_terms, axes=[1]) ** 2
            * (omega_over_hbar / np.pi) ** 0.5
            * math.exp(-(x**2))
        )

    else:
        # mixed state

        # taking only the real part because the imaginary part cancels out during summation,
        # this saves some type casting of real matrices into complex ones.
        reduced_dm = math.real(reduced_state.dm())

        # build matrix of terms Hn Hm / sqrt(n! m!)
        hermite_polys = math.expand_dims(hermite_polys, axis=-1)
        hermite_matrix = math.matmul(hermite_polys, hermite_polys, transpose_b=True)

        # build matrix of terms 1 / sqrt( 2**(n+m) )
        prefactor = math.expand_dims(2 ** (-math.arange(0, cutoff) / 2), axis=-1)
        prefactor = math.matmul(prefactor, prefactor, transpose_b=True)

        # build terms inside the sum: `\rho_{n,m} Hn Hm / sqrt( 2**(n+m) n! m!)`
        sum_terms = (
            math.expand_dims(prefactor, 0) * math.expand_dims(reduced_dm, 0) * hermite_matrix
        )

        # calculate the pdf and multiply by factors outside the sum
        rho_dist = (
            math.sum(sum_terms, axes=[1, 2])
            * (omega_over_hbar / np.pi) ** 0.5
            * math.exp(-(x**2))
        )

    # draw a sample from the distribution
    pdf = math.Categorical(probs=rho_dist, name="rho_dist")
    sample_idx = pdf.sample()
    homodyne_sample = math.gather(q_tensor, sample_idx)

    # create "projector state" to calculate the conditional output state
    projector_state = lab.DisplacedSqueezed(
        r=settings.HOMODYNE_SQUEEZING, phi=0, x=homodyne_sample, y=0, modes=reduced_state.modes
    ) >> lab.Rgate(quadrature_angle, modes=reduced_state.modes)

    return homodyne_sample, projector_state
