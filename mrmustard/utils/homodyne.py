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

import numpy as np
from scipy.special import factorial

from mrmustard import settings
from mrmustard.types import Tuple, List, Union, Iterable, Tensor
import mrmustard.lab as lab
from mrmustard.math import Math

if TYPE_CHECKING:
    from mrmustard.lab.abstract import State

math = Math()


def physicist_hermite_polys(x: Tensor, cutoff: int):
    r"""Reduction of the multidimensional hermite polynomials into the one-dimensional
    physicist polys.

    Args:
        x (float): evaluate
        cutoff (int):

    Returns:


    """
    R = math.astensor(2 * np.ones([1, 1]))  # to get the physicist polys

    def f_hermite_polys(xi):
        return math.hermite(R, math.astensor([xi]), 1, cutoff)

    return math.map_fn(f_hermite_polys, x)


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
        other (State): state used to build the pdf

    Returns:
        tuple(float, State): homodyne outcome and projector state
    """
    # pylint: disable=import-outside-toplevel
    # from mrmustard.lab import Rgate, DisplacedSqueezed

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

    # pdf reconstruction parameters
    num_bins = int(1e2)  # TODO: make kwarg?
    q_mag = 7  # TODO: make kwarg?

    # build `\psi_n(x) \psi_m(x)` terms
    omega_over_hbar = 1 / settings.HBAR
    q_tensor = math.new_constant(np.linspace(-q_mag, q_mag, num_bins), "q_tensor")
    x = np.sqrt(omega_over_hbar) * q_tensor
    hermite_polys = math.expand_dims(physicist_hermite_polys(x, cutoff), axis=-1)
    hermite_matrix = math.matmul(hermite_polys, hermite_polys, transpose_b=True)

    prefactor = np.empty_like(reduced_state.dm())
    for idx, _ in np.ndenumerate(prefactor):
        n, m = idx[0], idx[1]
        prefactor[idx] = 1 / (np.sqrt(2 ** (n + m) * factorial(n) * factorial(m)))

    # build terms in the sum: `rho_{n,m} \psi_n(x) \psi_m(x)`
    sum_terms = (
        math.expand_dims(prefactor, 0)
        * math.expand_dims(reduced_state.dm(), 0)
        * math.cast(hermite_matrix, "complex128")
    )

    # calculate the pdf
    rho_dist = (
        math.cast(math.sum(sum_terms, axes=[1, 2]), "float64")
        * (omega_over_hbar / np.pi) ** 0.5
        * math.exp(-(x**2))
    )
    pdf = math.Categorical(probs=rho_dist, name="rho_dist")

    # draw sample from the distribution
    sample_idx = pdf.sample()
    homodyne_sample = math.gather(q_tensor, sample_idx)

    # create "projector state" to calculate the conditional output state
    projector_state = lab.DisplacedSqueezed(
        r=settings.HOMODYNE_SQUEEZING, phi=0, x=homodyne_sample, y=0, modes=reduced_state.modes
    ) >> lab.Rgate(quadrature_angle, modes=reduced_state.modes)

    return homodyne_sample, projector_state
