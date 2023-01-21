# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains functions for transforming to the wavefunction representation.
"""

import numpy as np
from mrmustard.math.caching import tensor_int_cache
from mrmustard import settings
from mrmustard.math import Math
from mrmustard.types import Tensor, Vector
math = Math()


@tensor_int_cache
def oscillator_eigenstates(q: Vector, cutoff: int) -> Tensor:
    r"""Harmonic oscillator eigenstate wavefunctions `\psi_n(q) = <q|n>` for n = 0, 1, 2, ..., cutoff-1.

    Args:
        q (Vector): a vector containing the q points at which the function is evaluated (units of \sqrt{\hbar})
        cutoff (int): maximum number of photons

    Returns:
        Tensor: a tensor of shape ``(cutoff, len(q))``. The entry with index ``[n, j]`` represents the eigenstate evaluated
            with number of photons ``n`` evaluated at position ``q[j]``, i.e., `\psi_n(q_j) = <q_j|n>`.

    .. details::

        .. admonition:: Definition
            :class: defn

        The q-quadrature eigenstates are defined as

        .. math::

            \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
                \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

        where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
    """
    omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
    x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

    # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
    prefactor = (omega_over_hbar / np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

    # Renormalized physicist hermite polys: Hn / sqrt(n!)
    R = np.array([[2 + 0j]])  # to get the physicist polys

    def f_hermite_polys(xi):
        poly = math.hermite_renormalized(R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff)
        return math.cast(poly, "float64")

    hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

    # (real) wavefunction
    psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
    return psi
