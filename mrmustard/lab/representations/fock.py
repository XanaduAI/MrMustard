 Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import numpy as np

from mrmustard.math import Math
# from mrmustard import settings
from mrmustard.representations import Representation
from mrmustard.representations.data import ArrayData
from mrmustard.typing import Scalar, Tensor, RealVector
# from mrmustard.math.caching import tensor_int_cache

math = Math()

class Fock(Representation):
    '''Fock Class is the Fock representation.'''

    def __init__(self, array):
        super().__init__()
        self.data = ArrayData(array) 


    def number_means(self) -> Tensor:
        r'''Returns the mean photon number in each mode.'''
        probs = self.probability()
        modes = list(range(len(probs.shape)))
        marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
        return math.astensor(
            [
                math.sum(marginal * math.arange(len(marginal), dtype=marginal.dtype))
                for marginal in marginals
            ]
        )
    

    def number_variances(self) -> Tensor:
        r"""Returns the variance of the number operator in each mode."""
        probs = self.probability()
        modes = list(range(len(probs.shape)))
        marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
        return math.astensor(
            [
                (
                    math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype) ** 2)
                    - math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype)) ** 2
                )
                for marginal in marginals
            ]
        )
    
    def number_stdev(self) -> RealVector:
        r"""Returns the square root of the photon number variances (standard deviation) in each mode."""
        return math.sqrt(self.number_variances())


    def number_cov(self):
        raise NotImplementedError("number_cov not yet implemented for non-gaussian states")


    def von_neumann_entropy(cov: Matrix, hbar: float) -> float:
        r"""Returns the Von Neumann entropy."""
        raise NotImplementedError("von_neumann_entropy not yet implemented for Fock representation") 
        # # @tensor_int_cache
        # def oscillator_eigenstates(q: RealVector, cutoff: int) -> Tensor:
        #     r"""Harmonic oscillator eigenstate wavefunctions `\psi_n(q) = <q|n>` for n = 0, 1, 2, ..., cutoff-1.

        #     Args:
        #         q (Vector): a vector containing the q points at which the function is evaluated (units of \sqrt{\hbar})
        #         cutoff (int): maximum number of photons

        #     Returns:
        #         Tensor: a tensor of shape ``(cutoff, len(q))``. The entry with index ``[n, j]`` represents the eigenstate evaluated
        #             with number of photons ``n`` evaluated at position ``q[j]``, i.e., `\psi_n(q_j) = <q_j|n>`.

        #     .. details::

        #         .. admonition:: Definition
        #             :class: defn

        #         The q-quadrature eigenstates are defined as

        #         .. math::

        #             \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
        #                 \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

        #         where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
        #     """
        #     omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
        #     x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

        #     # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
        #     prefactor = (omega_over_hbar / np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

        #     # Renormalized physicist hermite polys: Hn / sqrt(n!)
        #     R = np.array([[2 + 0j]])  # to get the physicist polys

        #     def f_hermite_polys(xi):
        #         poly = math.hermite_renormalized(R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff)
        #         return math.cast(poly, "float64")

        #     hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

        #     # (real) wavefunction
        #     psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
        #     return psi
        

        # def validate_contraction_indices(in_idx, out_idx, M, name):
        #     r"""Validates the indices used for the contraction of a tensor."""
        #     if len(set(in_idx)) != len(in_idx):
        #         raise ValueError(f"{name}_in_idx should not contain repeated indices.")
        #     if len(set(out_idx)) != len(out_idx):
        #         raise ValueError(f"{name}_out_idx should not contain repeated indices.")
        #     if not set(range(M)).intersection(out_idx).issubset(set(in_idx)):
        #         wrong_indices = set(range(M)).intersection(out_idx) - set(in_idx)
        #         raise ValueError(
        #             f"Indices {wrong_indices} in {name}_out_idx are trying to replace uncontracted indices."
        #         )