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

import numpy as np

from mrmustard.math import Math
from mrmustard.representations import Representation
from mrmustard.representations.data import ArrayData
from mrmustard.typing import Scalar, Tensor, RealVector

math = Math()

class FockKet(Representation):
    '''FockKet Class is the Fock representation of a ket state.'''

    def __init__(self, ket):
        super().__init__()
        self.data = ArrayData(ket) 


    def purity(self) -> Scalar:
        return 1.0


    def number_means(self) -> Tensor:
        r'''Returns the mean photon number in each mode.'''
        ket = self.data.array
        probs = math.abs(ket) ** 2
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
        ket = self.data.array
        probs = math.abs(ket) ** 2
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
    

    def norm(self):
        r"""
        Returns the norm. (:math:`|amp|` for ``ket``)
        """
        return math.abs(math.norm(self.data.array))
    

    def probability(self) -> Tensor: 
        r"""Maps a ket to probabilities.
        """
        return math.abs(self.data.array)
    

    def __eq__(self, other:Representation) -> bool:
        r"""Compares two Representations (States) equal or not"""


    def __rmul__(self, other:Representation) -> Representation:
        r"""Adds two Representations (States)"""


    def __add__(self, other:Representation) -> Representation:
        r"""Adds two Representations (States)"""


    def __truediv__(self, other:Representation) -> Representation:
        r"""Adds two Representations (States)"""