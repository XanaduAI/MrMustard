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
from mrmustard.representations import Fock
from mrmustard.representations.data import ArrayData
from mrmustard.typing import Scalar, Tensor, RealVector
from mrmustard.representations.fock import validate_contraction_indices

math = Math()

class FockKet(Fock):
    '''FockKet Class is the Fock representation of a ket state.'''

    def __init__(self):
        super().__init__()
        self.num_modes = len(self.array.shape) #TODO: BATCH ISSUE?


    def purity(self) -> Scalar:
        return 1.0


    # def number_means(self) -> Tensor:
    #     r'''Returns the mean photon number in each mode.'''
    #     ket = self.data.array
    #     probs = math.abs(ket) ** 2
    #     modes = list(range(len(probs.shape)))
    #     marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    #     return math.astensor(
    #         [
    #             math.sum(marginal * math.arange(len(marginal), dtype=marginal.dtype))
    #             for marginal in marginals
    #         ]
    #     )
    # 
    # 
    # def number_variances(self) -> Tensor:
    #     r"""Returns the variance of the number operator in each mode."""
    #     ket = self.data.array
    #     probs = math.abs(ket) ** 2
    #     modes = list(range(len(probs.shape)))
    #     marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    #     return math.astensor(
    #         [
    #             (
    #                 math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype) ** 2)
    #                 - math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype)) ** 2
    #             )
    #             for marginal in marginals
    #         ]
    #     )
    
    # def number_stdev(self) -> RealVector:
    #     r"""Returns the square root of the photon number variances (standard deviation) in each mode."""
    #     return math.sqrt(self.number_variances())



    # def number_cov(self):
    #     raise NotImplementedError("number_cov not yet implemented for non-gaussian states")
    

    def norm(self):
        r"""
        Returns the norm. (:math:`|amp|` for ``ket``)
        """
        return math.abs(math.norm(self.data.array))
    

    def probability(self, cutoffs: Sequence[int] ) -> Tensor: 
        r"""Maps a ket to probabilities.
        """
        #TODO: cutoffs
        return math.abs(self.data.array)[cutoffs]
    

    def apply_kraus_to_ket(kraus, ket, kraus_in_idx, kraus_out_idx=None):
        r"""Applies a kraus operator to a ket.
        It assumes that the ket is indexed as left_1, ..., left_n.

        The kraus op has indices that contract with the ket (kraus_in_idx) and indices that are left over (kraus_out_idx).
        The final index order will be sorted (note that an index appearing in both kraus_in_idx and kraus_out_idx will replace the original index).

        Args:
            kraus (array): the kraus operator to be applied
            ket (array): the ket to which the operator is applied
            kraus_in_idx (list of ints): the indices (counting from 0) of the kraus operator that contract with the ket
            kraus_out_idx (list of ints): the indices (counting from 0) of the kraus operator that are leftover

        Returns:
            array: the resulting ket with indices as kraus_out_idx + uncontracted ket indices
        """
        if kraus_out_idx is None:
            kraus_out_idx = kraus_in_idx

        if not set(kraus_in_idx).issubset(range(ket.ndim)):
            raise ValueError("kraus_in_idx should be a subset of the ket indices.")

        # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
        validate_contraction_indices(kraus_in_idx, kraus_out_idx, ket.ndim, "kraus")

        ket = MMTensor(ket, axis_labels=[f"left_{i}" for i in range(ket.ndim)])
        kraus = MMTensor(
            kraus,
            axis_labels=[f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx],
        )

        # contract the operator with the ket.
        # now the leftover indices are in the order kraus_out_idx + uncontracted ket indices
        kraus_ket = kraus @ ket

        # sort kraus_ket.axis_labels by the int at the end of each label.
        # Each label is guaranteed to have a unique int at the end.
        new_axis_labels = sorted(kraus_ket.axis_labels, key=lambda x: int(x.split("_")[-1]))

        return kraus_ket.transpose(new_axis_labels).tensor