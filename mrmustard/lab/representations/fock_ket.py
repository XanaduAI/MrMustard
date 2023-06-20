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
from mrmustard.lab.representations.fock import Fock
from mrmustard.typing import Tensor

math = Math()

class FockKet(Fock):
    r""" Fock representation of a state described by a ket.
    
    Args:
        data: the data used to represent the state to be encoded as Fock representation
    """

    def __init__(self, array:np.array):
        super().__init__(array=array)
        self.num_modes = len(self.array.shape)
        self.cutoffs = self.array.shape


    def purity(self) -> float:
        return 1.0


    def norm(self) -> float:
        return math.abs(math.norm(self.data.array))
    

    def probability(self) -> Tensor: 
        return math.abs(self.data.array) #TODO: cutoffs
    
    # NOTE : this is for transformations!
    # def apply_kraus_to_ket(self, kr:np.array, ket:np.array, 
    #                        kraus_in_idx:List[int], kraus_out_idx:Union[List[int], None]=None
    #                        ) -> Tensor:
    #     r"""Applies a kraus operator to a ket.
    #     It assumes that the ket is indexed as left_1, ..., left_n.

    #     The kraus op has indices that contract with the ket (kraus_in_idx) and indices that are 
    #     left over (kraus_out_idx).
    #     The final index order will be sorted (note that an index appearing in both kraus_in_idx and
    #       kraus_out_idx will replace the original index).

    #     Args:
    #         kr: the kraus operator to be applied
    #         ket: the ket to which the operator is applied
    #         kraus_in_idx: the indices (counting from 0) of the kraus operator that contract with 
    #                       the ket
    #         kraus_out_idx: the indices (counting from 0) of the kraus operator that are leftover

    #     Returns:
    #         array: the resulting ket with indices as kraus_out_idx + uncontracted ket indices

    #     Raises:
    #         ValueError: if the indices used for the contraction are incorrect  
    #     """
    #     if kraus_out_idx is None:
    #         kraus_out_idx = kraus_in_idx

    #     if not set(kraus_in_idx).issubset(range(ket.ndim)):
    #         raise ValueError("kraus_in_idx should be a subset of the ket indices.")

    #     ket = MMTensor(ket, axis_labels=[f"left_{i}" for i in range(ket.ndim)])

    #     # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    #     try:
    #         self.validate_contraction_indices(kraus_in_idx, kraus_out_idx, ket.ndim, "kraus")

    #         kraus = MMTensor(
    #             kr,
    #             axis_labels=(
    #             [f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx]
    #             )
    #         )

    #         # contract the operator with the ket.
    #         # now the leftover indices are in the order kraus_out_idx + uncontracted ket indices
    #         kraus_ket = kraus @ ket

    #         # sort kraus_ket.axis_labels by the int at the end of each label.
    #         # Each label is guaranteed to have a unique int at the end.
    #         new_axis_labels = sorted(kraus_ket.axis_labels, key=lambda x: int(x.split("_")[-1]))

    #         return kraus_ket.transpose(new_axis_labels).tensor
        
    #     except ValueError as e:
    #         raise e