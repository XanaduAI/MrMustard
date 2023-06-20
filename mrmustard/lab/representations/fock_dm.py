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

class FockDM(Fock):

    def __init__(self, array):
        super().__init__(array=array)
        self.num_modes = len(self.array.shape) // 2
        self.cutoffs = self.array.shape


    def purity(self) -> float:
        dm = self.data.array
        cutoffs = self.data.cutoffs
        d = int(np.prod(cutoffs))  # combined cutoffs in all modes
        dm = math.reshape(dm, (d, d))
        dm = dm / math.trace(dm)  # assumes all nonzero values are included in the density matrix
        return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


    def norm(self) -> float:
        r""" The norm. (:math:`|amp|^2` for ``dm``). """
        return math.sum(math.all_diagonals(self.data.array, real = True))


    def probability(self) -> Tensor:
        return math.all_diagonals(self.data.array, real = True) #TODO: cutoffs adjust
    
    # NOTE : this is for transformations!
    # def apply_kraus_to_dm(kr:np.array, dm:np.array, 
    #                       kraus_in_idx:List[int], kraus_out_idx:Union[List[int], None]=None
    #                       ) -> Tensor:
    #     r""" Applies a kraus operator to a density matrix.
        
    #     Assumes that the density matrix is indexed as left_1, ..., left_n, right_1, ..., right_n.

    #     The kraus operator has indices that contract with the density matrix (kraus_in_idx) and 
    #     indices that are leftover (kraus_out_idx).
    #     `kraus` will contract from the left and from the right with the density matrix. For right 
    #     contraction the kraus op is conjugated.

    #     Args:
    #         kr: the operator to be applied
    #         dm: the density matrix to which the operator is applied
    #         kraus_in_idx: the indices (counting from 0) of the kraus operator that contract with 
    #                       the density matrix
    #         kraus_out_idx: the indices (counting from 0) of the kraus operator that are leftover

    #     Returns:
    #         The resulting density matrix

    #     Raises:
    #         ValueError: if the indices used for the contraction are incorrect 
    #     """
    #     if kraus_out_idx is None:
    #         kraus_out_idx = kraus_in_idx

    #     if not set(kraus_in_idx).issubset(range(dm.ndim // 2)):
    #         raise ValueError("kraus_in_idx should be a subset of the density matrix indices.")

    #     dm = MMTensor(
    #             dm,
    #             axis_labels=[f"left_{i}" for i in range(dm.ndim // 2)]
    #             + [f"right_{i}" for i in range(dm.ndim // 2)],
    #         )

    #     # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    #     self.validate_contraction_indices(kraus_in_idx, kraus_out_idx, dm.ndim // 2)

    #     kraus = MMTensor(
    #         kr,
    #         axis_labels=(
    #         [f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx]
    #         )
    #         )
    #     kraus_conj = MMTensor(
    #         math.conj(kraus.tensor),
    #         axis_labels=[f"out_right_{i}" for i in kraus_out_idx]
    #         + [f"right_{i}" for i in kraus_in_idx],
    #     )

    #     # contract the kraus operator with the density matrix from the left and from the right.
    #     k_dm_k = kraus @ dm @ kraus_conj
    #     # now the leftover indices are in the order:
    #     # out_left_idx + uncontracted left indices + uncontracted right indices + out_right_idx

    #     # sort k_dm_k.axis_labels by the int at the end of each label, first left, then right
    #     N = k_dm_k.tensor.ndim // 2
    #     left = sorted(k_dm_k.axis_labels[:N], key=lambda x: int(x.split("_")[-1]))
    #     right = sorted(k_dm_k.axis_labels[N:], key=lambda x: int(x.split("_")[-1]))

    #     return k_dm_k.transpose(left + right).tensor