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
# from math.mmtensor import MMTensor
from mrmustard.lab.representations.fock import Fock
from mrmustard.lab.representations.data.array_data import ArrayData
from mrmustard.typing import Scalar, Tensor, RealVector
from mrmustard.lab.representations.fock import validate_contraction_indices

math = Math()

class FockDM(Fock):

    def __init__(self, array):
        super().__init__(array=array)
        self.num_modes = len(self.array.shape) // 2 #TODO: BATCH ISSUE?


    def purity(self):
        r"""Returns the purity of a density matrix."""
        dm = self.data.array
        cutoffs = self.data.cutoffs
        d = int(np.prod(cutoffs))  # combined cutoffs in all modes
        dm = math.reshape(dm, (d, d))
        dm = dm / math.trace(dm)  # assumes all nonzero values are included in the density matrix
        return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


    def norm(self):
        r"""
        Returns the norm. (:math:`|amp|^2` for ``dm``)
        """
        return math.sum(math.all_diagonals(self.data.array, real = True))


    def probability(self) -> Tensor: 
        r"""Extracts the diagonals of a density matrix.

        Args:
            dm: the density matrix

        Returns:
            Tensor: the probabilities vector
        """
        #TODO: cutoffs adjust
        return math.all_diagonals(self.data.array, real = True)
    

    def apply_kraus_to_dm(kraus, dm, kraus_in_idx, kraus_out_idx=None):
        r"""Applies a kraus operator to a density matrix.
        It assumes that the density matrix is indexed as left_1, ..., left_n, right_1, ..., right_n.

        The kraus operator has indices that contract with the density matrix (kraus_in_idx) and indices that are leftover (kraus_out_idx).
        `kraus` will contract from the left and from the right with the density matrix. For right contraction the kraus op is conjugated.

        Args:
            kraus (array): the operator to be applied
            dm (array): the density matrix to which the operator is applied
            kraus_in_idx (list of ints): the indices (counting from 0) of the kraus operator that contract with the density matrix
            kraus_out_idx (list of ints): the indices (counting from 0) of the kraus operator that are leftover (default None, in which case kraus_out_idx = kraus_in_idx)

        Returns:
            array: the resulting density matrix
        """
        if kraus_out_idx is None:
            kraus_out_idx = kraus_in_idx

        if not set(kraus_in_idx).issubset(range(dm.ndim // 2)):
            raise ValueError("kraus_in_idx should be a subset of the density matrix indices.")

        # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
        validate_contraction_indices(kraus_in_idx, kraus_out_idx, dm.ndim // 2, "kraus")

        dm = MMTensor(
            dm,
            axis_labels=[f"left_{i}" for i in range(dm.ndim // 2)]
            + [f"right_{i}" for i in range(dm.ndim // 2)],
        )
        kraus = MMTensor(
            kraus,
            axis_labels=[f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx],
        )
        kraus_conj = MMTensor(
            math.conj(kraus.tensor),
            axis_labels=[f"out_right_{i}" for i in kraus_out_idx]
            + [f"right_{i}" for i in kraus_in_idx],
        )

        # contract the kraus operator with the density matrix from the left and from the right.
        k_dm_k = kraus @ dm @ kraus_conj
        # now the leftover indices are in the order:
        # out_left_idx + uncontracted left indices + uncontracted right indices + out_right_idx

        # sort k_dm_k.axis_labels by the int at the end of each label, first left, then right
        N = k_dm_k.tensor.ndim // 2
        left = sorted(k_dm_k.axis_labels[:N], key=lambda x: int(x.split("_")[-1]))
        right = sorted(k_dm_k.axis_labels[N:], key=lambda x: int(x.split("_")[-1]))

        return k_dm_k.transpose(left + right).tensor