# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the mm_einsum function."""

import numpy as np
from mrmustard.lab_dev import *
from mrmustard.physics.mm_einsum import mm_einsum
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz


class TestMmEinsum:
    g0 = Ket.random([0])
    g0123 = Ket.random([0, 1, 2, 3])
    u01 = Unitary.random([0, 1])
    f0 = Ket.random([0]).to_fock()

    def test_mm_einsum_with_two_gaussians(self):
        """Test that mm_einsum works for two gaussians."""
        res = mm_einsum(
            self.g0.ansatz, [0], self.g0.ansatz.conj(), [0], [], contraction_order=[(0,)], shapes={}
        )
        assert isinstance(res, PolyExpAnsatz)
        assert np.isclose(res.scalar, 1)

    def test_mm_einsum_with_two_multimode_gaussians(self):
        """Test that mm_einsum works for two multimode gaussians."""
        res = mm_einsum(
            self.g0123.ansatz,
            [0, 1, 2, 3],
            self.g0123.ansatz.conj(),
            [0, 1, 2, 3],
            [],
            contraction_order=[(0, 1, 2, 3)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert np.isclose(res.scalar, 1)

    def test_mm_einsum_with_leftover_indices(self):
        """Test that mm_einsum works for two multimode gaussians."""
        res = mm_einsum(
            self.g0.ansatz,
            [0],
            self.g0.ansatz,
            [1],
            self.u01.ansatz,
            [2, 3, 0, 1],
            [2, 3],
            contraction_order=[(0,), (1,)],
            shapes={0},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res == (self.g0 >> (self.g0.on(1) >> self.u01)).ansatz
