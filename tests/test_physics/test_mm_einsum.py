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


# unit tests for mm_einsum


class TestMmEinsum:
    g0 = Ket.random([0])
    g1 = Ket.random([0])
    g0123 = Ket.random([0, 1, 2, 3])
    u01 = Unitary.random([0, 1])
    f0 = Ket.random([0]).to_fock()
    f1 = Ket.random([0]).to_fock()

    def test_with_two_single_mode_gaussians(self):
        """Test that mm_einsum works for two gaussians."""
        res = mm_einsum(
            self.g0.ansatz,
            [0],
            self.g1.ansatz.conj,
            [0],
            output=[],
            contraction_order=[(0, 1)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert np.isclose(res.scalar, self.g0 >> self.g1.dual)

    def test_with_two_multimode_gaussians(self):
        """Test that mm_einsum works for two multimode gaussians."""
        res = mm_einsum(
            self.g0123.ansatz,
            [0, 1, 2, 3],
            self.g0123.ansatz.conj,
            [0, 1, 2, 3],
            output=[],
            contraction_order=[(0, 1)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert np.isclose(res.scalar, self.g0123 >> self.g0123.dual)

    def test_with_leftover_indices(self):
        """Test that mm_einsum works for two multimode gaussians."""
        res = mm_einsum(
            self.g0.ansatz,
            [0],
            self.g0.ansatz,
            [1],
            self.u01.ansatz,
            [2, 3, 0, 1],
            output=[2, 3],
            contraction_order=[(0, 2), (1, 2)],
            shapes={0},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res == (self.g0 >> (self.g0.on(1) >> self.u01)).ansatz

    def test_single_mode_with_batch(self):
        """Test that mm_einsum works for a single mode with batch dimensions."""
        res = mm_einsum(
            (self.g0.ansatz + self.g0.ansatz),
            ["hello", 0],
            self.g1.ansatz.conj,
            [0],
            output=["hello"],
            contraction_order=[(0, 1)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (2,)
        assert np.allclose(res.scalar, self.g0.contract(self.g1.dual, mode="zip").ansatz.scalar)

    def test_multimode_with_batch(self):
        """Test that mm_einsum works for a multimode with batch dimensions."""
        res = mm_einsum(
            (self.g0123.ansatz + self.g0123.ansatz),
            ["hello", 0, 1, 2, 3],
            self.g0123.ansatz.conj,
            [0, 1, 2, 3],
            output=["hello"],
            contraction_order=[(0, 1)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (2,)
        assert np.allclose(
            res.scalar, self.g0123.contract(self.g0123.dual, mode="zip").ansatz.scalar
        )

    def test_single_mode_fock(self):
        """Test that mm_einsum works for a single mode fock state."""
        res = mm_einsum(
            self.f0.ansatz,
            [0],
            self.f0.ansatz.conj,
            [0],
            output=[],
            contraction_order=[(0, 1)],
            shapes={0: 20},
        )
        assert isinstance(res, ArrayAnsatz)
        assert np.isclose(res.scalar, self.f0 >> self.f0.dual)

    def test_single_mode_fock_with_batch(self):
        """Test that mm_einsum works for a single mode fock state with batch dimensions."""
        batched = ArrayAnsatz(np.array([self.f0.fock_array(), self.f0.fock_array()]), batch_dims=1)
        res = mm_einsum(
            batched,
            ["hello", 0],
            self.f1.ansatz.conj,
            [0],
            output=["hello"],
            contraction_order=[(0, 1)],
            shapes={0: 20},
        )
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert np.allclose(res.scalar, self.f0.contract(self.f1.dual, mode="zip").ansatz.scalar)

    def test_single_mode_with_complex_batch(self):
        A = np.random.random((4, 3, 2, 2))
        b = np.random.random((4, 3, 2))
        c = np.random.random((4, 3))
        g0 = PolyExpAnsatz(A, b, c)

        res = mm_einsum(
            g0,
            ["hello", "world", 0, 1],
            g0,
            ["hello", "world", 0, 1],
            output=["hello"],
            contraction_order=[(0, 1)],
            shapes={},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (3,)
