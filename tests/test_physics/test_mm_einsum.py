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
from mrmustard import math
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
            fock_dims={0: 20},
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
            fock_dims={},
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
            fock_dims={},
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
            fock_dims={},
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
            fock_dims={},
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
            fock_dims={0: 20},
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
            fock_dims={0: 20},
        )
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert np.allclose(res.scalar, self.f0.contract(self.f1.dual, mode="zip").ansatz.scalar)

    def test_single_mode_fock_with_double_batch(self):
        array1 = np.random.random((3, 4, 5, 6))
        array2 = np.random.random((3, 5, 6))
        f1 = ArrayAnsatz(array1, batch_dims=2)
        f2 = ArrayAnsatz(array2, batch_dims=1)

        res = mm_einsum(
            f1,
            ["hello", "world", 0, 1],
            f2,
            ["hello", 0, 1],
            output=["world"],
            contraction_order=[(0, 1)],
            fock_dims={},
        )
        assert isinstance(res, ArrayAnsatz)

    def test_fock_to_bargmann_because_of_zero_fock_dim(self):
        res = mm_einsum(
            self.g0.ansatz,
            [0],
            self.f0.ansatz.conj,
            [0],
            output=[],
            contraction_order=[(0, 1)],
            fock_dims={0: 0},
        )
        assert isinstance(res, PolyExpAnsatz)

    def test_2mode_staircase_fock(self):
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        f1 = Ket.random([1]).to_fock()
        res = mm_einsum(
            s0.ansatz,
            [0],
            s1.ansatz,
            [1],
            bs01.ansatz,
            [2, 3, 0, 1],
            f1.dual.ansatz,
            [3],
            output=[2],
            contraction_order=[(0, 2), (1, 2), (2, 3)],
            fock_dims={0: 0, 1: 0, 2: 20, 3: f1.auto_shape()[0]},
        )
        assert isinstance(res, ArrayAnsatz)
        assert res == ((s1 >> s0 >> bs01).to_fock((20, 20)) >> f1.dual).ansatz

    def test_2mode_staircase_bargmann(self):
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        f1 = Ket.random([1]).to_fock()
        res = mm_einsum(
            s0.ansatz,
            [0],
            s1.ansatz,
            [1],
            bs01.ansatz,
            [2, 3, 0, 1],
            f1.dual.ansatz,
            [3],
            output=[2],
            contraction_order=[(0, 2), (1, 2), (2, 3)],
            fock_dims={0: 0, 1: 0, 2: 0, 3: 0},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert res == ((s1 >> s0 >> bs01) >> f1.dual.to_bargmann()).ansatz

    def test_3mode_staircase_fock(self):
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2 = SqueezedVacuum(2, 0.3, 0.8)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        bs12 = BSgate((1, 2), 0.5, 0.2)
        f1 = Ket.random([1]).to_fock()
        f2 = Ket.random([2]).to_fock()
        d1 = f1.auto_shape()[0]
        d2 = f2.auto_shape()[0]
        res = mm_einsum(
            s0.ansatz,
            [0],
            s1.ansatz,
            [1],
            s2.ansatz,
            [2],
            bs01.ansatz,
            [3, 4, 0, 1],
            bs12.ansatz,
            [5, 6, 4, 2],
            f1.dual.ansatz,
            [5],
            f2.dual.ansatz,
            [6],
            output=[3],
            contraction_order=[(0, 3), (1, 3), (2, 4), (4, 5), (4, 6), (3, 4)],
            fock_dims={0: 0, 1: 0, 2: 0, 3: 20, 4: d1 + d2, 5: d1, 6: d2},
        )
        assert isinstance(res, ArrayAnsatz)
        assert (
            res
            == (
                ((s1 >> (s0 >> bs01)) >> (s2 >> bs12)).to_fock((20, d1 + d2, d1 + d2))
                >> f1.dual
                >> f2.dual
            ).ansatz
        )

    def test_3mode_staircase_bargmann(self):
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2 = SqueezedVacuum(2, 0.3, 0.8)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        bs12 = BSgate((1, 2), 0.5, 0.2)
        f1 = Ket.random([1]).to_fock()
        f2 = Ket.random([2]).to_fock()
        res = mm_einsum(
            s0.ansatz,
            [0],
            s1.ansatz,
            [1],
            s2.ansatz,
            [2],
            bs01.ansatz,
            [3, 4, 0, 1],
            bs12.ansatz,
            [5, 6, 4, 2],
            f1.dual.ansatz,
            [5],
            f2.dual.ansatz,
            [6],
            output=[3],
            contraction_order=[(0, 3), (1, 3), (2, 4), (4, 6), (4, 5), (3, 4)],
            fock_dims={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        )
        assert isinstance(res, PolyExpAnsatz)
        assert (
            res
            == (
                ((s1 >> (s0 >> bs01)) >> (s2 >> bs12))
                >> f1.dual.to_bargmann()
                >> f2.dual.to_bargmann()
            ).ansatz
        )

    def test_no_hilbert_wires_left_with_batch(self):
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2_0 = SqueezedVacuum(2, 0.3, 0.8)
        s2_1 = SqueezedVacuum(2, 0.2, 0.3)
        s2_2 = SqueezedVacuum(2, 0.1, 0.5)
        bs01_0 = BSgate((0, 1), 0.5, 0.2)
        bs01_1 = BSgate((0, 1), 0.3, 1.2)
        bs12 = BSgate((1, 2), 0.5, 0.2)
        g0 = Ket.random([0])
        f1 = Ket.random([1]).to_fock()
        f2 = Ket.random([2]).to_fock()
        d1 = f1.auto_shape()[0]
        d2 = f2.auto_shape()[0]
        res = mm_einsum(
            s0.ansatz,
            [0],
            s1.ansatz,
            [1],
            (s2_0 + s2_1 + s2_2).ansatz,
            ["hello", 2],
            (bs01_0 + bs01_1).ansatz,
            ["world", 3, 4, 0, 1],
            bs12.ansatz,
            [5, 6, 4, 2],
            f1.dual.ansatz,
            [5],
            f2.dual.ansatz,
            [6],
            g0.dual.ansatz,
            [3],
            output=["hello", "world"],
            contraction_order=[(3, 7), (0, 3), (1, 3), (3, 4), (2, 4), (4, 5), (4, 6)],
            fock_dims={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: d1, 6: d2},
        )

        assert np.allclose(
            res.array[0, 0],
            (s1 >> (s0 >> bs01_0) >> (s2_0 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
        assert np.allclose(
            res.array[1, 0],
            (s1 >> (s0 >> bs01_0) >> (s2_1 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
        assert np.allclose(
            res.array[2, 0],
            (s1 >> (s0 >> bs01_0) >> (s2_2 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
        assert np.allclose(
            res.array[0, 1],
            (s1 >> (s0 >> bs01_1) >> (s2_0 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
        assert np.allclose(
            res.array[1, 1],
            (s1 >> (s0 >> bs01_1) >> (s2_1 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
        assert np.allclose(
            res.array[2, 1],
            (s1 >> (s0 >> bs01_1) >> (s2_2 >> bs12)) >> g0.dual >> f1.dual >> f2.dual,
        )
