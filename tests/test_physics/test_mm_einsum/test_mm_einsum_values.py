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

"""Tests for mm_einsum values: numerical correctness by comparing with .contract() results."""

import pytest

from mrmustard import math
from mrmustard.lab import Attenuator, BSgate, GaussianKet, Rgate, SqueezedVacuum, TraceOut, Unitary
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum import mm_einsum


class TestMmEinsumValues:
    """Tests for mm_einsum numerical correctness by comparing with .contract() results."""

    def test_2mode_staircase_fock(self):
        """Test that mm_einsum works for a 2 mode staircase fock state."""
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        f1 = GaussianKet.random([1]).to_fock()
        res = mm_einsum(
            "x,y,zwxy,w->z",
            s0.ansatz,
            s1.ansatz,
            bs01.ansatz,
            f1.dual.ansatz,
            fock_dims={"x": 20, "y": 20, "z": 20, "w": f1.auto_shape()[0]},
        )
        assert isinstance(res, ArrayAnsatz)
        assert res == ((s1 >> s0 >> bs01).to_fock((20, 20)) >> f1.dual).ansatz

        with pytest.raises(ValueError):
            _ = mm_einsum(
                "x,y,xyzw,w->z",
                s0.ansatz,
                s1.ansatz,
                bs01.ansatz,
                f1.dual.ansatz,
                fock_dims={"x": 20, "y": 20},
            )

    def test_3mode_staircase_bargmann(self):
        """Test that mm_einsum works for a 3 mode staircase bargmann state."""
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2 = SqueezedVacuum(2, 0.3, 0.8)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        bs12 = BSgate((1, 2), 0.5, 0.2)
        f1 = GaussianKet.random([1]).to_fock(10)
        f2 = GaussianKet.random([2]).to_fock(10)
        res = mm_einsum(
            "x,y,z,abxy,cdbz,c,d->a",
            s0.ansatz,
            s1.ansatz,
            s2.ansatz,
            bs01.ansatz,
            bs12.ansatz,
            f1.dual.ansatz,
            f2.dual.ansatz,
            contraction_path=[(0, 3), (0, 1), (1, 2), (1, 2), (1, 2), (0, 1)],
            fock_dims={"a": 10, "b": 10, "c": 10, "d": 10},
        )
        assert isinstance(res, ArrayAnsatz)
        assert (
            res == (((s1 >> (s0 >> bs01)) >> (s2 >> bs12)).to_fock(10) >> f1.dual >> f2.dual).ansatz
        )

    def test_3mode_staircase_fock(self):
        """Test that mm_einsum works for a 3 mode staircase fock state."""
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2 = SqueezedVacuum(2, 0.3, 0.8)
        bs01 = BSgate((0, 1), 0.5, 0.2)
        bs12 = BSgate((1, 2), 0.5, 0.2)
        f1 = GaussianKet.random([1]).to_fock()
        f2 = GaussianKet.random([2]).to_fock()
        d1 = f1.auto_shape()[0]
        d2 = f2.auto_shape()[0]
        res = mm_einsum(
            "x,y,z,abxy,cdbz,c,d->a",
            s0.ansatz,
            s1.ansatz,
            s2.ansatz,
            bs01.ansatz,
            bs12.ansatz,
            f1.dual.ansatz,
            f2.dual.ansatz,
            fock_dims={
                "x": 20 + d1 + d2,
                "y": 20 + d1 + d2,
                "z": d1 + d2,
                "a": 20,
                "b": d1 + d2,
                "c": d1,
                "d": d2,
            },
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

    def test_diagonal_fock_operator(self):
        """Test that mm_einsum works for a diagonal fock operator."""
        R = Rgate(0, 0.5)
        f0 = GaussianKet.random([0]).to_fock()
        d = f0.auto_shape()[0]
        r = ArrayAnsatz(R.fock_array(d), batch_dims=0)
        res = mm_einsum(
            "i,ii->i",
            f0.ansatz,
            r,
        )
        assert isinstance(res, ArrayAnsatz)
        assert res == (f0 >> R).ansatz

    def test_multimode_with_batch(self):
        """Multimode with batch dims."""
        g0123 = GaussianKet.random([0, 1, 2, 3])
        g_batched = g0123 + g0123
        res = mm_einsum("Habcd,abcd->H", g_batched.ansatz, g0123.ansatz.conj)
        assert res.batch_shape == (2,)
        assert res == (g_batched.contract(g0123.dual)).ansatz

    def test_no_hilbert_wires_left_with_batch(self):
        """Broadcasting across batch dims with remaining array output."""
        s0 = SqueezedVacuum(0, 0.1, 0.4)
        s1 = SqueezedVacuum(1, 0.2, 0.7)
        s2 = SqueezedVacuum(2, [0.3, 0.2, 0.1], [0.8, 0.3, 0.5])
        bs01 = BSgate((0, 1), [0.5, 0.3], [0.2, 1.2])
        bs12 = BSgate((1, 2), 0.5, 0.2)
        g0 = GaussianKet.random([0])
        f1 = GaussianKet.random([1]).to_fock()
        f2 = GaussianKet.random([2]).to_fock()
        d1 = f1.auto_shape()[0]
        d2 = f2.auto_shape()[0]
        res = mm_einsum(
            "x,y,Hz,Wabxy,cdbz,c,d,a->(HW)",
            s0.ansatz,
            s1.ansatz,
            s2.ansatz,
            bs01.ansatz,
            bs12.ansatz,
            f1.dual.ansatz,
            f2.dual.ansatz,
            g0.dual.ansatz,
            fock_dims={
                "x": 20 + d1 + d2,
                "y": 20 + d1 + d2,
                "z": d1 + d2,
                "a": 20,
                "b": d1 + d2,
                "c": d1,
                "d": d2,
            },
        )
        assert res.array.shape[0] == 6

    def test_single_mode_fock(self):
        """Single-mode fock state contraction (arrays only)."""
        f0 = GaussianKet.random([0]).to_fock()
        res = mm_einsum(
            "x,x->",
            f0.ansatz,
            f0.ansatz.conj,
        )
        assert isinstance(res, ArrayAnsatz)
        assert math.allclose(res.scalar, f0 >> f0.dual)

    def test_single_mode_fock_leftover_index(self):
        """Single-mode fock with a leftover index (arrays)."""
        f0 = GaussianKet.random([0]).to_fock()
        f01 = GaussianKet.random([0, 1]).to_fock()
        res = mm_einsum(
            "x,xy->y",
            f0.ansatz,
            f01.ansatz.conj,
        )
        assert isinstance(res, ArrayAnsatz)
        assert res == (f0 >> f01.dual).ansatz

    def test_sublist_extension(self):
        """Tests that the sublist style can support >26 indices."""
        sv0 = SqueezedVacuum(0, 0.1)
        sv1 = SqueezedVacuum(1, 0.2)
        sv2 = SqueezedVacuum(2, 0.3)
        sv3 = SqueezedVacuum(3, 0.4)
        bs_12_first = BSgate((1, 2), theta=0.1)
        bs_01 = BSgate((0, 1), theta=0.2)
        bs_12_second = BSgate((1, 2), theta=0.3)
        rot_0 = Rgate(0, theta=0.1)
        att_0 = Attenuator(0, 0.9)
        att_1 = Attenuator(1, 0.9)
        att_2 = Attenuator(2, 0.9)

        res = mm_einsum(
            sv0.ansatz,
            [0],
            sv1.ansatz,
            [1],
            sv2.ansatz,
            [2],
            bs_01.ansatz,
            [3, 4, 0, 5],
            bs_12_first.ansatz,
            [5, 6, 1, 2],
            bs_12_second.ansatz,
            [7, 8, 4, 6],
            sv0.adjoint.ansatz,
            [9],
            sv1.adjoint.ansatz,
            [10],
            sv2.adjoint.ansatz,
            [11],
            bs_01.adjoint.ansatz,
            [12, 13, 9, 14],
            bs_12_first.adjoint.ansatz,
            [14, 15, 10, 11],
            bs_12_second.adjoint.ansatz,
            [16, 17, 13, 15],
            rot_0.ansatz,
            [18, 3],
            rot_0.adjoint.ansatz,
            [19, 12],
            att_0.ansatz,
            [20, 19, 21, 18],
            att_1.ansatz,
            [22, 16, 23, 7],
            att_2.ansatz,
            [24, 17, 25, 8],
            sv3.ansatz,
            [26],
            sv3.adjoint.ansatz,
            [27],
            [20, 22, 24, 27, 21, 23, 25, 26],
        )

        assert isinstance(res, PolyExpAnsatz)
        assert (
            res
            == (
                sv0
                >> sv1
                >> sv2
                >> sv3
                >> bs_12_first
                >> bs_01
                >> bs_12_second
                >> rot_0
                >> att_0
                >> att_1
                >> att_2
            ).ansatz
        )

    def test_trace_out(self):
        """Tests the trace out phase in mm_einsum."""
        # in Bargmann
        state_0 = SqueezedVacuum(mode=0, r=0.1)
        attenuator_0 = Attenuator(mode=0, transmissivity=0.7)
        state_0_adjoint = state_0.adjoint
        res = mm_einsum(
            "a,b,gagb->",
            state_0.ansatz,
            state_0_adjoint.ansatz,
            attenuator_0.ansatz,
        ).scalar
        expected = state_0 >> attenuator_0 >> TraceOut((0,))
        assert math.allclose(res, expected)

        # in Fock
        state_0_fock = state_0.to_fock()
        state_0_fock_adjoint = state_0_fock.adjoint
        expected = state_0_fock >> attenuator_0 >> TraceOut((0,))
        res = mm_einsum(
            "a,b,gagb->",
            state_0_fock.ansatz,
            state_0_fock_adjoint.ansatz,
            attenuator_0.ansatz,
            fock_dims={"a": 4, "b": 4},
        ).scalar
        assert math.allclose(res, expected)

    def test_with_leftover_indices(self):
        """Two states through a beamsplitter -> 2-mode array."""
        g0 = GaussianKet.random([0])
        u01 = Unitary.random([0, 1])
        dz = 20
        res = mm_einsum(
            "x,y,zwxy->zw",
            g0.ansatz,
            g0.ansatz,
            u01.ansatz,
            fock_dims={"x": dz, "y": dz, "z": dz, "w": dz},
        )
        assert isinstance(res, ArrayAnsatz)
        f0 = g0.to_fock(dz)
        f01 = u01.to_fock(dz)
        expected = (f0 >> (f0.on(1) >> f01)).ansatz
        assert res == expected

    def test_with_two_multimode_gaussians(self):
        """Two multimode gaussians contracted to a scalar (array)."""
        g0123 = GaussianKet.random([0, 1, 2, 3])
        res = mm_einsum(
            "abcd,abcd->",
            g0123.ansatz,
            g0123.ansatz.conj,
            contraction_path=[(0, 1)],
            fock_dims={"a": 3, "b": 4, "c": 5, "d": 6},
        )
        assert isinstance(res, ArrayAnsatz)
        f0123 = g0123.to_fock((3, 4, 5, 6))
        assert math.allclose(res.scalar, f0123 >> f0123.dual)

    def test_with_two_single_mode_gaussians(self):
        """Two single-mode gaussians contracted to a scalar (array)."""
        g0 = GaussianKet.random([0])
        res = mm_einsum(
            "x,x->",
            g0.ansatz,
            g0.ansatz.conj,
            contraction_path=[(0, 1)],
            fock_dims={"x": 1},
        )
        assert isinstance(res, ArrayAnsatz)
        f0 = g0.to_fock(1)
        assert math.allclose(res.scalar, f0 >> f0.dual)
