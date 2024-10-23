# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Representation class."""

# pylint: disable=missing-function-docstring

import pytest

from mrmustard import math

from mrmustard.physics.representations import Representation, RepEnum
from mrmustard.physics.wires import Wires
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.triples import displacement_gate_Abc, bargmann_to_quadrature_Abc

from ..random import Abc_triple


class TestRepresentation:
    r"""
    Tests for the Representation class.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.fixture
    def d_gate_rep(self):
        ansatz = PolyExpAnsatz.from_function(fn=displacement_gate_Abc, x=0.1, y=0.1)
        wires = Wires((), (), set([0]), set([0]))
        return Representation(ansatz, wires)

    @pytest.fixture
    def btoq_rep(self):
        ansatz = PolyExpAnsatz.from_function(fn=bargmann_to_quadrature_Abc, n_modes=1, phi=0.2)
        wires = Wires((), (), set([0]), set([0]))
        idx_reps = {}
        for i in wires.input.indices:
            idx_reps[i] = (RepEnum.BARGMANN, None, tuple())
        for i in wires.output.indices:
            idx_reps[i] = (RepEnum.QUADRATURE, float(0.2), tuple())
        return Representation(ansatz, wires, idx_reps)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init(self, triple):
        empty_rep = Representation()
        assert empty_rep.ansatz is None
        assert empty_rep.wires == Wires()
        assert empty_rep._idx_reps == {}

        ansatz = PolyExpAnsatz(*triple)
        wires = Wires(set([0, 1]))
        rep = Representation(ansatz, wires)
        assert rep.ansatz == ansatz
        assert rep.wires == wires
        assert rep._idx_reps == dict.fromkeys(
            wires.indices, (RepEnum.from_ansatz(ansatz), None, tuple())
        )

    @pytest.mark.parametrize("triple", [Abc_n2])
    def test_adjoint_idx_reps(self, triple):
        ansatz = PolyExpAnsatz(*triple)
        wires = Wires(modes_out_bra=set([0]), modes_out_ket=set([0]))
        idx_reps = {0: (RepEnum.BARGMANN, None, tuple()), 1: (RepEnum.QUADRATURE, 0.1, tuple())}
        rep = Representation(ansatz, wires, idx_reps)
        adj_rep = rep.adjoint
        assert adj_rep._idx_reps == {
            1: (RepEnum.BARGMANN, None, tuple()),
            0: (RepEnum.QUADRATURE, 0.1, tuple()),
        }

    @pytest.mark.parametrize("triple", [Abc_n2])
    def test_dual_idx_reps(self, triple):
        ansatz = PolyExpAnsatz(*triple)
        wires = Wires(modes_out_bra=set([0]), modes_in_bra=set([0]))
        idx_reps = {0: (RepEnum.BARGMANN, None, tuple()), 1: (RepEnum.QUADRATURE, 0.1, tuple())}
        rep = Representation(ansatz, wires, idx_reps)
        adj_rep = rep.dual
        assert adj_rep._idx_reps == {
            1: (RepEnum.BARGMANN, None, tuple()),
            0: (RepEnum.QUADRATURE, 0.1, tuple()),
        }

    def test_matmul_btoq(self, d_gate_rep, btoq_rep):
        q_dgate = d_gate_rep @ btoq_rep
        assert q_dgate._idx_reps == {
            0: (RepEnum.QUADRATURE, 0.2, ()),
            1: (RepEnum.BARGMANN, None, ()),
        }

    def test_to_bargmann(self, d_gate_rep):
        d_fock = d_gate_rep.to_fock(shape=(4, 6))
        d_barg = d_fock.to_bargmann()
        assert d_fock.ansatz._original_abc_data == d_gate_rep.ansatz.triple
        assert d_barg == d_gate_rep
        assert all([k[0] == RepEnum.BARGMANN for k in d_barg._idx_reps.values()])

    def test_to_fock(self, d_gate_rep):
        d_fock = d_gate_rep.to_fock(shape=(4, 6))
        assert d_fock.ansatz == ArrayAnsatz(
            math.hermite_renormalized(*displacement_gate_Abc(x=0.1, y=0.1), shape=(4, 6))
        )
        assert all([k[0] == RepEnum.FOCK for k in d_fock._idx_reps.values()])
