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
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.representations import Representation
from mrmustard.physics.triples import bargmann_to_quadrature_Abc, displacement_gate_Abc
from mrmustard.physics.wires import ReprEnum, Wires

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
        wires = Wires(set(), set(), {0}, {0})
        return Representation(ansatz, wires)

    @pytest.fixture
    def btoq_rep(self):
        ansatz = PolyExpAnsatz.from_function(fn=bargmann_to_quadrature_Abc, n_modes=1, phi=0.2)
        wires = Wires(set(), set(), {0}, {0})
        for w in wires.output:
            w.repr = ReprEnum.QUADRATURE
            w.repr_params = [0.2]
        return Representation(ansatz, wires)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init(self, triple):
        empty_rep = Representation()
        assert empty_rep.ansatz is None
        assert empty_rep.wires == Wires()

        ansatz = PolyExpAnsatz(*triple)
        wires = Wires({0, 1})
        rep = Representation(ansatz, wires)
        assert rep.ansatz == ansatz
        assert rep.wires == wires

    def test_matmul_btoq(self, d_gate_rep, btoq_rep):
        q_dgate = d_gate_rep @ btoq_rep
        for w in q_dgate.wires.input.wires:
            assert w.repr == ReprEnum.BARGMANN
        for w in q_dgate.wires.output.wires:
            assert w.repr == ReprEnum.QUADRATURE
            assert w.repr_params == [0.2]

    def test_to_bargmann(self, d_gate_rep):
        d_fock = d_gate_rep.to_fock(shape=(4, 6))
        d_barg = d_fock.to_bargmann()
        assert d_fock.ansatz._original_abc_data == d_gate_rep.ansatz.triple
        assert d_barg == d_gate_rep
        for w in d_barg.wires.wires:
            assert w.repr == ReprEnum.BARGMANN

    def test_to_fock(self, d_gate_rep):
        d_fock = d_gate_rep.to_fock(shape=(4, 6))
        assert d_fock.ansatz == ArrayAnsatz(
            math.hermite_renormalized(*displacement_gate_Abc(x=0.1, y=0.1), shape=(4, 6))
        )
        for w in d_fock.wires.wires:
            assert w.repr == ReprEnum.FOCK
