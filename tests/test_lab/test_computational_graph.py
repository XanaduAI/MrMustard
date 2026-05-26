# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for computational graphs."""

import pytest
import rustworkx as rx

from mrmustard import math
from mrmustard.lab import (
    Attenuator,
    BSgate,
    CircuitComponent,
    Coherent,
    ComputationalGraph,
    Dgate,
    Number,
    SqueezedVacuum,
    TraceOut,
)
from mrmustard.physics.wires import ReprEnum


class TestComputationalGraph:
    r"""
    Tests ``ComputationalGraph`` objects.
    """

    def test_add_component(self):
        comp_graph = ComputationalGraph()
        component = CircuitComponent(name="test_node")
        idx = comp_graph.add_component(component, "test_node")
        assert idx == 0
        assert comp_graph.name_to_idx["test_node"] == 0
        assert comp_graph.uncontracted_wires["test_node"] == component.wires.standard_order
        assert isinstance(comp_graph.graph, rx.PyDiGraph)

        component2 = Coherent(mode=0, alpha=0.1)
        idx2 = comp_graph.add_component(component2, "coherent_state")
        assert idx2 == 1
        assert comp_graph.name_to_idx["coherent_state"] == 1
        assert comp_graph.uncontracted_wires["coherent_state"] == component2.wires.standard_order

    def test_add_components(self):
        comp_graph = ComputationalGraph()
        components = [CircuitComponent(name="test_node"), Coherent(mode=0, alpha=0.1)]
        names = ["test_node", "coherent_state"]
        idxs = comp_graph.add_components(components, names)
        assert idxs == [0, 1]
        assert comp_graph.name_to_idx["test_node"] == 0
        assert comp_graph.name_to_idx["coherent_state"] == 1

        with pytest.raises(ValueError, match="longer than"):
            comp_graph.add_components(components[:1], names)

    def test_add_wire(self):
        comp_graph = ComputationalGraph()
        coherent_state = Coherent(mode=0, alpha=0.1)
        dgate = Dgate(mode=0, alpha=0.1)
        comp_graph.add_components([coherent_state, dgate], ["coherent_state", "dgate"])

        with pytest.raises(ValueError, match="Invalid string"):
            comp_graph.add_wire("coherent_state[ok0]", "dgate")

        with pytest.raises(ValueError, match="Component"):
            comp_graph.add_wire("foo[ok0]", "dgate[ik0]")

        with pytest.raises(ValueError, match="Wire"):
            comp_graph.add_wire("coherent_state[ok2]", "dgate[ik0]")

        comp_graph.add_wire("coherent_state[ok0]", "dgate[ik0]")
        assert comp_graph.graph.has_edge(0, 1)

        # test mode > 10
        comp_graph = ComputationalGraph()
        coherent_state = Coherent(mode=10, alpha=0.1)
        dgate = Dgate(mode=10, alpha=0.1)
        comp_graph.add_components([coherent_state, dgate], ["coherent_state", "dgate"])
        comp_graph.add_wire("coherent_state[ok10]", "dgate[ik10]")
        assert comp_graph.graph.has_edge(0, 1)

    def test_add_wires(self):
        comp_graph = ComputationalGraph()
        coherent_state = Coherent(mode=0, alpha=0.1)
        dgate = Dgate(mode=0, alpha=0.1)
        comp_graph.add_components(
            [coherent_state, dgate, coherent_state.on(1), dgate.on(1)],
            ["coherent_state", "dgate", "coherent_state_1", "dgate_1"],
        )
        wires = [("coherent_state[ok0]", "dgate[ik0]"), ("coherent_state_1[ok1]", "dgate_1[ik1]")]
        comp_graph.add_wires(wires)
        assert comp_graph.graph.has_edge(0, 1)
        assert comp_graph.graph.has_edge(2, 3)

    def test_fock_config(self):
        shape0 = 4
        shape1 = 6
        output_shape = shape0 + shape1

        sv0 = SqueezedVacuum(0, r=0.1).to_fock(shape0)
        sv1 = SqueezedVacuum(1, r=0.2).to_fock(shape1)
        bs_gate = BSgate((0, 1), theta=0.1)
        num0 = Number(0, n=shape0 - 1).dual
        num1 = Number(1, n=shape1 - 1).dual

        # BS gate output wires
        comp_graph = ComputationalGraph()
        comp_graph.add_components([sv0, sv1, bs_gate], ["sv0", "sv1", "bs_gate"])
        comp_graph.add_wire("sv0[ok0]", "bs_gate[ik0]")
        comp_graph.add_wire("sv1[ok1]", "bs_gate[ik1]")

        assert comp_graph.mm_einsum_str == "a,b,cdab->cd"
        assert comp_graph.fock_config() == {
            0: shape0,
            1: shape1,
            2: output_shape,
            3: output_shape,
        }

        # BS gate input wires
        comp_graph = ComputationalGraph()
        comp_graph.add_components([bs_gate, num0, num1], ["bs_gate", "num0", "num1"])
        comp_graph.add_wire("bs_gate[ok0]", "num0[ik0]")
        comp_graph.add_wire("bs_gate[ok1]", "num1[ik1]")
        assert comp_graph.mm_einsum_str == "abcd,a,b->cd"
        assert comp_graph.fock_config() == {
            0: shape0,
            1: shape1,
            2: output_shape,
            3: output_shape,
        }

    def test_representation_config(self):
        coherent = Coherent(0, alpha=0.1)
        dgate = Dgate(0, alpha=0.1)

        comp_graph = ComputationalGraph()
        comp_graph.add_component(coherent, "coherent")
        comp_graph.add_component(dgate, "dgate")
        comp_graph.add_wire("coherent[ok0]", "dgate[ik0]")

        assert comp_graph.representation_config() == {
            "coherent": ReprEnum.BARGMANN,
            "dgate": ReprEnum.BARGMANN,
        }

        dgate_fock = dgate.to_fock()

        comp_graph = ComputationalGraph()
        comp_graph.add_component(coherent, "coherent")
        comp_graph.add_component(dgate_fock, "dgate_fock")
        comp_graph.add_wire("coherent[ok0]", "dgate_fock[ik0]")

        assert comp_graph.representation_config() == {
            "coherent": ReprEnum.BARGMANN,
            "dgate_fock": ReprEnum.FOCK,
        }

    def test_run(self):
        r = [0.1, 0.2]
        theta = 0.1
        output_loss = 0.3
        pnr_loss = 0.1
        transmissivity = [1 - output_loss, 1 - pnr_loss]

        state_0 = SqueezedVacuum(mode=0, r=r[0])
        state_1 = SqueezedVacuum(mode=1, r=r[1])
        bs_gate = BSgate(modes=(0, 1), theta=theta)
        attenuator_0 = Attenuator(mode=0, transmissivity=transmissivity[0])
        attenuator_1 = Attenuator(mode=1, transmissivity=transmissivity[1])
        pnr = Number(mode=1, n=1).dual

        state_0_adjoint = state_0.adjoint
        state_1_adjoint = state_1.adjoint
        bs_gate_adjoint = bs_gate.adjoint
        pnr_adjoint = pnr.adjoint

        components = [
            state_0,
            state_1,
            bs_gate,
            attenuator_0,
            attenuator_1,
            pnr,
            state_0_adjoint,
            state_1_adjoint,
            bs_gate_adjoint,
            pnr_adjoint,
        ]
        names = [
            "state_0",
            "state_1",
            "bs_gate",
            "attenuator_0",
            "attenuator_1",
            "pnr",
            "state_0_adjoint",
            "state_1_adjoint",
            "bs_gate_adjoint",
            "pnr_adjoint",
        ]
        wires = [
            ("state_0[ok0]", "bs_gate[ik0]"),
            ("state_1[ok1]", "bs_gate[ik1]"),
            ("bs_gate[ok0]", "attenuator_0[ik0]"),
            ("bs_gate[ok1]", "attenuator_1[ik1]"),
            ("state_0_adjoint[ob0]", "bs_gate_adjoint[ib0]"),
            ("state_1_adjoint[ob1]", "bs_gate_adjoint[ib1]"),
            ("attenuator_0[ib0]", "bs_gate_adjoint[ob0]"),
            ("attenuator_1[ib1]", "bs_gate_adjoint[ob1]"),
            ("attenuator_1[ok1]", "pnr[ik1]"),
            ("attenuator_1[ob1]", "pnr_adjoint[ib1]"),
        ]

        comp_graph = ComputationalGraph()
        comp_graph.add_components(components, names)
        comp_graph.add_wires(wires)

        with pytest.raises(RuntimeError, match=r"Graph is underspecified."):
            comp_graph.run()

        # trace out operation
        comp_graph.add_wire("attenuator_0[ok0]", "attenuator_0[ob0]")
        result = comp_graph.run()
        expected = (
            state_0 >> state_1 >> bs_gate >> attenuator_0 >> attenuator_1 >> TraceOut(0)
        ) >> pnr
        assert math.allclose(result, expected)

        # expectation value
        comp_graph = ComputationalGraph()
        comp_graph.add_components(components, names)
        comp_graph.add_wires(wires)

        dgate = Dgate(0, alpha=0.3)
        comp_graph.add_component(dgate, "dgate")
        comp_graph.add_wire("attenuator_0[ok0]", "dgate[ik0]")
        comp_graph.add_wire("dgate[ok0]", "attenuator_0[ob0]")

        result = comp_graph.run()
        expected = (
            state_0 >> state_1 >> bs_gate >> attenuator_0 >> attenuator_1 >> pnr
        ).expectation(dgate)
        assert math.allclose(
            result, expected, atol=1e-6
        )  # tol is due to rshift doing the expectation in Fock

    def test_standard_order(self):
        state_0 = SqueezedVacuum(mode=0, r=0.1)
        state_1 = SqueezedVacuum(mode=1, r=0.2)
        attenuator_0 = Attenuator(mode=0, transmissivity=0.3)
        attenuator_1 = Attenuator(mode=1, transmissivity=0.4)

        state_0_adjoint = state_0.adjoint
        state_1_adjoint = state_1.adjoint

        components = [
            state_0,
            state_1,
            attenuator_0,
            attenuator_1,
            state_0_adjoint,
            state_1_adjoint,
        ]
        names = [
            "state_0",
            "state_1",
            "attenuator_0",
            "attenuator_1",
            "state_0_adjoint",
            "state_1_adjoint",
        ]
        wires = [
            ("state_0[ok0]", "attenuator_0[ik0]"),
            ("state_1[ok1]", "attenuator_1[ik1]"),
            ("state_0_adjoint[ob0]", "attenuator_0[ib0]"),
            ("state_1_adjoint[ob1]", "attenuator_1[ib1]"),
        ]

        comp_graph = ComputationalGraph()
        comp_graph.add_components(components, names)
        comp_graph.add_wires(wires)

        assert comp_graph.mm_einsum_str == "a,b,cgda,ehfb,g,h->cedf"
        component = comp_graph.to_component()
        assert component == component.to_standard_order()

    def test_to_component(self):
        # Bargmann
        coherent = Coherent(0, alpha=0.1)
        dgate = Dgate(0, alpha=0.1)

        comp_graph = ComputationalGraph()
        comp_graph.add_component(coherent, "coherent")
        comp_graph.add_component(dgate, "dgate")
        comp_graph.add_wire("coherent[ok0]", "dgate[ik0]")

        component = comp_graph.to_component()
        expected = coherent >> dgate
        assert component == expected

        # Fock
        output_dim = dgate.auto_shape()[0]
        input_dim = coherent.auto_shape()[0]
        dgate_fock = dgate.to_fock((output_dim, input_dim))

        comp_graph = ComputationalGraph()
        comp_graph.add_component(coherent, "coherent")
        comp_graph.add_component(dgate_fock, "dgate_fock")
        comp_graph.add_wire("coherent[ok0]", "dgate_fock[ik0]")

        component = comp_graph.to_component()
        expected = coherent >> dgate_fock
        assert component == expected
