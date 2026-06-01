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

"""Tests for circuits."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab import (
    Attenuator,
    BSgate,
    Circuit,
    CircuitComponent,
    Coherent,
    CXgate,
    CZgate,
    Dgate,
    GaussianDM,
    Interferometer,
    MZgate,
    Number,
    Sgate,
    SqueezedVacuum,
    TraceOut,
    Vacuum,
)
from mrmustard.physics.wires import ReprEnum


class TestCircuit:
    r"""
    Tests the ``Circuit`` class.
    """

    def test_init(self):
        empty_circuit = Circuit([])
        assert empty_circuit.components == []

        components = [
            Coherent(0, 0.1, name="s0"),
            Coherent(1, 0.2, name="s1"),
            BSgate((0, 1), 0.3, name="bs"),
            Number(1, n=2, name="pnr").dual,
        ]
        circuit = Circuit(components=components)
        assert circuit.components == components

        with pytest.raises(ValueError, match=r"must be a `State`"):
            circuit = Circuit([BSgate((0, 1), 0.3), Coherent(0, 0.1)])

        with pytest.raises(ValueError, match="Duplicate component name"):
            circuit = Circuit([Coherent(0, 0.1), Coherent(1, 0.2), BSgate((0, 1), 0.3)])

    def test_expectation(self):
        expectation_operator = Dgate(0, 0.2 + 0.1j)

        dummy_rho = GaussianDM.random((0, 1), name="dm")
        dgate0 = Dgate(0, alpha=0.1, name="d0")
        dgate1 = Dgate(1, alpha=0.1, name="d1")
        components = [dummy_rho, dgate0, dgate1]
        circuit = Circuit(components)
        assert math.allclose(
            circuit.expectation(expectation_operator),
            (dummy_rho >> dgate0 >> dgate1).expectation(expectation_operator),
        )

        state_0 = SqueezedVacuum(mode=0, r=0.1, name="s0")
        state_1 = SqueezedVacuum(mode=1, r=0.2, name="s1")
        bs_gate = BSgate(modes=(0, 1), theta=np.pi, name="bs")
        attenuator_0 = Attenuator(mode=0, transmissivity=0.11, name="att0")
        attenuator_1 = Attenuator(mode=1, transmissivity=0.02, name="att1")
        pnr = Number(mode=1, n=1, name="pnr").dual
        components = [state_0, state_1, bs_gate, attenuator_0, attenuator_1, pnr]
        circuit = Circuit(components)
        assert math.allclose(
            circuit.expectation(expectation_operator),
            (
                (state_0 >> state_1 >> bs_gate >> attenuator_0 >> attenuator_1).to_fock(2) >> pnr
            ).expectation(expectation_operator),
            atol=1e-6,
        )

    def test_repr(self):
        vac01 = Vacuum((0, 1), name="vac01")
        vac2 = Vacuum(2, name="vac2")
        vac012 = Vacuum((0, 1, 2), name="vac012")
        s0 = Sgate(0, r=0.0, phi=2.0, name="s0")
        s1 = Sgate(1, r=1.0, phi=3.0, name="s1")
        bs01 = BSgate((0, 1), name="bs01")
        bs12 = BSgate((1, 2), name="bs12")
        n0 = Number(0, n=3, name="n0")
        n1 = Number(1, n=3, name="n1")
        n2 = Number(2, n=3, name="n2")
        cx = CXgate((0, 1), 0.1, name="cx")
        cz = CZgate((0, 1), 0.1, name="cz")
        mz = MZgate((0, 1), 0.2, 0.1, name="mz")
        cc = CircuitComponent(ansatz_factory=bs01.ansatz_factory, wires=bs01.wires, name="my_cc")

        assert repr(Circuit([])) == ""

        circ1 = Circuit([vac012])
        r1 = ""
        r1 += "\nmode 0:     ◖Vac◗"
        r1 += "\nmode 1:     ◖Vac◗"
        r1 += "\nmode 2:     ◖Vac◗"
        assert repr(circ1) == r1 + "\n\n"

        circ2 = Circuit([vac012, s0, s1, bs01, bs12, cc, n0.dual, n1.dual])
        r2 = ""
        r2 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC──|3)=(3,)"
        r2 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC──|3)=(3,)"
        r2 += "\nmode 2:     ◖Vac◗────────────────────────────╰BS(0.0,0.0)──────────────"
        assert repr(circ2) == r2 + "\n\n"

        circ3 = Circuit([vac01, s0, s1, vac2, bs01, bs12, n2.dual, cc, n0.dual, n1.dual])
        r3 = ""
        r3 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC────────|3)=(3,)"
        r3 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC────────|3)=(3,)"
        r3 += "\nmode 2:            ◖Vac◗─────────────────────╰BS(0.0,0.0)──|3)=(3,)          "
        assert repr(circ3) == r3 + "\n\n"

        circ4 = Circuit([vac01, s0, s1, bs01, cx, cz, mz, n0.dual, n1.dual])
        r4 = ""
        r4 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•────────────╭•─────────╭•─────────╭•────────────|3)=(3,)"
        r4 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╰CX(0.1,)──╰CZ(0.1,)──╰MZ(0.2,0.1)──|3)=(3,)"
        assert repr(circ4) == r4 + "\n\n"

    def test_repr_issue_344(self):
        r"""
        Tests the bug reported in GH issue #344.
        https://github.com/XanaduAI/MrMustard/issues/344
        """
        circ1 = Circuit(
            [
                Vacuum((0, 1), name="vac01"),
                Sgate(0, 1.0, 2.0, name="s0"),
                Sgate(1, -1.0, -2.0, name="s1"),
            ]
        )
        r1 = ""
        r1 += "\nmode 0:     ◖Vac◗──S(1.0,2.0)──"
        r1 += "\nmode 1:     ◖Vac◗──S(-1.0,-2.0)"
        r1 += "\n\n"
        assert repr(circ1) == r1

    def test_run(self):
        # GBS example
        state_0 = SqueezedVacuum(mode=0, r=0.1, name="s0")
        state_1 = SqueezedVacuum(mode=1, r=0.2, name="s1")
        bs_gate = BSgate(modes=(0, 1), theta=np.pi, name="bs")
        attenuator_0 = Attenuator(mode=0, transmissivity=0.11, name="att0")
        attenuator_1 = Attenuator(mode=1, transmissivity=0.02, name="att1")
        pnr = Number(mode=1, n=1, name="pnr").dual
        components = [state_0, state_1, bs_gate, attenuator_0, attenuator_1, pnr]
        circuit = Circuit(components)

        # explicit to_fock call to match the Fock shape of the ``Circuit``
        expected_result = (state_0 >> state_1 >> bs_gate >> attenuator_0 >> attenuator_1).to_fock(
            2
        ) >> pnr
        assert circuit.run(output_shape=(2, 2)) == expected_result

        # Ket example
        state_2 = SqueezedVacuum(2, 0.3, name="s2")
        interf = Interferometer.random(modes=(0, 1, 2), name="inter")
        pnr1 = Number(1, 6, name="pnr1").dual
        pnr2 = Number(2, 4, name="pnr2").dual
        components = [state_0, state_1, state_2, interf, pnr1, pnr2]
        circuit = Circuit(components)

        # explicit to_fock call to match the Fock shape of the ``Circuit``
        expected_result = (
            (state_0 >> state_1 >> state_2 >> interf).to_fock((12, 7, 5)) >> pnr1 >> pnr2
        )
        assert circuit.run(output_shape=(12,)) == expected_result

        # DM example
        dummy_rho = GaussianDM.random((0, 1), name="dm")
        dgate0 = Dgate(0, alpha=0.1, name="d0")
        dgate1 = Dgate(1, alpha=0.1, name="d1")
        components = [dummy_rho, dgate0, dgate1]
        circuit = Circuit(components)
        assert circuit.run() == dummy_rho >> dgate0 >> dgate1
        assert (
            circuit.run(representation=ReprEnum.FOCK) == (dummy_rho >> dgate0 >> dgate1).to_fock()
        )  # Fock repr

    def test_trace_out(self):
        dummy_rho = GaussianDM.random((0, 1, 2), name="dm")
        dgate0 = Dgate(0, alpha=0.1, name="d0")
        dgate1 = Dgate(1, alpha=0.1, name="d1")
        components = [dummy_rho, dgate0, dgate1]
        circuit = Circuit(components)

        traced_1 = circuit.trace_out(1)
        assert traced_1.run() == dummy_rho >> dgate0 >> dgate1 >> TraceOut(1)

        traced_1_2 = circuit.trace_out((1, 2))
        assert traced_1_2.run() == dummy_rho >> dgate0 >> dgate1 >> TraceOut((1, 2))
