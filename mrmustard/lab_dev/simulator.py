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

"""
Simulators for quantum circuits.
"""

from __future__ import annotations

from abc import ABC, abstractclassmethod
import numpy as np

from ..physics.representations import Bargmann
from .circuit_components import CircuitComponent
from .circuits import Circuit
from .wires import Wire, Wires

__all__ = ["Simulator", "SimulatorBargmann"]


class Simulator(ABC):
    r"""
    A base class for simulators.
    """

    @abstractclassmethod
    def run(self, circuit: Circuit) -> any:
        r"""
        Simulates the given circuit.

        Arguments:
            circuit: The circuit to simulate.
        """
        ...


class SimulatorBargmann(Simulator):
    r"""
    A simulator for circuits whose components are in the Bargmann representation.
    """

    @staticmethod
    def _validate(c: CircuitComponent):
        r"""
        Checks that given components is has ``Bargmann`` representation.
        """
        rep = c.representation
        if not isinstance(rep, Bargmann):
            msg = f"Expected `Bargmann`, found {rep.__class__}"
            raise ValueError(msg)

    def run(self, circuit: Circuit) -> CircuitComponent:
        r"""Bargmann simulator

        [Code Block]
        from mrmustard.lab_dev.transformations import Dgate
        from mrmustard.lab_dev.simulator import SimulatorBargmann

        dgate1 = Dgate(x=[0.2,0.3,0.1], modes=[0,2,3])
        dgate2 = Dgate(x=[0.2,0.1,-0.2,-0.3], modes=[2,3, 6,101])
        dgate3 = Dgate(x=[0.5], modes=[1])
        circuit = dgate1>>dgate2>>dgate3
        circuit.draw()

        simulator = SimulatorBargmann()
        output = simulator.run(circuit=circuit)
        output.wires.list_of_types_and_modes_of_wires()

        """
        # _preparecomponent()
        component1 = circuit.components[0]
        for component2 in circuit.components[1:]:
            modes_out_ket_component1 = component1.wires.modes_out_ket
            modes_out_bra_component1 = component1.wires.modes_out_bra

            modes_in_ket_component2 = component2.wires.modes_in_ket
            modes_in_bra_component2 = component2.wires.modes_in_bra

            intersection_ket = list(set(modes_out_ket_component1) & set(modes_in_ket_component2))
            intersection_bra = list(set(modes_out_bra_component1) & set(modes_in_bra_component2))

            index_A_matrix_component1 = []
            index_A_matrix_component2 = []

            for mode in intersection_ket:
                index_A_matrix_component1 += [
                    component1.wires.calculate_index_for_a_wire_on_given_mode_and_type(
                        "out_ket", mode
                    )
                ]
                index_A_matrix_component2 += [
                    component2.wires.calculate_index_for_a_wire_on_given_mode_and_type(
                        "in_ket", mode
                    )
                ]

            for mode in intersection_bra:
                index_A_matrix_component1 += [
                    component1.wires.calculate_index_for_a_wire_on_given_mode_and_type(
                        "out_bra", mode
                    )
                ]
                index_A_matrix_component2 += [
                    component2.wires.calculate_index_for_a_wire_on_given_mode_and_type(
                        "in_bra", mode
                    )
                ]

            new_Bargmann = (
                component1.representation[index_A_matrix_component1]
                @ component2.representation[index_A_matrix_component2]
            )

            modes_in_ket_new = list(
                set(np.concatenate([component1.wires.modes_in_ket, component2.wires.modes_in_ket]))
            )
            modes_in_bra_new = list(
                set(np.concatenate([component1.wires.modes_in_bra, component2.wires.modes_in_bra]))
            )
            modes_out_ket_new = list(
                set(
                    np.concatenate([component1.wires.modes_out_ket, component2.wires.modes_out_ket])
                )
            )
            modes_out_bra_new = list(
                set(
                    np.concatenate([component1.wires.modes_out_bra, component2.wires.modes_out_bra])
                )
            )

            component1 = CircuitComponent.from_ABC(
                "test",
                A=new_Bargmann.A,
                B=new_Bargmann.b,
                c=new_Bargmann.c,
                modes_in_bra=modes_in_bra_new,
                modes_out_bra=modes_out_bra_new,
                modes_in_ket=modes_in_ket_new,
                modes_out_ket=modes_out_ket_new,
            )
        return component1
