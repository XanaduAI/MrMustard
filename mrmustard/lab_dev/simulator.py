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
        c = circuit.components[0]
        self._validate(c[0])
        ret = CircuitComponent.from_ABC(
            "",
            c[0].representation.A,
            c[0].representation.b,
            c[0].representation.c,
            list(c[0].wires.in_ket.keys()),  # write convenience methods
            list(c[0].wires.out_ket.keys()),
            list(c[0].wires.in_bra.keys()),
            list(c[0].wires.out_bra.keys()),
        )

        for c in circuit.components[1:]:
            self._validate(c)
            rep = ret.representation[..] @ c.representation[..]
            ret = CircuitComponent.from_ABC(
                "",
                c[0].representation.A,
                c[0].representation.b,
                c[0].representation.c,
                list(c[0].wires.in_ket.keys()),  # write convenience methods
                list(c[0].wires.out_ket.keys()),
                list(c[0].wires.in_bra.keys()),
                list(c[0].wires.out_bra.keys()),
            )

        return ret
