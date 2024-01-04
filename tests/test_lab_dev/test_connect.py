# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Tests that the connect function connects things correctly.
"""

from mrmustard.lab_dev.circuit_components import connect
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Dgate


def test_connect():
    vacuum = Vacuum(3)
    d1 = Dgate(1, modes=[0, 8, 9])
    d2 = Dgate(1, modes=[0, 1, 2])

    circ = vacuum >> d1 >> d1 >> d2
    components = connect(circ.components)

    # check that all the modes are still there and no new modes are added
    assert list(components[0].wires.out_ket.keys()) == [0, 1, 2]
    assert list(components[1].wires.out_ket.keys()) == [0, 8, 9]
    assert list(components[2].wires.out_ket.keys()) == [0, 8, 9]
    assert list(components[3].wires.out_ket.keys()) == [0, 1, 2]

    # check connections on mode 0
    assert components[0].wires.out_ket[0] == components[1].wires.in_ket[0]
    assert components[1].wires.out_ket[0] == components[2].wires.in_ket[0]
    assert components[2].wires.out_ket[0] == components[3].wires.in_ket[0]

    # check connections on mode 1
    assert components[0].wires.out_ket[1] == components[3].wires.in_ket[1]

    # check connections on mode 2
    assert components[0].wires.out_ket[2] == components[3].wires.in_ket[2]

    # check connections on mode 8
    assert components[1].wires.out_ket[8] == components[2].wires.in_ket[8]

    # check connections on mode 9
    assert components[1].wires.out_ket[9] == components[2].wires.in_ket[9]
