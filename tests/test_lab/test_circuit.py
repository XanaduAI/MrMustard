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


from hypothesis import given

from mrmustard.lab import *
from tests.random import angle, medium_float, n_mode_pure_state, r


def test_circuit_placement_SD():
    "tests that Sgate and Dgate can be placed in any order"
    assert Sgate(1.0)[1] >> Dgate(1.0)[0] == Dgate(1.0)[0] >> Sgate(1.0)[1]


def test_circuit_placement_SR():
    "tests that Sgate and Rgate can be placed in any order"
    assert Sgate(1.0)[1] >> Rgate(1.0)[0] == Rgate(1.0)[0] >> Sgate(1.0)[1]


def test_circuit_placement_RD():
    "tests that Rgate and Dgate can be placed in any order"
    assert Rgate(1.0)[1] >> Dgate(1.0)[0] == Dgate(1.0)[0] >> Rgate(1.0)[1]


def test_circuit_placement_BS():
    "tests that BSgate and Sgate can be placed in any order"
    assert BSgate(1.0)[1, 2] >> Sgate(1.0)[0] == Sgate(1.0)[0] >> BSgate(1.0)[1, 2]


def test_circuit_placement_BSBS():
    "tests that BSgates can be placed in any order"
    assert BSgate(1.0)[1, 2] >> BSgate(1.0)[0, 3] == BSgate(1.0)[0, 3] >> BSgate(1.0)[1, 2]


def test_is_unitary():
    "test that the is_unitary property is correct"
    assert not (Ggate(1) >> Attenuator(0.1)).is_unitary
    assert Ggate(1).is_unitary
    assert (Ggate(1) >> Ggate(1)).is_unitary
    assert not (Ggate(2) >> Attenuator([0.1, 0.2])).is_unitary


@given(
    r=r, phi1=angle, phi2=angle, x=medium_float, y=medium_float, G=n_mode_pure_state(num_modes=1)
)
def test_shift(r, phi1, phi2, x, y, G):
    "test that the leftshift/rightshift operator works as expected"
    circ = Sgate(r, phi1) >> Dgate(x, y) >> Rgate(phi2)
    assert G == (circ << G) >> circ
