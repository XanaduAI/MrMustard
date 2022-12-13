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

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard.physics import gaussian
from mrmustard.lab import *
from mrmustard import settings
from tests import random


def test_circuit_placement_SD():
    'tests that Sgate and Dgate can be placed in any order'
    assert Sgate(1.0)[1] >> Dgate(1.0)[0] == Dgate(1.0)[0] >> Sgate(1.0)[1]


def test_circuit_placement_SR():
    'tests that Sgate and Rgate can be placed in any order'
    assert Sgate(1.0)[1] >> Rgate(1.0)[0] == Rgate(1.0)[0] >> Sgate(1.0)[1]


def test_circuit_placement_RD():
    'tests that Rgate and Dgate can be placed in any order'
    assert Rgate(1.0)[1] >> Dgate(1.0)[0] == Dgate(1.0)[0] >> Rgate(1.0)[1]


def test_circuit_placement_BS():
    'tests that BSgate and Sgate can be placed in any order'
    assert BSgate(1.0)[1, 2] >> Sgate(1.0)[0] == Sgate(1.0)[0] >> BSgate(1.0)[1, 2]


def test_circuit_placement_BSBS():
    'tests that BSgates can be placed in any order'
    assert BSgate(1.0)[1, 2] >> BSgate(1.0)[0, 3] == BSgate(1.0)[0, 3] >> BSgate(1.0)[1, 2]
