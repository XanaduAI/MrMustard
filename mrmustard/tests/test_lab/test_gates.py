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
from mrmustard.lab.states import *
from mrmustard.physics.abstract import State
from mrmustard.lab.gates import *
from mrmustard import settings
from mrmustard.tests import random

@given(input = random.random_pure_state(num_modes=1))
def test_Dgate_1mode_vacuum(input):
    state = Dgate()(input)

def test_Dgate_2mode_vacuum():
    D = Dgate()
    state = D[0,1](Vacuum(2))

def test_Dgate_2mode_vacuum():
    D = Dgate()
    vac = State.from_fock(fock=D[0,1](Vacuum(2)).ket(cutoffs=[4,4]), mixed=False)
    state = D(vac)