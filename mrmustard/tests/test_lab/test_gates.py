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
from mrmustard.lab.gates import *
from mrmustard import settings


# a strategy to generate pure coherent states 

def test_Dgate_1mode_vacuum(state):
    state = Dgate()(Vacuum(1))


# test that Dgate can be applied to 2 mode vacuum state
def test_Dgate_2mode_vacuum():
    D = Dgate()
    state = D[0,1](Vacuum(2))

# test that Dgate can be applied to vacuum in fock representation
def test_Dgate_2mode_vacuum():
    D = Dgate()
    vac = State.from_fock(D[0,1](Vacuum(2).fock(cutoffs=[4,4])))
    state = D(vac)