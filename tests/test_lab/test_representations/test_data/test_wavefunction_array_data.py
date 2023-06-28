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

from __future__ import annotations
import numpy as np
import operator as op
import pytest
from copy import deepcopy
from tools_for_tests import factory

#from tests.test_lab.test_representations.test_data.test_array_data import TestArrayData
from mrmustard.lab.representations.data.wavefunctionarray_data import WavefunctionArrayData

#########   Instantiating class to test  #########
@pytest.fixture
def PARAMS() -> dict:
    r""" Parameters for the class instance which is created. """
    params = {'qs':np.ones(10), 'array':np.ones(10)}
    return params


@pytest.fixture()
def DATA(PARAMS) -> WavefunctionArrayData:
    r""" Instance of the class that must be tested. """
    return factory(WavefunctionArrayData, **PARAMS)



class TestWavefunctionArrayData():

    #########   Common to different methods  #########
    @pytest.mark.parametrize('operator', [op.add, op.sub, op.mul])
    @pytest.mark.parametrize('other', [WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10))])
    def test_different_qs_raises_ValueError(self, DATA, other, operator):
        with pytest.raises(ValueError):
            _ = operator(DATA, other)

        

    def test_qs_for_new_objects_are_same_as_initial_qs(self):
        # test on add, sub, 
        pass

    ####################  Init  ######################
    # NOTE : tested in parent
    
    ##################  Equality  ####################

    def test_eq_returns_false_if_array_same_but_qs_different(self):
        pass

    ###########  Object multiplication  ##############
    # NOTE : done in parent

    ###############  Outer product  ##################
    # TODO : test and
    
    #################### Other #######################

    def test_qs_is_same_returns_true_when_same_qs(self):
        pass

    def test_qs_is_same_returns_false_when_different_qs(self):
        pass

    def test_qs_is_same_raises_TypeError_if_other_has_no_qs(self):
        pass