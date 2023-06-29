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
from mock_data import MockNoCommonAttributesObject

from tests.test_lab.test_representations.test_data.test_data import TestData
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
    @pytest.mark.parametrize('other', [
        WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)),
        WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)),
        WavefunctionArrayData(qs=np.zeros(9), array=np.ones(10)),
        WavefunctionArrayData(qs=np.zeros(8), array=np.zeros(10)),
        WavefunctionArrayData(qs=np.eye(8), array=np.zeros(10))
                            ])
    @pytest.mark.parametrize('operator', [op.add, op.sub, op.mul])
    def test_different_value_or_shape_of_qs_raises_ValueError(self, DATA, other, operator):
        with pytest.raises(ValueError):
            operator(DATA, other)


    @pytest.mark.parametrize('other', [WavefunctionArrayData(qs=np.ones(10), array=np.ones(10))])
    @pytest.mark.parametrize('operator', [op.add, op.sub, op.mul, op.and_])
    def test_qs_for_new_objects_are_same_as_initial_qs_after_arity2_operation(self, 
                                                                              DATA, 
                                                                              other, 
                                                                              operator):
        new_obj = operator(DATA, other)
        assert np.allclose(DATA.qs, new_obj.qs)


    @pytest.mark.parametrize('operator', [op.neg])
    def test_qs_for_new_objects_are_same_as_initial_qs_after_arity1_operation(self,
                                                                              DATA,
                                                                              operator):
        new_obj = operator(DATA)
        assert np.allclose(DATA.qs, new_obj.qs)
        

    ####################  Init  ######################
    # NOTE : tested in parent
    

    ##################  Equality  ####################
    @pytest.mark.parametrize('other, truth_val', [
        (WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)), False),
        (WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)), False)
    ])
    def test_eq_returns_false_if_qs_different_irrespective_of_array(self, DATA, other, truth_val):
        assert (DATA == other) == truth_val


    ###########  Object multiplication  ##############
    # NOTE : done in parent


    ###############  Outer product  ##################
    # TODO : test and
    

    #################### Other #######################
    @pytest.mark.parametrize('other, truth_val', [
        (WavefunctionArrayData(qs=np.ones(10), array=np.zeros(10)), True), 
        (WavefunctionArrayData(qs=np.ones(10), array=np.ones(10)), True),
        (WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)), False), 
        (WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)), False)
                            ])
    def test_qs_is_same_returns_true_when_same_qs_and_false_when_diff_qs_irrespective_of_array(self,
                                                                        DATA,
                                                                        other,
                                                                        truth_val):
        assert DATA._qs_is_same(other) == truth_val


    @pytest.mark.parametrize('other', [MockNoCommonAttributesObject()])
    def test_qs_is_same_raises_TypeError_if_other_has_no_qs(self, DATA, other):
        with pytest.raises(TypeError):
            DATA._qs_is_same(other)