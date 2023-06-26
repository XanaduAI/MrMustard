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

import numpy as np

from mrmustard.lab.representations.converter import Converter
from mrmustard.lab.representations.fock_ket import FockKet
from mrmustard.lab.representations.fock_dm import FockDM
from mrmustard.lab.representations.bargmann_ket import BargmannKet
from mrmustard.lab.representations.bargmann_dm import BargmannDM
from mrmustard.lab.representations.wigner_ket import WignerKet
from mrmustard.lab.representations.wigner_dm import WignerDM
from mrmustard.lab.representations.wavefunctionq_ket import WaveFunctionQKet
from mrmustard.lab.representations.wavefunctionq_dm import WaveFunctionQDM

from mrmustard import settings
from mrmustard.math import Math
math = Math()
import pytest

from thewalrus.random import random_symplectic, random_covariance


class TestConverter():

    #######################Test Conversion###########################
    #Wigner -> Bargmann
    @pytest.mark.parametrize("N",(2,4,6))
    def test_convert_from_wignerket_to_bargmannket(self, N):
        converter = Converter()
        symplectic = math.astensor(random_symplectic(N//2))
        displacement = math.astensor(np.random.rand(N))
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        bargmann_ket = converter.convert(source=wigner_ket, destination="Bargmann")
        assert isinstance(bargmann_ket, BargmannKet), "The conversion is not correct!"

    @pytest.mark.parametrize("N",(3,4,5))
    def test_convert_from_wignerdm_to_bargmanndm(self, N):
        converter = Converter()
        cov = math.astensor(random_covariance(N))
        means = math.astensor(np.random.rand(2*N))
        wigner_dm = WignerDM(cov=cov, means=means)
        bargmann_dm = converter.convert(source=wigner_dm, destination="Bargmann")
        assert isinstance(bargmann_dm, BargmannDM), "The conversion is not correct!"

    #Wigner -> Fock
    @pytest.mark.parametrize("N",(2,4,6))
    def test_convert_from_wignerket_to_fockket(self, N):
        converter = Converter()
        symplectic = math.astensor(random_symplectic(N//2))
        displacement = math.astensor(np.random.rand(N))
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        fock_ket = converter.convert(source=wigner_ket, destination="Fock")
        assert isinstance(fock_ket, FockKet), "The conversion is not correct!"

    @pytest.mark.parametrize("N",(3,4,5))
    def test_convert_from_wignerdm_to_fockdm(self, N):
        converter = Converter()
        cov = math.astensor(random_covariance(N))
        means = math.astensor(np.random.rand(2*N))
        wigner_dm = WignerDM(cov=cov, means=means)
        fock_dm = converter.convert(source=wigner_dm, destination="Fock")
        assert isinstance(fock_dm, FockDM), "The conversion is not correct!"

    #Fock -> WaveFunctionQ
    def test_convert_from_fockket_to_wavefunctionqket(self):
        pass

    def test_convert_from_fockdm_to_wavefunctionqdm(self):
        pass

class TestConverterFockWithCutoffs():

    def test_convert_from_wignerket_to_fockket_with_cutoffs(self):
        pass

    def test_convert_from_wignerdm_to_fockdm_with_cutoffs(self):
        pass