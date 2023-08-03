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

from mrmustard.physics.gaussian import squeezing_symplectic, squeezed_vacuum_cov
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
from hypothesis import given
from tests.random import medium_float, r, angle
from mrmustard.math import Math
math = Math()
import pytest

from thewalrus.random import random_symplectic, random_covariance

class TestConverter():
    def __init__(self) -> None:
        self.converter = Converter()

    #######################Test Conversion###########################
    #Wigner -> Bargmann
    @given(x=medium_float, y=medium_float)
    def test_convert_from_wignerket_to_bargmannket_cohenrent_state(self, x, y):
        """Test that the Bargmann representation of a ket is correct for a coherent state"""
        hbar = settings.HBAR
        N = 1
        symplectic = math.eye(N * 2, dtype=math.float64)
        displacement = math.sqrt(2 * hbar, dtype=symplectic.dtype) * math.concat([x, y], axis=0)
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        bargmann_ket = self.converter.convert(source=wigner_ket, destination="Bargmann")
        assert isinstance(bargmann_ket, BargmannKet), "The conversion is not correct!"
        assert bargmann_ket.A == 0.0
        assert bargmann_ket.b == (x + 1j*y)
        assert bargmann_ket.c == math.exp(-1/2*math.norm(x + 1j*y)**2)
    
    @given(r=r, phi=angle)
    def test_convert_from_wignerket_to_bargmannket_squeezed_state(self, r, phi):
        """Test that the Bargmann representation of a ket from Wigner is correct for a squeezed state"""
        hbar = settings.HBAR
        N = 1
        symplectic = squeezing_symplectic(r=r, phi=phi)
        displacement = math.sqrt(2 * hbar, dtype=symplectic.dtype) * math.concat([0, 0], axis=0)
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        bargmann_ket = self.converter.convert(source=wigner_ket, destination="Bargmann")
        assert isinstance(bargmann_ket, BargmannKet), "The conversion is not correct!"
        assert bargmann_ket.A == math.tanh(r)*math.exp(1j*phi)
        assert bargmann_ket.b == 0
        assert bargmann_ket.c == math.sqrt(math.sech(r))

    @given(x=medium_float, y=medium_float)
    def test_convert_from_wignerdm_to_bargmanndm_coherent_state(self, x, y):
        """Test that the Bargmann representation of a dm from Wigner is correct for a coherent state"""
        hbar = settings.HBAR
        N=2
        cov = math.eye(N * 2, dtype=math.float64)
        means = math.sqrt(2 * hbar, dtype=cov.dtype) * math.concat([x, y], axis=0)
        wigner_dm = WignerDM(cov=cov, means=means)
        bargmann_dm = self.converter.convert(source=wigner_dm, destination="Bargmann")
        assert isinstance(bargmann_dm, BargmannDM), "The conversion is not correct!"
        assert bargmann_dm.A == math.zeros(N * N, dtype=math.float64)
        assert bargmann_dm.b == math.concat([x - 1j* y, x+ 1j* y], axis=0)
        assert bargmann_dm.c == math.exp(-math.norm(x + 1j*y)**2)
    

    @given(r=r, phi=angle)
    def test_convert_from_wignerdm_to_bargmanndm_squeezed_state(self, r, phi):
        """Test that the Bargmann representation of a dm is correct for a squeezed state"""
        hbar = settings.HBAR
        N = 2
        cov = squeezed_vacuum_cov(r, phi)
        means = math.sqrt(2 * hbar, dtype=cov.dtype) * math.concat([0, 0], axis=0)
        wigner_dm = WignerDM(cov=cov, means=means)
        bargmann_dm = self.converter.convert(source=wigner_dm, destination="Bargmann")
        assert isinstance(bargmann_dm, BargmannDM), "The conversion is not correct!"
        assert bargmann_dm.A == math.array([[math.tanh(r)*math.exp(-1j*phi),0],[0, math.tanh(r)*math.exp(1j*phi)]])
        assert bargmann_dm.b == math.concat([0, 0], axis=0)
        assert bargmann_dm.c == math.sech(r)

    #Wigner -> Fock
    @given(x=medium_float, y=medium_float)
    def test_convert_from_wignerket_to_fockket_coherent_state(self, x, y):
        """Test that the Fock representation of a ket from Wigner is correct for a coherent state"""
        hbar = settings.HBAR
        N = 1
        symplectic = math.eye(N * 2, dtype=math.float64)
        displacement = math.sqrt(2 * hbar, dtype=symplectic.dtype) * math.concat([x, y], axis=0)
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        fock_ket = self.converter.convert(source=wigner_ket, destination="Fock")
        assert isinstance(fock_ket, FockKet), "The conversion is not correct!"
        assert fock_ket[0] == math.exp(-1/2*math.norm(x + 1j*y)**2)
        assert fock_ket[1] == fock_ket[0] * (x + 1j*y)
        assert fock_ket[2] == fock_ket[1] * (x + 1j*y)/math.sqrt(2)


    @given(x=medium_float, y=medium_float)
    def test_convert_from_wignerdm_to_fockdm_coherent_state(self, x, y):
        """Test that the Fock representation of a dm from Wigner is correct for a coherent state"""
        hbar = settings.HBAR
        N=2
        cov = math.eye(N * 2, dtype=math.float64)
        means = math.sqrt(2 * hbar, dtype=cov.dtype) * math.concat([x, y], axis=0)
        wigner_dm = WignerDM(cov=cov, means=means)
        fock_dm = self.converter.convert(source=wigner_dm, destination="Fock")
        assert isinstance(fock_dm, FockDM), "The conversion is not correct!"
        assert fock_dm[0,0] == math.exp(-math.norm(x + 1j*y)**2)
        assert fock_dm[0,1] == fock_dm[0,0]**2 * (x + 1j*y)
        assert fock_dm[1,0] == fock_dm[0,1]


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