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

class TestConverter():
    ######################Test Init#################################
    def test_init_converter(self):
        conv = Converter()
        if not isinstance(conv, Converter):
            raise IndentationError("")

    #######################Test Conversion###########################
    #Wigner -> Bargmann
    def test_convert_from_wignerket_to_bargmannket(self):
        converter = Converter()
        symplectic = np.array([[0,1],[-1,0]])
        displacement = np.array([0,0])
        wigner_ket = WignerKet(symplectic=symplectic, displacement=displacement)
        bargmann_ket = converter.convert(source=wigner_ket, destination="Bargmann")
        assert isinstance(bargmann_ket, BargmannKet), "The conversion is not correct!"

    def test_convert_from_wignerdm_to_bargmanndm(self):
        pass

    #Wigner -> Fock
    def test_convert_from_wignerket_to_fockket(self):
        pass

    def test_convert_from_wignerdm_to_fockdm(self):
        pass

    #Fock -> WaveFunctionQ
    def test_convert_from_fockket_to_wavefunctionqket(self):
        pass

    def test_convert_from_fockdm_to_wavefunctionqdm(self):
        pass