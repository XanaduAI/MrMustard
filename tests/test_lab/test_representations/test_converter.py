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

from mrmustard.lab.representations.converter import Converter

class TestConverter():
    ######################Test Init#################################
    def test_init_converter(self):
        conv = Converter()
        if not isinstance(conv, Converter):
            raise IndentationError("")

    #######################Test Conversion###########################
    #Wigner -> Bargmann
    def test_convert_from_wignerket_to_bargmannket(self):
        pass

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