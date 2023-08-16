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

""" This package contains the modules implementing base classes for representations.

    Representations classes serves for the state. 
    We support the Wigner representation, Fock representation, Bargmann representation and WaveFunctionQ representation for now.
    Each of them has an abstract class and two concrete classes with ket and density matrix to represent pure and mixed states.

    Each representation class has a corresponding data class. The storage of data and all algebras come from the data attribute.

    All functions related to the specific representation will be found in this own representation classes.

    We support the converter to transfer from one representation class to another.
"""

__all__ = [
    "WignerKet",
    "WignerDM",
    "Fock",
    "FockKet",
    "FockDM",
    "Bargmann",
    "BargmannKet",
    "BargmannDM",
    "WaveFunctionQKet",
    "WaveFunctionQDM",
]
