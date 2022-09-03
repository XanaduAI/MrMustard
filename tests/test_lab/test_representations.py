# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mrmustard.math import Math
math = Math()

from mrmustard.lab.abstract.representations import Wigner, Characteristic, Ket, DensityMatrix

W = Wigner(cov, mean) # gaussian in Wigner representation
W = Wigner(array=array) # Wigner function
W = Wigner.from_repr(Characteristic(cov, mean)) # convert from characteristic representation

Wigner(cov, mean) + Wigner(cov, mean) == Wigner(math.concat(cov, cov), math.concat(mean, mean))

C = Characteristic(cov, mean) # gaussian in characteristic representation
C = Characteristic(array=array)
C = Characteristic.from_repr(Wigner(cov, mean)) # convert from Wigner representation

F = Ket(array) # Fock ket representation (in Hilbert space)
F = DensityMatrix(array) # Fock density matrix representation (in convex space)

WFq = WavefunctionQ(array=array) # wave function in position (technically a ket)
WFp = WavefunctionP(array=array) # wave function in momentum (technically a ket)
WFq = WavefunctionQ(cov, mean) # wave function in position (technically a ket)

