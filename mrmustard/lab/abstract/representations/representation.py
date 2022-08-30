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

"""This file contains the abstract Representation class."""



from abc import ABC, abstractmethod

class Representation(ABC):

    @abstractmethod # so that Representation can't be instantiated but subclasses can still call super().__init__()
    def __init__(self, representation: Representation):
        if isinstance(representation, DensityMatrix):
            self.from_dm(representation)
        if isinstance(representation, Ket):
            self.from_ket(representation)
        if isinstance(representation, WaveFunction):
            self.from_wf(representation)
        if isinstance(representation, Bargmann):
            self.from_bargmann(representation)
        if isinstance(representation, Wigner):
            self.from_wigner(representation)
        else:
            raise ValueError("Cannot convert representation {representation.__class__.__name__} to {cls.__name__}")

    def from_ket(self, Ket):
        raise NotImplementedError("Cannot convert a ket to {cls.__qualname__}")

    def from_dm(self, DensityMatrix):
        raise NotImplementedError("Cannot convert a density matrix to {cls.__qualname__}")

    def from_bargmann(self, Bargmann):
        raise NotImplementedError("Cannot convert bargmann to {cls.__qualname__}")
    
    def from_wf(self, Wavefunction):
        raise NotImplementedError("Cannot convert a wavefunction to {cls.__qualname__}")