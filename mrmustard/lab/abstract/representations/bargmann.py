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

class Bargmann(Representation, Holomorphic):
    """Bargmann representation of a quantum state.

    The Bargmann representation is a representation of a quantum state
    as a function of the Bargmann coefficients. The Bargmann coefficients
    are the coefficients of the expansion of the state in the basis of
    the Bargmann functions.

    Args:
        data (Representation, Array): The data to initialize the Bargmann representation with.
    """

    def __init__(self, data: Union[Representation, Array]):
        if isinstance(data, Array):
            self.array = data
        else:
            super().__init__(data)

    def from_wf(self, wf):
        """Convert a wavefunction to Bargmann representation.

        Args:
            wf (WaveFunction): The wavefunction to convert to Bargmann representation.
        """
        print(f'wf->{self.__class__.__qualname__}')

    def from_dm(self, dm):
        """Convert a density matrix to Bargmann representation.

        Args:
            dm (DensityMatrix): The density matrix to convert to Bargmann representation.
        """
        print(f'dm->{self.__class__.__qualname__}')
        pass

    def from_ket(self, ket):
        """Convert a ket to Bargmann representation.

        Args:
            ket (Ket): The ket to convert to Bargmann representation.
        """
        print(f'ket->{self.__class__.__qualname__}')
        pass

    def from_bargmann(self, bargmann):
        """Convert Bargmann representation to Bargmann representation.

        Args:
            bargmann (Bargmann): The Bargmann representation to convert to Bargmann representation.
        """
        pass

    def from_wigner(self, wigner):
        """Convert Wigner representation to Bargmann representation.

        Args:
            wigner (Wigner): The Wigner representation to convert to Bargmann representation.
        """
        pass