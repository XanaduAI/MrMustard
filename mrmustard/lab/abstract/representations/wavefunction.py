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


class WaveFunction(Ket):

    def __init__(self, data: Union[Representation, Array]):
        if isinstance(data, Array):
            self.array = data
        elif purity(data) < 1: 
            raise ValueError("Cannot convert a mixed state to a wavefunction")
        else:
            super().__init__(data)

    def from_bargmann(self, bargmann):
        print('implementing bargmann->ket transform...')
        A,b,c = bargmann.data
        self.data = math.hermite_renormalized(A,b,c)