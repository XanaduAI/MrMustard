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

from mrmustard.representations import Representation

class WaveFunctionQ(Representation):

    def __init__(self):
        super().__init__()


    def purity(self):
        raise NotImplementedError("Get this of this state from other representations!")
    

    def number_means(self):
        raise NotImplementedError("Get this of this state from other representations!")
    
    
    def number_cov(self):
        raise NotImplementedError("Get this of this state from other representations!")
    

    def norm(self):
        raise NotImplementedError("Get this of this state from other representations!")


    def von_neumann_entropy(self):
        raise NotImplementedError("Get this of this state from other representations!") 