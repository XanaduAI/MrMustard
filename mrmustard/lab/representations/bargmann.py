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

from mrmustard.math import Math
from mrmustard.representations import Representation
from mrmustard.typing import Scalar, RealMatrix, RealVector

math = Math()

class Bargmann(Representation):

    def __init__(self):
        super().__init__()


    def purity(self):
        raise NotImplementedError("Get this of this state from other representations!")
    

    def number_means(self):
        raise NotImplementedError("Get this of this state from other representations!")
    
    
    def number_cov(self):
        raise NotImplementedError("Get this of this state from other representations!")
    

    def __eq__(self, other:Representation) -> bool:
        r"""Compares two Representations (States) equal or not"""
        return self.data.__eq__(other)


    def __rmul__(self, other:Representation) -> Representation:
        r"""Multiplies two Representations (States)"""
        return self.data.__rmul__(other)


    def __add__(self, other:Representation) -> Representation:
        r"""Adds two Representations (States)"""
        return self.data.__add__(other)


    def __truediv__(self, other:Representation) -> Representation:
        r"""Divides two Representations (States)"""
        return self.data.__truediv__(other)
    

    def norm(self):
        raise NotImplementedError("Get this of this state from other representations!")


    def von_neumann_entropy(self):
        raise NotImplementedError("Get this of this state from other representations!") 
