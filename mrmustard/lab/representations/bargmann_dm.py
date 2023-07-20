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

from mrmustard.lab.representations.bargmann import Bargmann
from mrmustard.typing import Matrix, Scalar, Vector, Tensor

class BargmannDM(Bargmann):

    def __init__(self, A: Matrix, b: Vector, c: Scalar) -> None:
        super().__init__(A=A, b=b, c=c)
        self.num_modes = A.shape[-1] // 2

    @property
    def purity(self) -> Scalar:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")
    

    @property    
    def norm(self) -> float:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")


    @property
    def von_neumann_entropy(self) -> float:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")
    

    @property
    def number_means(self) -> Vector:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")
    

    @property
    def number_cov(self) -> Matrix:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")
    

    @property
    def number_variances(self) -> int:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")
    

    @property
    def number_stdev(self) -> int:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")


    @property
    def probability(self) -> Tensor:
        raise NotImplementedError(f"This property is not available in {self.__class__.__qualname__} representation")