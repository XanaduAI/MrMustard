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

from mrmustard.lab.representations.wigner import Wigner
from mrmustard.typing import Matrix, Vector

class WignerKet(Wigner):
    r""" Wigner representation of a Ket state.
    
    Args:
        cov: covariance matrices (real symmetric)
        mean: means (real)
        coeffs: coefficients (complex) 
    """

    def __init__(self,
                 cov: Matrix, 
                 means: Vector, 
                 coeffs: Matrix
                 ) -> None:
        
        super().__init__(cov=cov, means=means, coeffs=coeffs)
        self.num_modes = self.cov.shape[-1]