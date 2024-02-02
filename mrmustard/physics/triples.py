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

"""
This module contains the ABC triples for states, transformations in Bargmann representation.
"""

from typing import Union
from mrmustard import math
from mrmustard.utils.typing import Matrix, Vector, Scalar

def vacuum_ABC_triples(num_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of the pure vacuum state.

    Args:
        num_modes (int): number of modes

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the pure vacuum state.
    """
    return math.zeros((num_modes, num_modes)), math.zeros(num_modes), 1

def coherent_ABC_triples(x: Union[Scalar, Vector], y: Union[Scalar, Vector]) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of the pure coherent state.

    Args:
        x (scalar or vector): real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y (scalar or vector): imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)  

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the pure coherent state.
    """
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    num_modes = x.shape
    return math.zeros((num_modes, num_modes)), x+1j*y, math.exp(-0.5*(x**2 + y**2))