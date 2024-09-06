# Copyright 2024 Xanadu Quantum Technologies Inc.

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
The class representing a complex fourier transform.
"""

from typing import Sequence
from mrmustard.lab_dev.transformations.base import Map
from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples

__all__ = ["CFT"]


class CFT(Map):
    r"""The Complex Fourier Transformation as a channel.
    The main use is to convert between Characteristic functions and phase space functions.

    Args:
        num_modes: number of modes of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            name="CFT",
        )
        self._representation = Bargmann.from_function(
            fn=triples.complex_fourier_transform_Abc, n_modes=len(modes)
        )
