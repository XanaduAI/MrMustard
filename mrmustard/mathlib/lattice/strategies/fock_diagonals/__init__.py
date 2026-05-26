# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagonal-lattice strategies for exact Fock-space recurrences.

Exports ``fock_diagonals`` (multimode diagonal amplitudes),
``fock_diagonals_1leftover`` (conditional density matrices, one leftover mode),
and ``generate_partitions`` (bounded integer compositions utility; from ``utils``).
"""

from .conditional_dm import fock_diagonals_1leftover
from .diagonal_amplitudes import fock_diagonals
from .utils import generate_partitions

__all__ = [
    "fock_diagonals",
    "fock_diagonals_1leftover",
    "generate_partitions",
]
