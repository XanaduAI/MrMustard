# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the mm_einsum function."""

from mrmustard.physics.mm_einsum import mm_einsum
from mrmustard.lab_dev import Number, BSgate, QuadratureEigenstate
from mrmustard.physics.wires import Wires
import numpy as np


def test_mm_einsum():
    n = 100
    input1 = Number([0], n).to_fock().dm()
    input2 = Number([1], n).to_fock().dm()

    bs_p = (BSgate([0, 1], np.pi / 4) >> QuadratureEigenstate([1], 0, np.pi / 2).dual).to_fock(
        (2 * n + 1, n + 1, n + 1)
    )

    part1 = input1 @ bs_p.adjoint
    part1.representation._wires = Wires(modes_out_bra={0, 1}, modes_out_ket={0})

    part2 = input2 @ bs_p
    part2.representation._wires = Wires(modes_in_bra={1}, modes_out_ket={0}, modes_in_ket={0})

    expected = (part1 >> part2).representation.ansatz.array
    result = mm_einsum(
        input1, [0, 1], input2, [2, 3], bs_p.adjoint, [4, 0, 2], bs_p, [5, 1, 3], [4, 5]
    )
    assert np.allclose(expected, result.array)
