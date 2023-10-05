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

"""This module contains tests for simulating quantum circuits with tensor networks."""

from mrmustard.math.tensor_networks import *
from mrmustard.lab.gates import BSgate, Ggate, Rgate, Sgate, Dgate

import numpy as np
import pytest


class TestTransformations:
    r"""
    Tests that transformations can be contracted by tensor networks.
    """

    @pytest.mark.parametrize("modes", [[0], [1, 2], [3, 4, 5, 6]])
    @pytest.mark.parametrize("cutoff", [2, 5])
    def test_squeeze_and_displace(self, modes, cutoff):
        r"""
        Tests with squeezing followed by displacement.
        """
        g1 = Sgate(0.1, modes=modes)
        g2 = Dgate(0.2, modes=modes)

        for mode in modes:
            connect(g1.output.ket[mode], g2.input.ket[mode])
        contraction = contract([g1, g2], dim=cutoff)

        smat = Sgate(0.1).U(cutoffs=[cutoff])
        dmat = Dgate(0.2).U(cutoffs=[cutoff])
        expected = np.dot(smat, dmat)
        if len(modes) == 2:
            expected = np.kron(expected, expected)
            expected = expected.reshape(*contraction.shape)
        if len(modes) == 4:
            expected = np.kron(expected, expected)
            expected = np.kron(expected, expected)
            expected = expected.reshape(*contraction.shape)

        assert np.allclose(contraction, expected)
