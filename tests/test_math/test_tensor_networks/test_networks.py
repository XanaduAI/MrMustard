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

"""This module contains tests for the networks.py module."""

from mrmustard.math.tensor_networks import *

import numpy as np
import pytest

# ~~~~~~~
# Helpers
# ~~~~~~~


class TId(Tensor):
    r"""
    A tensor whose value is the identity matrix of size ``cutoff``.
    """

    def value(self, cutoff):
        return np.eye(cutoff)


# ~~~~~~~
# Tests
# ~~~~~~~


class TestConnect:
    r"""
    Tests the function to connect wires between tensors.
    """

    def test_ids(self):
        r"""
        Tests that the ``id``s of tensors connected to each other are equal.
        """
        t1 = TId("t1", [1, 2, 3], [4, 5, 6])
        t2 = TId("t2", None, None, [7, 8, 9])
        t3 = TId("t3", None, None, None, [10])

        connect(t1.input.ket[1], t1.output.ket[4], 2)
        connect(t1.output.ket[5], t2.input.bra[8])
        connect(t1.input.ket[3], t3.output.bra[10])

        assert t1.input.ket[1].contraction_id == t1.output.ket[4].contraction_id
        assert t1.output.ket[5].contraction_id == t2.input.bra[8].contraction_id
        assert t1.input.ket[3].contraction_id == t3.output.bra[10].contraction_id

    def test_error(self):
        r"""
        Tests that wires that are already connected can no longer be connected.
        """
        t1 = TId("t1", [1, 2, 3], [4, 5, 6])
        t2 = TId("t2", None, None, [7, 8, 9])
        connect(t1.output.ket[5], t2.input.bra[8])

        with pytest.raises(ValueError, match="already connected"):
            connect(t1.output.ket[5], t2.input.bra[8])
