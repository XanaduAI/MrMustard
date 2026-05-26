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

"""Tests for the to_fock function."""

import pytest

from mrmustard import math
from mrmustard.lab import GaussianKet
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum.core import to_fock
from mrmustard.physics.utils import random_Abc


class TestToFock:
    """Tests for the to_fock function."""

    def test_array_ansatz_reduction(self):
        """Test that ArrayAnsatz is reduced to the given shape."""
        f = GaussianKet.random([0, 1]).to_fock((15, 20))
        reduced = to_fock(f.ansatz, (10, 12))
        assert isinstance(reduced, ArrayAnsatz)
        assert reduced.array.shape == (10, 12)

    def test_empty_shape_with_lin_sup(self):
        """Test that empty shape with lin_sup doesn't double-sum."""
        # Create a PolyExpAnsatz with lin_sup (linear superposition)
        g = (GaussianKet.random([0]) + GaussianKet.random([0])).ansatz
        assert g._lin_sup is True

        # Convert with empty shape and sum_lin_sup=True (preserve_lin_sup=False)
        result = to_fock(g, (), preserve_lin_sup=False)
        assert isinstance(result, ArrayAnsatz)
        assert result.batch_dims == 0
        assert result.batch_shape == ()

        # Verify the result matches the scalar property (which already sums lin_sup)
        expected = g.scalar
        assert result.array.shape == expected.shape
        assert math.allclose(result.array, expected)

    @pytest.mark.parametrize(
        "preserve_lin_sup,expected_batch_dims,expected_batch_shape",
        [
            (True, 1, (2,)),
            (False, 0, ()),
        ],
    )
    def test_preserve_lin_sup(self, preserve_lin_sup, expected_batch_dims, expected_batch_shape):
        """Test linear superposition preservation behavior."""
        g = (GaussianKet.random([0]) + GaussianKet.random([0])).ansatz
        result = to_fock(g, (10,), preserve_lin_sup=preserve_lin_sup)
        assert isinstance(result, ArrayAnsatz)
        assert result.batch_dims == expected_batch_dims
        assert result.batch_shape == expected_batch_shape

    @pytest.mark.parametrize("stable", [True, False])
    def test_stable_parameter(self, stable):
        """Test that stable parameter can be set."""
        g = PolyExpAnsatz(*random_Abc(1))
        result = to_fock(g, (10,), stable=stable)
        assert isinstance(result, ArrayAnsatz)

    def test_with_batch_dimensions(self):
        """Test to_fock with batched PolyExpAnsatz."""
        g = PolyExpAnsatz(*random_Abc(1, batch=(3,)))
        result = to_fock(g, (10,), preserve_lin_sup=True)
        assert isinstance(result, ArrayAnsatz)
        assert result.batch_shape == (3,)
        assert result.array.shape == (3, 10)
