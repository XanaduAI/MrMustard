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

"""Tests for the to_bargmann function."""

from mrmustard import settings
from mrmustard.lab import GaussianKet
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum import to_bargmann
from mrmustard.physics.utils import random_Abc


class TestToBargmann:
    """Tests for the to_bargmann function."""

    def test_array_ansatz_single_mode(self):
        """Test ArrayAnsatz single mode conversion."""
        array = settings.get_rng().random((15,))
        f = ArrayAnsatz(array, batch_dims=0)

        result = to_bargmann(f)
        assert isinstance(result, PolyExpAnsatz)
        assert result.num_CV_vars == 1

    def test_array_ansatz_with_batch(self):
        """Test ArrayAnsatz with batch dimensions."""
        array = settings.get_rng().random((2, 3, 10, 12))
        f = ArrayAnsatz(array, batch_dims=2)

        result = to_bargmann(f)
        assert isinstance(result, PolyExpAnsatz)
        assert result.batch_shape == (2, 3)

    def test_array_ansatz_with_original_abc(self):
        """Test ArrayAnsatz conversion when original ABC data exists."""
        g = GaussianKet.random([0, 1])
        f = g.to_fock((10, 12))

        result = to_bargmann(f.ansatz)
        assert isinstance(result, PolyExpAnsatz)

    def test_array_ansatz_without_original_abc(self):
        """Test ArrayAnsatz conversion without original ABC data."""
        array = settings.get_rng().random((10, 12))
        f = ArrayAnsatz(array, batch_dims=0)

        result = to_bargmann(f)
        assert isinstance(result, PolyExpAnsatz)
        assert result.c.shape == (10, 12)

    def test_polyexp_ansatz_unchanged(self):
        """Test that PolyExpAnsatz is returned unchanged."""
        g = PolyExpAnsatz(*random_Abc(2))
        result = to_bargmann(g)
        assert isinstance(result, PolyExpAnsatz)
        assert result is g
