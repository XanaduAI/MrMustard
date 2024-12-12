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

"""Tests for the Batch class."""

# pylint: disable=missing-function-docstring

import pytest
import numpy as np

from mrmustard import math
from mrmustard.physics.batches import Batch

from ..conftest import skip_tf


class TestBatch:
    r"""
    Tests for the Batch class.
    """

    array5688 = np.random.random((5, 6, 8, 8))

    def test_init(self):
        batch_default = Batch(data=self.array5688)
        assert math.allclose(batch_default.data, self.array5688)
        assert len(batch_default.batch_labels) == 1
        assert batch_default.batch_shape == (5,)
        assert batch_default.core_shape == (6, 8, 8)
        assert batch_default.shape == self.array5688.shape

        batch = Batch(data=self.array5688, batch_shape=(5, 6), batch_labels=("a", "b"))
        assert math.allclose(batch.data, self.array5688)
        assert batch.batch_labels == ("a", "b")
        assert batch.batch_shape == (5, 6)
        assert batch.core_shape == (8, 8)
        assert batch.shape == self.array5688.shape

        assert math.allclose(batch_default, batch)
        assert batch_default != batch

        with pytest.raises(ValueError, match="batch shape"):
            Batch(self.array5688, batch_shape=(6, 6))  # pylint: disable=pointless-statement

    def test_getitem(self):
        batch = Batch(data=self.array5688, batch_shape=(5, 6), batch_labels=("a", "b"))

        batch_slice0 = batch[0]
        assert isinstance(batch_slice0, Batch)
        assert math.allclose(batch_slice0.data, self.array5688[0])
        assert batch_slice0.batch_labels == ("b",)
        assert batch_slice0.batch_shape == (6,)
        assert batch_slice0.core_shape == (8, 8)
        assert batch_slice0.shape == (6, 8, 8)

        batch_slice1 = batch[:, 0]
        assert isinstance(batch_slice1, Batch)
        assert math.allclose(batch_slice1.data, self.array5688[:, 0])
        assert batch_slice1.batch_labels == ("a",)
        assert batch_slice1.batch_shape == (5,)
        assert batch_slice1.core_shape == (8, 8)
        assert batch_slice1.shape == (5, 8, 8)

        batch_slice2 = batch[0, 0]
        assert not isinstance(batch_slice2, Batch)
        assert math.allclose(batch_slice2, self.array5688[0, 0])

        with pytest.raises(IndexError, match="indices"):
            batch[:, :, 0]  # pylint: disable=pointless-statement

    def test_ufunc(self):
        skip_tf()
        batch = Batch(data=self.array5688, batch_shape=(5, 6), batch_labels=("a", "b"))
        # __call__
        assert math.allclose(math.exp(batch), math.exp(self.array5688))
        # reduce
        assert math.allclose(math.sum(batch, axis=(1,)), math.sum(self.array5688, axis=(1,)))

        with pytest.raises(ValueError, match="out of bounds"):
            math.sum(batch, axis=(2,))  # pylint: disable=pointless-statement
