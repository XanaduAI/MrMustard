# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains tests for ``ArrayAnsatz`` objects."""

from unittest.mock import patch

import numpy as np
import pytest
from ipywidgets import HTML, HBox, Tab, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math, settings
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz


class TestArrayAnsatz:
    r"""Tests the array ansatz."""

    array578 = settings.rng.random((5, 7, 8))
    array1578 = settings.rng.random((1, 5, 7, 8))
    array2578 = settings.rng.random((2, 5, 7, 8))
    array5578 = settings.rng.random((5, 5, 7, 8))

    def test_init_batched(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        assert isinstance(fock, ArrayAnsatz)
        assert np.allclose(fock.array, self.array1578)

    def test_init_non_batched(self):
        fock = ArrayAnsatz(self.array578, batch_dims=0)
        assert isinstance(fock, ArrayAnsatz)
        assert fock.array.shape == (5, 7, 8)
        assert np.allclose(fock.array, self.array578)

    def test_add(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock2 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (2, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] + self.array2578[0])
        assert np.allclose(fock1_add_fock2.array[1], self.array2578[1] + self.array2578[1])

    def test_sum_with_different_batch_raise_errors(self):
        array = settings.rng.random((2, 4, 5))
        array2 = settings.rng.random((3, 4, 8, 9))
        aa1 = ArrayAnsatz(array=array, batch_dims=0)
        aa2 = ArrayAnsatz(array=array2, batch_dims=1)

        with pytest.raises(ValueError):
            aa1 + aa2

    def test_and(self):
        array5123 = settings.rng.random((5, 1, 2, 3))
        array5234 = settings.rng.random((5, 2, 3, 4))
        fock1 = ArrayAnsatz(array=array5123, batch_dims=1)
        fock2 = ArrayAnsatz(array=array5234, batch_dims=1)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 1, 2, 3, 2, 3, 4)
        assert np.allclose(fock_test.array, np.einsum("abcd,aefg->abcdefg", array5123, array5234))

    def test_call(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        with pytest.raises(AttributeError, match="Cannot call"):
            fock(0)

    def test_conj(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_conj = fock.conj
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_contract_fock_fock(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock2 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = fock1.contract(
            fock2,
            idx1=["a", 0, 1, 2],
            idx2=["b", 0, 1, 3],
            idx_out=["a", "b", 2, 3],
        )
        assert fock_test.array.shape == (2, 1, 8, 8)
        assert np.allclose(
            fock_test.array,
            np.einsum("acde, bcdh -> abeh", self.array2578, self.array1578),
        )

    def test_divide_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_equal(self):
        array = settings.rng.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array)
        assert aa1 == aa2

    def test_from_function(self):
        def gen_array(x):
            return x

        x = math.astensor(settings.rng.random((5, 8, 8)))
        fock = ArrayAnsatz.from_function(gen_array, batch_dims=1, x=x)
        assert fock.array.shape == (5, 8, 8)
        assert math.allclose(fock.array, x)

    def test_multiply_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_neg(self):
        array = settings.rng.random((2, 4, 5))
        aa = ArrayAnsatz(array=array)
        minusaa = -aa
        assert isinstance(minusaa, ArrayAnsatz)
        assert np.allclose(minusaa.array, -array)

    def test_reduce(self):
        array1 = math.astensor(np.arange(27).reshape((1, 3, 3, 3)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)

        fock2 = fock1.reduce((3, 3, 3))
        assert fock1 == fock2

        fock3 = fock1.reduce((2, 2, 2))
        array3 = math.astensor([[[[0, 1], [3, 4]], [[9, 10], [12, 13]]]])
        assert fock3 == ArrayAnsatz(array3, batch_dims=1)

        fock4 = fock1.reduce((1, 3, 1))
        array4 = math.astensor([[[0], [3], [6]]])
        assert fock4 == ArrayAnsatz(array4, batch_dims=1)

    def test_reduce_error(self):
        array1 = math.astensor(np.arange(27).reshape((3, 3, 3)))
        fock1 = ArrayAnsatz(array1)

        with pytest.raises(ValueError, match="Expected shape"):
            fock1.reduce((1, 2))

        with pytest.raises(ValueError, match="Expected shape"):
            fock1.reduce((1, 2, 3, 4, 5))

    def test_reduce_padded(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        with pytest.warns(UserWarning):
            fock1 = fock.reduce((8, 8, 8))
        assert fock1.array.shape == (1, 8, 8, 8)

    def test_reorder(self):
        array1 = math.astensor(np.arange(8).reshape((1, 2, 2, 2)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        fock2 = fock1.reorder(order=(2, 1, 0))
        assert np.allclose(fock2.array, np.array([[[[0, 4], [2, 6]], [[1, 5], [3, 7]]]]))
        assert np.allclose(fock2.array, np.arange(8).reshape((1, 2, 2, 2), order="F"))

    def test_sub(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock2 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock1_add_fock2 = fock1 - fock2
        assert fock1_add_fock2.array.shape == (2, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] - self.array2578[0])
        assert np.allclose(fock1_add_fock2.array[1], self.array2578[1] - self.array2578[1])

    def test_to_from_dict(self):
        array1 = math.astensor(settings.rng.random((2, 5, 5, 1)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        assert ArrayAnsatz.from_dict(fock1.to_dict()) == fock1

    def test_trace(self):
        array1 = math.astensor(settings.rng.random((2, 5, 5, 1, 3, 3)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        fock2 = fock1.trace([0, 3], [1, 4])
        assert fock2.array.shape == (2, 1)
        assert np.allclose(fock2.array, np.einsum("abbdee->ad", array1))

    def test_truediv_by_scalar(self):
        array = settings.rng.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 / 6
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array / 6)

    @pytest.mark.parametrize("shape", [(1, 8), (1, 8, 8)])
    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr(self, mock_display, shape):
        """Test the IPython repr function."""
        rep_np = ArrayAnsatz(settings.rng.random(shape), batch_dims=1)
        rep_np._ipython_display_()
        [hbox] = mock_display.call_args.args
        assert isinstance(hbox, HBox)

        # the CSS, the header+ansatz, and the tabs of plots
        [css, left, plots] = hbox.children
        assert isinstance(css, HTML)
        assert isinstance(left, VBox)
        assert isinstance(plots, Tab)

        # left contains header and ansatz
        left = left.children
        assert len(left) == 2 and all(isinstance(w, HTML) for w in left)

        # one plot for magnitude, another for phase
        assert plots.titles == ("Magnitude", "Phase")
        plots = plots.children
        assert len(plots) == 2 and all(isinstance(p, FigureWidget) for p in plots)

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_batch_1(self, mock_display):
        """Test the IPython repr function does nothing with real batch."""
        rep = ArrayAnsatz(settings.rng.random((2, 8)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_3_dims_or_less(self, mock_display):
        """Test the IPython repr function does nothing with 4+ dims."""
        rep = ArrayAnsatz(settings.rng.random((1, 4, 4, 4)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(settings.rng.random((1, 8)), batch_dims=1)
        rep._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)

    def test_reorder_batch(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=2)
        fock_reordered = fock.reorder_batch([1, 0])
        assert fock_reordered.array.shape == (5, 1, 7, 8)

    def test_promote_core_to_batch_single_index(self):
        """Test promoting a single core dimension to batch."""
        # array shape: (2, 5, 7, 8) with batch_dims=1 -> 1 batch dim, 3 core dims
        fock = ArrayAnsatz(self.array2578, batch_dims=1)
        promoted = fock.promote_core_to_batch([1])  # promote core dim 1 (shape 7)

        # Expected: (2, 7, 5, 8) with batch_dims=2 -> 2 batch dims, 2 core dims
        assert promoted.batch_dims == 2
        assert promoted.array.shape == (2, 7, 5, 8)
        assert promoted.batch_shape == (2, 7)
        assert promoted.core_shape == (5, 8)

    def test_promote_core_to_batch_multiple_indices(self):
        """Test promoting multiple core dimensions to batch."""
        # array shape: (2, 5, 7, 8) with batch_dims=1 -> 1 batch dim, 3 core dims
        fock = ArrayAnsatz(self.array2578, batch_dims=1)
        promoted = fock.promote_core_to_batch([0, 2])  # promote core dims 0 and 2 (shapes 5, 8)

        # Expected: (2, 5, 8, 7) with batch_dims=3 -> 3 batch dims, 1 core dim
        assert promoted.batch_dims == 3
        assert promoted.array.shape == (2, 5, 8, 7)
        assert promoted.batch_shape == (2, 5, 8)
        assert promoted.core_shape == (7,)

    def test_promote_core_to_batch_empty_indices(self):
        """Test promoting empty list returns same ansatz."""
        fock = ArrayAnsatz(self.array2578, batch_dims=1)
        promoted = fock.promote_core_to_batch([])
        assert promoted is fock

    def test_promote_core_to_batch_no_batch_dims(self):
        """Test promoting core to batch when no initial batch dims."""
        # array shape: (5, 7, 8) with batch_dims=0 -> 0 batch dims, 3 core dims
        fock = ArrayAnsatz(self.array578, batch_dims=0)
        promoted = fock.promote_core_to_batch([1])  # promote core dim 1 (shape 7)

        # Expected: (7, 5, 8) with batch_dims=1 -> 1 batch dim, 2 core dims
        assert promoted.batch_dims == 1
        assert promoted.array.shape == (7, 5, 8)
        assert promoted.batch_shape == (7,)
        assert promoted.core_shape == (5, 8)

    def test_promote_core_to_batch_errors(self):
        """Test error cases for promote_core_to_batch."""
        fock = ArrayAnsatz(self.array2578, batch_dims=1)  # 3 core dims

        # Invalid indices
        with pytest.raises(ValueError, match="All core indices must be in range"):
            fock.promote_core_to_batch([3])  # index 3 >= 3 core dims

        with pytest.raises(ValueError, match="All core indices must be in range"):
            fock.promote_core_to_batch([-1])  # negative index

        # Duplicate indices
        with pytest.raises(ValueError, match="Core indices must be unique"):
            fock.promote_core_to_batch([0, 1, 0])

    def test_demote_batch_to_core_single_index(self):
        """Test demoting a single batch dimension to core."""
        # array shape: (2, 5, 7, 8) with batch_dims=2 -> 2 batch dims, 2 core dims
        fock = ArrayAnsatz(self.array2578, batch_dims=2)
        demoted = fock.demote_batch_to_core([1])  # demote batch dim 1 (shape 5)

        # Expected: (2, 5, 7, 8) with batch_dims=1 -> 1 batch dim, 3 core dims
        assert demoted.batch_dims == 1
        assert demoted.array.shape == (2, 5, 7, 8)
        assert demoted.batch_shape == (2,)
        assert demoted.core_shape == (5, 7, 8)

    def test_demote_batch_to_core_multiple_indices(self):
        """Test demoting multiple batch dimensions to core."""
        # Create a 3-batch-dim ansatz: (2, 3, 4, 5, 6) with batch_dims=3
        array = settings.rng.random((2, 3, 4, 5, 6))
        fock = ArrayAnsatz(array, batch_dims=3)
        demoted = fock.demote_batch_to_core([0, 2])  # demote batch dims 0 and 2 (shapes 2, 4)

        # Expected: (3, 2, 4, 5, 6) with batch_dims=1 -> 1 batch dim, 4 core dims
        assert demoted.batch_dims == 1
        assert demoted.array.shape == (3, 2, 4, 5, 6)
        assert demoted.batch_shape == (3,)
        assert demoted.core_shape == (2, 4, 5, 6)

    def test_demote_batch_to_core_empty_indices(self):
        """Test demoting empty list returns same ansatz."""
        fock = ArrayAnsatz(self.array2578, batch_dims=1)
        demoted = fock.demote_batch_to_core([])
        assert demoted is fock

    def test_demote_batch_to_core_all_batch_dims(self):
        """Test demoting all batch dimensions to core."""
        fock = ArrayAnsatz(self.array2578, batch_dims=2)  # 2 batch, 2 core
        demoted = fock.demote_batch_to_core([0, 1])  # demote both batch dims

        # Expected: batch_dims=0, all dims become core
        assert demoted.batch_dims == 0
        assert demoted.array.shape == (2, 5, 7, 8)
        assert demoted.batch_shape == ()
        assert demoted.core_shape == (2, 5, 7, 8)

    def test_demote_batch_to_core_errors(self):
        """Test error cases for demote_batch_to_core."""
        fock = ArrayAnsatz(self.array2578, batch_dims=1)  # 1 batch dim

        # Invalid indices
        with pytest.raises(ValueError, match="All batch indices must be in range"):
            fock.demote_batch_to_core([1])  # index 1 >= 1 batch dim

        with pytest.raises(ValueError, match="All batch indices must be in range"):
            fock.demote_batch_to_core([-1])  # negative index

        # Duplicate indices
        with pytest.raises(ValueError, match="Batch indices must be unique"):
            fock.demote_batch_to_core([0, 0])

    def test_promote_demote_roundtrip(self):
        """Test that promote then demote maintains data integrity."""
        # Start with (2, 5, 7, 8) batch_dims=1
        fock = ArrayAnsatz(self.array2578, batch_dims=1)

        # Promote core dims [0, 2] -> (2, 5, 8, 7) batch_dims=3
        promoted = fock.promote_core_to_batch([0, 2])
        assert promoted.batch_dims == 3
        assert promoted.array.shape == (2, 5, 8, 7)

        # Demote batch dims [1, 2] (the promoted ones) -> (2, 5, 8, 7) batch_dims=1
        demoted = promoted.demote_batch_to_core([1, 2])
        assert demoted.batch_dims == 1
        assert demoted.array.shape == (2, 5, 8, 7)

        # The data should be preserved (though reordered)
        # Check that we can access the same elements
        assert demoted.batch_shape == (2,)
        assert demoted.core_shape == (5, 8, 7)

    def test_promote_core_preserves_data(self):
        """Test that promoting core dimensions preserves the underlying data correctly."""
        # Use a simple array we can track
        array = np.arange(24).reshape((2, 3, 4))  # batch_dims=1
        fock = ArrayAnsatz(array, batch_dims=1)

        # Promote core dim 0 (shape 3) -> (2, 3, 4) batch_dims=2
        promoted = fock.promote_core_to_batch([0])

        # Check that data is preserved correctly
        assert promoted.array.shape == (2, 3, 4)
        assert promoted.batch_dims == 2

        # The original array[0, :, :] should match promoted.array[0, :, :]
        # after accounting for the new batch structure
        for i in range(2):
            for j in range(3):
                assert np.allclose(promoted.array[i, j, :], array[i, j, :])

    def test_demote_batch_preserves_data(self):
        """Test that demoting batch dimensions preserves the underlying data correctly."""
        # Use a simple array we can track
        array = np.arange(24).reshape((2, 3, 4))  # batch_dims=2
        fock = ArrayAnsatz(array, batch_dims=2)

        # Demote batch dim 1 (shape 3) -> (2, 3, 4) batch_dims=1
        demoted = fock.demote_batch_to_core([1])

        # Check that data is preserved correctly
        assert demoted.array.shape == (2, 3, 4)
        assert demoted.batch_dims == 1

        # The data should be the same
        assert np.allclose(demoted.array, array)
