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

    array578 = settings.get_rng().random((5, 7, 8))
    array1578 = settings.get_rng().random((1, 5, 7, 8))
    array2578 = settings.get_rng().random((2, 5, 7, 8))
    array5578 = settings.get_rng().random((5, 5, 7, 8))

    def test_add(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock1_add_fock1 = fock1 + fock1
        assert fock1_add_fock1.array.shape == (2, 5, 7, 8)
        assert np.allclose(fock1_add_fock1.array[0], self.array2578[0] + self.array2578[0])
        assert np.allclose(fock1_add_fock1.array[1], self.array2578[1] + self.array2578[1])

    def test_and(self):
        array5123 = settings.get_rng().random((5, 1, 2, 3))
        array5234 = settings.get_rng().random((5, 2, 3, 4))
        fock1 = ArrayAnsatz(array=array5123, batch_dims=1)
        fock2 = ArrayAnsatz(array=array5234, batch_dims=1)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 1, 2, 3, 2, 3, 4)
        assert np.allclose(fock_test.array, np.einsum("abcd,aefg->abcdefg", array5123, array5234))

    def test_batch_getitem_int_slice_none_ellipsis(self):
        array = math.arange(2 * 3 * 4).reshape((2, 3, 4))
        ans = ArrayAnsatz(array, batch_dims=1)

        # int reduces batch dims
        sub = ans[1]
        assert isinstance(sub, ArrayAnsatz)
        assert sub.batch_dims == 0
        assert sub.core_shape == (3, 4)

        # slice preserves batch dims
        sub = ans[:]
        assert sub.batch_dims == 1
        assert sub.core_shape == (3, 4)

        # None inserts a batch dim
        sub = ans[None]
        assert sub.batch_dims == 2
        assert sub.array.shape[:2] == (1, 2)

        # Ellipsis fills remaining batch axes
        ans2 = ArrayAnsatz(array.reshape((2, 1, 3, 4)), batch_dims=2)
        sub = ans2[..., None]
        assert sub.batch_dims == 3
        assert sub.array.shape[:3] == (2, 1, 1)

    def test_call(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        with pytest.raises(AttributeError, match="Cannot call"):
            fock(0)

    def test_concat_axis_out_of_range_error(self):
        """Test that concatenation fails when axis is out of range."""
        array1 = settings.get_rng().random((2, 4, 5))
        array2 = settings.get_rng().random((3, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        ansatz2 = ArrayAnsatz(array2, batch_dims=1)

        with pytest.raises(ValueError, match=r"axis .* is out of range"):
            ansatz1.concat(ansatz2, axis=5)

    def test_concat_basic(self):
        """Test basic concatenation of two array ansatze along axis 0."""
        array1 = settings.get_rng().random((2, 4, 5))
        array2 = settings.get_rng().random((3, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        ansatz2 = ArrayAnsatz(array2, batch_dims=1)

        concatenated = ansatz1.concat(ansatz2, axis=0)

        assert concatenated.batch_shape == (5,)
        assert concatenated.core_shape == (4, 5)
        assert math.allclose(concatenated.array[:2], array1)
        assert math.allclose(concatenated.array[2:], array2)

    def test_concat_mismatch_batch_dims_error(self):
        """Test that concatenation fails when batch_dims don't match."""
        array1 = settings.get_rng().random((2, 4, 5))
        array2 = settings.get_rng().random((3, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        ansatz2 = ArrayAnsatz(array2, batch_dims=0)

        with pytest.raises(ValueError, match="different number of batch dimensions"):
            ansatz1.concat(ansatz2, axis=0)

    def test_concat_mismatch_batch_shape_error(self):
        """Test that concatenation fails when batch shapes don't match at non-concat axes."""
        array1 = settings.get_rng().random((2, 3, 4, 5))
        array2 = settings.get_rng().random((2, 5, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=2)
        ansatz2 = ArrayAnsatz(array2, batch_dims=2)

        with pytest.raises(ValueError, match="Batch shapes must match except at axis"):
            ansatz1.concat(
                ansatz2, axis=0
            )  # axis 0: sizes 2 and 2 match, axis 1: sizes 3 and 5 don't match

    def test_concat_mismatch_core_shape_error(self):
        """Test that concatenation fails when core shapes don't match."""
        array1 = settings.get_rng().random((2, 4, 5))
        array2 = settings.get_rng().random((3, 4, 6))
        ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        ansatz2 = ArrayAnsatz(array2, batch_dims=1)

        with pytest.raises(ValueError, match="different core_shape"):
            ansatz1.concat(ansatz2, axis=0)

    def test_concat_multidimensional_batch(self):
        """Test concatenation with multi-dimensional batch shapes."""
        array1 = settings.get_rng().random((2, 3, 4, 5))
        array2 = settings.get_rng().random((5, 3, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=2)
        ansatz2 = ArrayAnsatz(array2, batch_dims=2)

        # Concatenate along axis 0
        concatenated = ansatz1.concat(ansatz2, axis=0)
        assert concatenated.batch_shape == (7, 3)
        assert concatenated.core_shape == (4, 5)
        assert math.allclose(concatenated.array[:2], array1)
        assert math.allclose(concatenated.array[2:], array2)

        # Concatenate along axis 1
        array3 = settings.get_rng().random((2, 4, 4, 5))
        ansatz3 = ArrayAnsatz(array3, batch_dims=2)
        concatenated2 = ansatz1.concat(ansatz3, axis=1)
        assert concatenated2.batch_shape == (2, 7)
        assert concatenated2.core_shape == (4, 5)

    def test_concat_negative_axis(self):
        """Test concatenation with negative axis index."""
        array1 = settings.get_rng().random((2, 3, 4, 5))
        array2 = settings.get_rng().random((2, 5, 4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=2)
        ansatz2 = ArrayAnsatz(array2, batch_dims=2)

        # Concatenate along axis -1 (should be axis 1)
        concatenated = ansatz1.concat(ansatz2, axis=-1)
        assert concatenated.batch_shape == (2, 8)

    def test_concat_no_batch(self):
        """Test concatenation fails appropriately when there are no batch dims."""
        array1 = settings.get_rng().random((4, 5))
        array2 = settings.get_rng().random((4, 5))
        ansatz1 = ArrayAnsatz(array1, batch_dims=0)
        ansatz2 = ArrayAnsatz(array2, batch_dims=0)

        with pytest.raises(ValueError, match=r"axis .* is out of range"):
            ansatz1.concat(ansatz2, axis=0)

    def test_conj(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_conj = fock.conj
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_contract_fock_fock(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock2 = ArrayAnsatz(self.array2578, batch_dims=1)
        # contract first two core axes (5 and 7)
        fock_test = fock1.contract(fock2, ([0, 1], [0, 1]))
        assert fock_test.array.shape == (2, 8, 8)
        assert np.allclose(
            fock_test.array, np.einsum("acde, acdh -> aeh", self.array2578, self.array2578)
        )

    def test_divide_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_equal(self):
        array = settings.get_rng().random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array)
        assert aa1 == aa2

    def test_init_batched(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=1)
        assert isinstance(fock, ArrayAnsatz)
        assert np.allclose(fock.array, self.array1578)

    def test_init_non_batched(self):
        fock = ArrayAnsatz(self.array578, batch_dims=0)
        assert isinstance(fock, ArrayAnsatz)
        assert fock.array.shape == (5, 7, 8)
        assert np.allclose(fock.array, self.array578)

    @pytest.mark.parametrize("shape", [(1, 8), (1, 8, 8)])
    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr(self, mock_display, shape):
        """Test the IPython repr function."""
        rep_np = ArrayAnsatz(settings.get_rng().random(shape), batch_dims=1)
        rep_np.display()
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
        rep = ArrayAnsatz(settings.get_rng().random((2, 8)), batch_dims=1)
        rep.display()
        mock_display.assert_not_called()

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_3_dims_or_less(self, mock_display):
        """Test the IPython repr function does nothing with 4+ dims."""
        rep = ArrayAnsatz(settings.get_rng().random((1, 4, 4, 4)), batch_dims=1)
        rep.display()
        mock_display.assert_not_called()

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(settings.get_rng().random((1, 8)), batch_dims=1)
        rep.display()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)

    def test_multiply_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_neg(self):
        array = settings.get_rng().random((2, 4, 5))
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

    def test_reorder_batch(self):
        fock = ArrayAnsatz(self.array1578, batch_dims=2)
        fock_reordered = fock.reorder_batch([1, 0])
        assert fock_reordered.array.shape == (5, 1, 7, 8)

    def test_sub(self):
        fock1 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock2 = ArrayAnsatz(self.array2578, batch_dims=1)
        fock1_add_fock2 = fock1 - fock2
        assert fock1_add_fock2.array.shape == (2, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] - self.array2578[0])
        assert np.allclose(fock1_add_fock2.array[1], self.array2578[1] - self.array2578[1])

    def test_sum_with_different_batch_raise_errors(self):
        array = settings.get_rng().random((2, 4, 5))
        array2 = settings.get_rng().random((3, 4, 8, 9))
        aa1 = ArrayAnsatz(array=array, batch_dims=0)
        aa2 = ArrayAnsatz(array=array2, batch_dims=1)

        with pytest.raises(ValueError):
            aa1 + aa2

    def test_to_from_dict(self):
        array1 = math.astensor(settings.get_rng().random((2, 5, 5, 1)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        assert ArrayAnsatz.from_dict(fock1.to_dict()) == fock1

    def test_trace(self):
        array1 = math.astensor(settings.get_rng().random((2, 5, 5, 1, 3, 3)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        fock2 = fock1.trace([0, 3], [1, 4])
        assert fock2.array.shape == (2, 1)
        assert np.allclose(fock2.array, np.einsum("abbdee->ad", array1))

    def test_truediv_by_scalar(self):
        array = settings.get_rng().random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 / 6
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array / 6)
