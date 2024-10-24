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

# pylint: disable = missing-function-docstring, disable=too-many-public-methods

from unittest.mock import patch

import numpy as np
from ipywidgets import HBox, VBox, HTML, Tab
from plotly.graph_objs import FigureWidget
import pytest

from mrmustard import math
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz


class TestArrayAnsatz:
    r"""Tests the array ansatz."""

    array578 = np.random.random((5, 7, 8))
    array1578 = np.random.random((1, 5, 7, 8))
    array2578 = np.random.random((2, 5, 7, 8))
    array5578 = np.random.random((5, 5, 7, 8))

    def test_init_batched(self):
        fock = ArrayAnsatz(self.array1578, batched=True)
        assert isinstance(fock, ArrayAnsatz)
        assert np.allclose(fock.array, self.array1578)

    def test_init_non_batched(self):
        fock = ArrayAnsatz(self.array578, batched=False)
        assert isinstance(fock, ArrayAnsatz)
        assert fock.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock.array[0, :, :, :], self.array578)

    def test_add(self):
        fock1 = ArrayAnsatz(self.array2578, batched=True)
        fock2 = ArrayAnsatz(self.array5578, batched=True)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] + self.array5578[0])
        assert np.allclose(fock1_add_fock2.array[4], self.array2578[0] + self.array5578[4])
        assert np.allclose(fock1_add_fock2.array[5], self.array2578[1] + self.array5578[0])

    def test_algebra_with_different_shape_of_array_raise_errors(self):
        array = np.random.random((2, 4, 5))
        array2 = np.random.random((3, 4, 8, 9))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)

        with pytest.raises(Exception, match="Cannot add"):
            aa1 + aa2  # pylint: disable=pointless-statement

        with pytest.raises(Exception, match="Cannot add"):
            aa1 - aa2  # pylint: disable=pointless-statement

        with pytest.raises(Exception, match="Cannot multiply"):
            aa1 * aa2  # pylint: disable=pointless-statement

        with pytest.raises(Exception, match="Cannot divide"):
            aa1 / aa2  # pylint: disable=pointless-statement

        with pytest.raises(Exception):
            aa1 == aa2  # pylint: disable=pointless-statement

    def test_and(self):
        fock1 = ArrayAnsatz(self.array1578, batched=True)
        fock2 = ArrayAnsatz(self.array5578, batched=True)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 5, 7, 8, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgh -> bpcdefgh", self.array1578, self.array5578), -1),
        )

    def test_call(self):
        fock = ArrayAnsatz(self.array1578, batched=True)
        with pytest.raises(AttributeError, match="Cannot call"):
            fock(0)

    def test_conj(self):
        fock = ArrayAnsatz(self.array1578, batched=True)
        fock_conj = fock.conj
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_divide_on_a_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batched=True)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_equal(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array)
        assert aa1 == aa2

    def test_matmul_fock_fock(self):
        array2 = math.astensor(np.random.random((5, 6, 7, 8, 10)))
        fock1 = ArrayAnsatz(self.array2578, batched=True)
        fock2 = ArrayAnsatz(array2, batched=True)
        fock_test = fock1[2] @ fock2[2]
        assert fock_test.array.shape == (10, 5, 7, 6, 7, 10)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgeh -> bpcdfgh", self.array2578, array2), -1),
        )

    def test_mul(self):
        fock1 = ArrayAnsatz(self.array1578, batched=True)
        fock2 = ArrayAnsatz(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 * fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, self.array5578), -1),
        )

    def test_multiply_a_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batched=True)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_neg(self):
        array = np.random.random((2, 4, 5))
        aa = ArrayAnsatz(array=array)
        minusaa = -aa
        assert isinstance(minusaa, ArrayAnsatz)
        assert np.allclose(minusaa.array, -array)

    @pytest.mark.parametrize("batched", [True, False])
    def test_reduce(self, batched):
        shape = (1, 3, 3, 3) if batched else (3, 3, 3)
        array1 = math.astensor(np.arange(27).reshape(shape))
        fock1 = ArrayAnsatz(array1, batched=batched)

        fock2 = fock1.reduce(3)
        assert fock1 == fock2

        fock3 = fock1.reduce(2)
        array3 = math.astensor([[[0, 1], [3, 4]], [[9, 10], [12, 13]]])
        assert fock3 == ArrayAnsatz(array3)

        fock4 = fock1.reduce((1, 3, 1))
        array4 = math.astensor([[[0], [3], [6]]])
        assert fock4 == ArrayAnsatz(array4)

    def test_reduce_error(self):
        array1 = math.astensor(np.arange(27).reshape((3, 3, 3)))
        fock1 = ArrayAnsatz(array1)

        with pytest.raises(ValueError, match="Expected shape"):
            fock1.reduce((1, 2))

        with pytest.raises(ValueError, match="Expected shape"):
            fock1.reduce((1, 2, 3, 4, 5))

    def test_reduce_padded(self):
        fock = ArrayAnsatz(self.array578)
        with pytest.warns(UserWarning):
            fock1 = fock.reduce((8, 8, 8))
        assert fock1.array.shape == (1, 8, 8, 8)

    def test_reorder(self):
        array1 = math.astensor(np.arange(8).reshape((1, 2, 2, 2)))
        fock1 = ArrayAnsatz(array1, batched=True)
        fock2 = fock1.reorder(order=(2, 1, 0))
        assert np.allclose(fock2.array, np.array([[[[0, 4], [2, 6]], [[1, 5], [3, 7]]]]))
        assert np.allclose(fock2.array, np.arange(8).reshape((1, 2, 2, 2), order="F"))

    def test_sub(self):
        fock1 = ArrayAnsatz(self.array2578, batched=True)
        fock2 = ArrayAnsatz(self.array5578, batched=True)
        fock1_sub_fock2 = fock1 - fock2
        assert fock1_sub_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_sub_fock2.array[0], self.array2578[0] - self.array5578[0])
        assert np.allclose(fock1_sub_fock2.array[4], self.array2578[0] - self.array5578[4])
        assert np.allclose(fock1_sub_fock2.array[9], self.array2578[1] - self.array5578[4])

    def test_sum_batch(self):
        fock = ArrayAnsatz(self.array2578, batched=True)
        fock_collapsed = fock.sum_batch()[0]
        assert fock_collapsed.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock_collapsed.array, np.sum(self.array2578, axis=0))

    def test_to_from_dict(self):
        array1 = math.astensor(np.random.random((2, 5, 5, 1, 7, 4, 1, 7, 3)))
        fock1 = ArrayAnsatz(array1, batched=True)
        assert ArrayAnsatz.from_dict(fock1.to_dict()) == fock1

    def test_trace(self):
        array1 = math.astensor(np.random.random((2, 5, 5, 1, 7, 4, 1, 7, 3)))
        fock1 = ArrayAnsatz(array1, batched=True)
        fock2 = fock1.trace([0, 3], [1, 6])
        assert fock2.array.shape == (2, 1, 4, 1, 3)
        assert np.allclose(fock2.array, np.einsum("bccefghfj -> beghj", array1))

    def test_truediv(self):
        fock1 = ArrayAnsatz(self.array1578, batched=True)
        fock2 = ArrayAnsatz(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 / fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, 1 / self.array5578), -1),
        )

    def test_truediv_a_scalar(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 / 6
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array / 6)

    @pytest.mark.parametrize("shape", [(1, 8), (1, 8, 8)])
    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr(self, mock_display, shape):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(np.random.random(shape), batched=True)
        rep._ipython_display_()
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
        rep = ArrayAnsatz(np.random.random((2, 8)), batched=True)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_3_dims_or_less(self, mock_display):
        """Test the IPython repr function does nothing with 4+ dims."""
        rep = ArrayAnsatz(np.random.random((1, 4, 4, 4)), batched=True)
        rep._ipython_display_()
        mock_display.assert_not_called()
