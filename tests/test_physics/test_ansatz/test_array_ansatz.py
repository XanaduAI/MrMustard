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
import pytest
from ipywidgets import HTML, HBox, Tab, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz


class TestArrayAnsatz:
    r"""Tests the array ansatz."""

    array578 = np.random.random((5, 7, 8))
    array1578 = np.random.random((1, 5, 7, 8))
    array2578 = np.random.random((2, 5, 7, 8))
    array5578 = np.random.random((5, 5, 7, 8))

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
        array = np.random.random((2, 4, 5))
        array2 = np.random.random((3, 4, 8, 9))
        aa1 = ArrayAnsatz(array=array, batch_dims=0)
        aa2 = ArrayAnsatz(array=array2, batch_dims=1)

        with pytest.raises(ValueError):
            aa1 + aa2  # pylint: disable=pointless-statement

    def test_and(self):
        array5123 = np.random.random((5, 1, 2, 3))
        array5234 = np.random.random((5, 2, 3, 4))
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
        fock_test = fock1.contract(fock2, batch_str="a,b->ab", idx1=[0, 1], idx2=[0, 1])
        assert fock_test.array.shape == (2, 1, 8, 8)
        assert np.allclose(
            fock_test.array, np.einsum("acde, bcdh -> abeh", self.array2578, self.array1578)
        )

    def test_divide_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_equal(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array)
        assert aa1 == aa2

    def test_from_function(self):
        def gen_array(x):
            return x

        x = math.astensor(np.random.random((5, 8, 8)))
        fock = ArrayAnsatz.from_function(gen_array, batch_dims=1, x=x)
        assert fock.array.shape == (5, 8, 8)
        assert math.allclose(fock.array, x)

    def test_multiply_by_scalar(self):
        fock1 = ArrayAnsatz(self.array1578, batch_dims=1)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_neg(self):
        array = np.random.random((2, 4, 5))
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
        array1 = math.astensor(np.random.random((2, 5, 5, 1)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        assert ArrayAnsatz.from_dict(fock1.to_dict()) == fock1

    def test_trace(self):
        array1 = math.astensor(np.random.random((2, 5, 5, 1, 3, 3)))
        fock1 = ArrayAnsatz(array1, batch_dims=1)
        fock2 = fock1.trace([0, 3], [1, 4])
        assert fock2.array.shape == (2, 1)
        assert np.allclose(fock2.array, np.einsum("abbdee->ad", array1))

    def test_truediv_by_scalar(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 / 6
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array / 6)

    @pytest.mark.parametrize("shape", [(1, 8), (1, 8, 8)])
    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr(self, mock_display, shape):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(np.random.random(shape), batch_dims=1)
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
        rep = ArrayAnsatz(np.random.random((2, 8)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_3_dims_or_less(self, mock_display):
        """Test the IPython repr function does nothing with 4+ dims."""
        rep = ArrayAnsatz(np.random.random((1, 4, 4, 4)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(np.random.random((1, 8)), batch_dims=1)
        rep._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)

    # --- Tests for core_str in contract ---

    @pytest.mark.parametrize(
        "shape1, shape2, core_str, batch_str, expected_shape, einsum_str",
        [
            # Basic matrix multiplication
            ((3, 4), (4, 5), "ab,bc->ac", None, (3, 5), "ab,bc->ac"),
            # Index survival (sum over last dim of first)
            ((3, 4), (4,), "ab,b->a", None, (3,), "ab,b->a"),
            # Index survival (sum over first dim of second)
            ((3, 4), (3, 5), "ab,ac->bc", None, (4, 5), "ab,ac->bc"),
            # Reordering output
            ((3, 4), (5, 4), "ab,cb->ac", None, (3, 5), "ab,cb->ac"),
            # Outer product
            ((2, 3), (4, 5), "ab,cd->abcd", None, (2, 3, 4, 5), "ab,cd->abcd"),
            # Batching: Outer product batch, matrix multiply core
            ((2, 3, 4), (5, 4, 6), "ab,bc->ac", "x,y->xy", (2, 5, 3, 6), "xab,ybc->xyac"),
            # Batching: Shared batch dim, matrix multiply core
            ((2, 3, 4), (2, 4, 6), "ab,bc->ac", "x,x->x", (2, 3, 6), "xab,xbc->xac"),
            # Batching: Outer product batch, index survival core
            ((2, 3, 4), (5, 4), "ab,b->a", "x,y->xy", (2, 5, 3), "xab,yb->xya"),
        ],
    )
    def test_contract_with_core_str(
        self, shape1, shape2, core_str, batch_str, expected_shape, einsum_str
    ):
        """Tests various core_str contraction scenarios."""
        # Determine core dims lengths from core_str input parts
        core_inputs = core_str.split("->")[0].split(",")
        core_len1 = len(core_inputs[0])
        core_len2 = len(core_inputs[1]) if len(core_inputs) > 1 else 0

        # Calculate batch dims based on total shape and core lengths
        batch_dims1 = len(shape1) - core_len1
        batch_dims2 = len(shape2) - core_len2

        array1 = np.random.rand(*shape1) + 1j * np.random.rand(*shape1)
        array2 = np.random.rand(*shape2) + 1j * np.random.rand(*shape2)

        ansatz1 = ArrayAnsatz(array1, batch_dims=batch_dims1)
        ansatz2 = ArrayAnsatz(array2, batch_dims=batch_dims2)

        result_ansatz = ansatz1.contract(other=ansatz2, core_str=core_str, batch_str=batch_str)

        # Verify shape
        assert result_ansatz.array.shape == expected_shape

        # Verify result using numpy.einsum as reference
        expected_result = np.einsum(einsum_str, array1, array2)
        assert np.allclose(result_ansatz.array, expected_result)

    def test_contract_with_core_str_shape_reduction(self):
        """Tests core_str contraction requiring shape reduction."""
        shape1 = (3, 4)  # core "ab"
        shape2 = (5, 5)  # core "bc" -> contracted dim b: size 4 vs 5
        core_str = "ab,bc->ac"
        einsum_str = "ab,bc->ac"

        array1 = np.random.rand(*shape1) + 1j * np.random.rand(*shape1)
        array2 = np.random.rand(*shape2) + 1j * np.random.rand(*shape2)

        ansatz1 = ArrayAnsatz(array1, batch_dims=0)
        ansatz2 = ArrayAnsatz(array2, batch_dims=0)

        result_ansatz = ansatz1.contract(other=ansatz2, core_str=core_str)

        # Expected shape after reduction (contracted dim becomes min(4, 5) = 4)
        expected_shape = (3, 5)  # a=3, c=5
        assert result_ansatz.array.shape == expected_shape

        # Verify result - numpy needs manual slicing for reference
        min_dim = min(shape1[1], shape2[0])  # min(4, 5) = 4
        expected_result = np.einsum(einsum_str, array1[:, :min_dim], array2[:min_dim, :])
        assert np.allclose(result_ansatz.array, expected_result)

        # Test reduction in other direction
        shape1_rev = (5, 5)  # "ab"
        shape2_rev = (3, 4)  # "bc" -> contracted dim b: size 5 vs 3
        array1_rev = np.random.rand(*shape1_rev) + 1j * np.random.rand(*shape1_rev)
        array2_rev = np.random.rand(*shape2_rev) + 1j * np.random.rand(*shape2_rev)
        ansatz1_rev = ArrayAnsatz(array1_rev, batch_dims=0)
        ansatz2_rev = ArrayAnsatz(array2_rev, batch_dims=0)
        result_ansatz_rev = ansatz1_rev.contract(other=ansatz2_rev, core_str=core_str)

        min_dim_rev = min(shape1_rev[1], shape2_rev[0])  # min(5, 3) = 3
        expected_shape_rev = (5, 4)  # a=5, c=4
        assert result_ansatz_rev.array.shape == expected_shape_rev
        expected_result_rev = np.einsum(
            einsum_str, array1_rev[:, :min_dim_rev], array2_rev[:min_dim_rev, :]
        )
        assert np.allclose(result_ansatz_rev.array, expected_result_rev)

    def test_contract_error_mixed_args(self):
        """Tests error when both core_str and idx1/idx2 are provided."""
        ansatz1 = ArrayAnsatz(np.random.rand(2, 3))
        ansatz2 = ArrayAnsatz(np.random.rand(3, 4))
        with pytest.raises(ValueError, match="Cannot specify both `core_str` and `idx1`/`idx2`"):
            ansatz1.contract(ansatz2, core_str="ab,bc->ac", idx1=(1,), idx2=(0,))

    def test_contract_error_missing_args(self):
        """Tests error when neither core_str nor idx1/idx2 are provided."""
        ansatz1 = ArrayAnsatz(np.random.rand(2, 3))
        ansatz2 = ArrayAnsatz(np.random.rand(3, 4))
        with pytest.raises(
            ValueError, match="Either `core_str` or both `idx1` and `idx2` must be provided"
        ):
            ansatz1.contract(ansatz2)  # No core specification

    @pytest.mark.parametrize(
        "core_str",
        [
            "ab,bc",  # missing "->"
            "ab->ac",  # missing second input
            "ab,bc,cd->ad",  # too many inputs
            "ab,bc->a->c",  # too many "->"
        ],
    )
    def test_contract_error_invalid_core_str_format(self, core_str):
        """Tests error on invalid core_str format."""
        ansatz1 = ArrayAnsatz(np.random.rand(2, 3))
        ansatz2 = ArrayAnsatz(np.random.rand(3, 4))
        with pytest.raises(ValueError, match="Invalid core_str format"):
            ansatz1.contract(ansatz2, core_str=core_str)

    def test_contract_error_core_str_out_of_bounds(self):
        """Tests error when core_str implies more dimensions than available."""
        ansatz1 = ArrayAnsatz(np.random.rand(2, 3))  # core_dims = 2
        ansatz2 = ArrayAnsatz(np.random.rand(3, 4))  # core_dims = 2

        # Test 1: core_str part for ansatz1 ("abc") has length 3, but core_dims is 2
        # The error happens when processing the character 'c' at index i=2.
        with pytest.raises(
            IndexError, match="Index for core char 'c' \\(2\\) out of bounds for self \\(dims=2\\)"
        ):
            ansatz1.contract(ansatz2, core_str="abc,de->ae")

        # Test 2: core_str part for ansatz2 ("def") has length 3, but core_dims is 2
        # The error happens when processing the character 'f' at index i=2.
        with pytest.raises(
            IndexError, match="Index for core char 'f' \\(2\\) out of bounds for other \\(dims=2\\)"
        ):
            ansatz1.contract(ansatz2, core_str="ab,def->ae")

    # --- End tests for core_str ---

    @pytest.mark.parametrize("shape", [(1, 8), (1, 8, 8)])
    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr(self, mock_display, shape):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(np.random.random(shape), batch_dims=1)
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
        rep = ArrayAnsatz(np.random.random((2, 8)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.physics.ansatz.array_ansatz.display")
    def test_ipython_repr_expects_3_dims_or_less(self, mock_display):
        """Test the IPython repr function does nothing with 4+ dims."""
        rep = ArrayAnsatz(np.random.random((1, 4, 4, 4)), batch_dims=1)
        rep._ipython_display_()
        mock_display.assert_not_called()

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        rep = ArrayAnsatz(np.random.random((1, 8)), batch_dims=1)
        rep._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)
