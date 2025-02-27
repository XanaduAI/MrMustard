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

"""This module contains tests for ``PolyExpAnsatz`` objects."""

# pylint: disable = too-many-public-methods, missing-function-docstring

from unittest.mock import patch

import numpy as np
import pytest
from ipywidgets import HTML, Box, IntSlider, IntText, Stack, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz
from mrmustard.physics.ansatz.polyexp_ansatz import PolyExpAnsatz
from mrmustard.physics.gaussian_integrals import (
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
)

from ...random import Abc_triple


class TestPolyExpAnsatz:
    r"""
    Tests the polyexp ansatz.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_non_batched(self, triple):
        A, b, c = triple
        bargmann = PolyExpAnsatz(*triple)

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_add(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = PolyExpAnsatz(*triple1)
        bargmann2 = PolyExpAnsatz(*triple2)
        bargmann_add = bargmann1 + bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.random(size=(1, 3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.random(size=(1, 2, 2))

        bargmann3 = PolyExpAnsatz(A1, b1, c1)
        bargmann4 = PolyExpAnsatz(A2, b2, c2)

        bargmann_add2 = bargmann3 + bargmann4

        assert np.allclose(bargmann_add2.A[0], A1)
        assert np.allclose(bargmann_add2.b[0], b1)
        assert np.allclose(bargmann_add2.c[0], c1[0])
        assert np.allclose(bargmann_add2.A[1], A2)
        assert np.allclose(bargmann_add2.b[1], b2)
        assert np.allclose(bargmann_add2.c[1][:2, :2], c2[0])

    def test_add_different_poly_wires(self):
        "tests that A and b are padded correctly"
        A1 = np.random.random((1, 2, 2))
        A2 = np.random.random((1, 3, 3))
        b1 = np.random.random((1, 2))
        b2 = np.random.random((1, 3))
        c1 = np.random.random((1,))
        c2 = np.random.random((1, 11))
        ansatz1 = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2, num_derived_vars=1)
        ansatz_sum = ansatz1 + ansatz2
        assert ansatz_sum.A.shape == (2, 3, 3)
        assert ansatz_sum.b.shape == (2, 3)
        assert ansatz_sum.c.shape == (2, 11)
        ansatz_sum = ansatz2 + ansatz1
        assert ansatz_sum.A.shape == (2, 3, 3)
        assert ansatz_sum.b.shape == (2, 3)
        assert ansatz_sum.c.shape == (2, 11)

    def test_add_error(self):
        bargmann = PolyExpAnsatz(*Abc_triple(3))
        fock = ArrayAnsatz(np.random.random((1, 4, 4, 4)), batched=True)

        with pytest.raises(TypeError, match="Cannot add"):
            bargmann + fock  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = PolyExpAnsatz(*triple1) & PolyExpAnsatz(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    def test_call(self):
        A, b, c = Abc_triple(5)
        ansatz = PolyExpAnsatz(A, b, c)

        assert np.allclose(ansatz(*math.zeros_like(b)), c)

        A, b, _ = Abc_triple(4)
        c = np.random.random(size=(1, 3, 3, 3))
        ansatz = PolyExpAnsatz(A, b, c, num_derived_vars=3)
        z = np.random.uniform(-10, 10, size=(7, 2))
        with pytest.raises(Exception, match="The ansatz was called with"):
            ansatz(*z)

        A = np.array([[0.0, 1.0], [1.0, 0.0]]) + 0.0j
        b = np.zeros(2) + 0.0j
        c = np.zeros(10, dtype=complex).reshape(1, -1) + 0.0j
        c[0, -1] = 1
        obj1 = PolyExpAnsatz(A, b, c, num_derived_vars=1)

        nine_factorial = np.prod(np.arange(1, 9))
        assert np.allclose(obj1(np.array([0.1 + 0.0j])), 0.1**9 / np.sqrt(nine_factorial))

    def test_partial_eval(self):
        A1, b1, _ = Abc_triple(4)
        A2, b2, _ = Abc_triple(4)
        A3, b3, _ = Abc_triple(4)

        batch = 3
        c = np.random.random(size=(batch, 5, 5)) / 1000 + 0.0j

        obj = PolyExpAnsatz([A1, A2, A3], [b1, b2, b3], c, num_derived_vars=2)
        z0 = [None, 2.0 + 0.0j]
        z1 = [1.0 + 0.0j]
        z2 = [1.0 + 0.0j, 2.0 + 0.0j]
        val_full = obj(*z2)
        partial = obj(*z0)
        val_partial = partial(*z1)
        assert np.allclose(val_partial, val_full)

        A1, b1, _ = Abc_triple(4)
        A2, b2, _ = Abc_triple(4)

        batch = 2
        c = np.random.random(size=(2, 5)) / 1000 + 0.0j

        obj = PolyExpAnsatz([A1, A2], [b1, b2], c, num_derived_vars=1)
        z0 = [None, 2.0 + 0.0j, None]
        z1 = [1.0 + 0.0j, 3.0 + 0.0j]
        z2 = [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]
        val_full = obj(*z2)
        partial = obj(*z0)
        val_partial = partial(*z1)
        assert np.allclose(val_partial, val_full)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = PolyExpAnsatz(*triple).conj

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    def test_contract_barg_barg(self):
        triple1 = Abc_triple(3)
        triple2 = Abc_triple(3)

        res1 = PolyExpAnsatz(*triple1).contract(PolyExpAnsatz(*triple2))
        exp1 = complex_gaussian_integral_2(triple1, triple2, [], [])
        assert np.allclose(res1.A, exp1[0])
        assert np.allclose(res1.b, exp1[1])
        assert np.allclose(res1.c, exp1[2])

        res2 = PolyExpAnsatz(*triple1).contract(PolyExpAnsatz(*triple2), idx1=0, idx2=0)
        exp2 = complex_gaussian_integral_2(triple1, triple2, [0], [0])
        assert np.allclose(res2.A, exp2[0])
        assert np.allclose(res2.b, exp2[1])
        assert np.allclose(res2.c, exp2[2])

    def test_decompose_ansatz(self):
        A, b, _ = Abc_triple(4)
        c = np.random.uniform(-10, 10, size=(1, 3, 3, 3)) + 0.0j
        ansatz = PolyExpAnsatz(A, b, c, num_derived_vars=3)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(1,)) + 0.0j
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (1, 2, 2))

        c2 = np.random.uniform(-10, 10, size=(1, 4)) + 0.0j
        ansatz2 = PolyExpAnsatz(A, b, c2, num_derived_vars=1)
        decomp_ansatz2 = ansatz2.decompose_ansatz()
        assert np.allclose(decomp_ansatz2.A, ansatz2.A)

    def test_decompose_ansatz_batch(self):
        """
        In this test the batch dimension of both ``z`` and ``Abc`` is tested.
        """
        A1, b1, _ = Abc_triple(4)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3)) + 0.0j
        A2, b2, _ = Abc_triple(4)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3)) + 0.0j
        ansatz = PolyExpAnsatz([A1, A2], [b1, b2], [c1, c2], num_derived_vars=3)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(1,)) + 0.0j
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 2, 2))
        assert np.allclose(decomp_ansatz.b.shape, (2, 2))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3)) + 0.0j
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3)) + 0.0j
        ansatz = PolyExpAnsatz([A1, A2], [b1, b2], [c1, c2], num_derived_vars=3)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(4,)) + 0.0j
        assert np.allclose(ansatz(z, z), decomp_ansatz(z, z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 4, 4))
        assert np.allclose(decomp_ansatz.b.shape, (2, 4))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9, 9))

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = PolyExpAnsatz(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

    def test_eq(self):
        A, b, c = Abc_triple(5)

        ansatz = PolyExpAnsatz(A, b, c)
        ansatz2 = PolyExpAnsatz(2 * A, 2 * b, 2 * c)

        assert ansatz == ansatz  # pylint: disable= comparison-with-itself
        assert ansatz2 == ansatz2  # pylint: disable= comparison-with-itself
        assert ansatz != ansatz2
        assert ansatz2 != ansatz

    def test_inconsistent_poly_shapes(self):
        A1 = np.random.random((1, 2, 2))
        A2 = np.random.random((1, 3, 3))
        b1 = np.random.random((1, 2))
        b2 = np.random.random((1, 3))
        c1 = np.random.random((1,))
        c2 = np.random.random((1, 5, 11))
        ansatz1 = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        with pytest.raises(ValueError):
            ansatz1 + ansatz2  # pylint: disable=pointless-statement

    @patch("mrmustard.physics.ansatz.polyexp_ansatz.display")
    def test_ipython_repr(self, mock_display):
        """Test the IPython repr function."""
        rep = PolyExpAnsatz(*Abc_triple(2))
        rep._ipython_display_()
        [box] = mock_display.call_args.args
        assert isinstance(box, Box)
        assert box.layout.max_width == "50%"

        # data on left, eigvals on right
        [data_vbox, eigs_vbox] = box.children
        assert isinstance(data_vbox, VBox)
        assert isinstance(eigs_vbox, VBox)

        # data forms a stack: header, ansatz, triple data
        [header, sub, table] = data_vbox.children
        assert isinstance(header, HTML)
        assert isinstance(sub, IntText)
        assert isinstance(table, HTML)

        # eigvals have a header and a unit circle plot
        [eig_header, unit_circle] = eigs_vbox.children
        assert isinstance(eig_header, HTML)
        assert isinstance(unit_circle, FigureWidget)

    @patch("mrmustard.physics.ansatz.polyexp_ansatz.display")
    def test_ipython_repr_batched(self, mock_display):
        """Test the IPython repr function for a batched repr."""
        A1, b1, c1 = Abc_triple(2)
        A2, b2, c2 = Abc_triple(2)
        rep = PolyExpAnsatz(np.array([A1, A2]), np.array([b1, b2]), np.array([c1, c2]))
        rep._ipython_display_()
        [vbox] = mock_display.call_args.args
        assert isinstance(vbox, VBox)

        [slider, stack] = vbox.children
        assert isinstance(slider, IntSlider)
        assert slider.max == 1  # the batch size - 1
        assert isinstance(stack, Stack)

        # max_width is spot-check that this is bargmann widget
        assert len(stack.children) == 2
        assert all(box.layout.max_width == "50%" for box in stack.children)

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        rep = PolyExpAnsatz(*Abc_triple(2))
        rep._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mul(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = PolyExpAnsatz(*triple1)
        bargmann2 = PolyExpAnsatz(*triple2)
        bargmann_mul = bargmann1 * bargmann2

        assert np.allclose(bargmann_mul.A, bargmann1.A + bargmann2.A)
        assert np.allclose(bargmann_mul.b, bargmann1.b + bargmann2.b)
        assert np.allclose(bargmann_mul.c, bargmann1.c * bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = PolyExpAnsatz(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

    def test_order_batch(self):
        ansatz = PolyExpAnsatz(
            A=[np.array([[0]]), np.array([[1]])],
            b=[np.array([1]), np.array([0])],
            c=[1, 2],
        )
        ansatz._order_batch()

        assert np.allclose(ansatz.A[0], np.array([[1]]))
        assert np.allclose(ansatz.b[0], np.array([0]))
        assert ansatz.c[0] == 2
        assert np.allclose(ansatz.A[1], np.array([[0]]))
        assert np.allclose(ansatz.b[1], np.array([1]))
        assert ansatz.c[1] == 1

    def test_polynomial_shape(self):
        A, b, _ = Abc_triple(4)
        c = np.array([[1, 2, 3]])
        ansatz = PolyExpAnsatz(A, b, c, num_derived_vars=1)

        poly_dim = ansatz.num_derived_vars
        poly_shape = ansatz.shape_derived_vars
        assert np.allclose(poly_dim, 1)
        assert np.allclose(poly_shape, (3,))

        A1, b1, _ = Abc_triple(4)
        c1 = np.array([[1, 2, 3]])
        ansatz1 = PolyExpAnsatz(A1, b1, c1, num_derived_vars=1)

        A2, b2, _ = Abc_triple(4)
        c2 = np.array([[1, 2, 3]])
        ansatz2 = PolyExpAnsatz(A2, b2, c2, num_derived_vars=1)

        ansatz3 = ansatz1 * ansatz2

        poly_dim = ansatz3.num_derived_vars
        poly_shape = ansatz3.shape_derived_vars
        assert np.allclose(poly_dim, 2)
        assert np.allclose(poly_shape, (3, 3))

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = PolyExpAnsatz(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    def test_simplify(self):
        A, b, c = Abc_triple(5)

        ansatz = PolyExpAnsatz(A, b, c)

        ansatz = ansatz + ansatz

        assert np.allclose(ansatz.A[0], ansatz.A[1])
        assert np.allclose(ansatz.A[0], A)
        assert np.allclose(ansatz.b[0], ansatz.b[1])
        assert np.allclose(ansatz.b[0], b)

        ansatz.simplify()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert ansatz.c == 2 * c

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = PolyExpAnsatz(*triple1)
        bargmann2 = PolyExpAnsatz(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0))

    def test_trace(self):
        triple = Abc_triple(4)
        bargmann = PolyExpAnsatz(*triple).trace([0], [2])
        A, b, c = complex_gaussian_integral_1(triple, [0], [2])

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    def test_batch_evaluation(self):
        """Test various batch evaluation scenarios for PolyExpAnsatz."""
        # Create a batched ansatz with batch size 4
        A = np.random.random((3, 3))
        A = A + A.T  # Make it symmetric
        b = np.random.random(3)
        c = np.random.random()
        ansatz = PolyExpAnsatz([A, A, A, A], [b, b, b, b], [c, c, c, c], num_derived_vars=0)

        # Test batch size
        assert ansatz.batch_size == 4

        # Test evaluation at a single point
        val = ansatz(1.0, 2.0, 3.0)
        assert val.shape == (4,)

        # Test evaluation at multiple points with 'kron' mode (Cartesian product)
        vals_kron = ansatz([1.0, 2.0], [0.3, -0.34], [0.4, 0.5, 0.3], batch_mode="kron")
        assert vals_kron.shape == (4, 2, 2, 3)

        # Test evaluation at multiple points with 'zip' mode (element-wise)
        vals_zip = ansatz([1.0, 2.0], [0.3, -0.34], [0.4, 0.5], batch_mode="zip")
        assert vals_zip.shape == (4, 2)

        # Test partial evaluation at a single point
        new_ansatz = ansatz(1.0, 1.0, None)
        assert new_ansatz.batch_size == 4

        # Test partial evaluation with batch
        new_ansatz_batch = ansatz([1.0, 2.0], [1.0, 2.0], None)
        assert new_ansatz_batch.batch_size == 8  # 4 * 2 = 8

        # Test full control with explicit eval
        points = np.random.random((7, 2, 5, 3))
        vals_eval = ansatz.eval(points)
        assert vals_eval.shape == (4, 7, 2, 5)

    def test_batch_mode_parameter(self):
        """Test the batch_mode parameter in the __call__ method."""
        # Create a simple ansatz
        A = np.random.random((3, 3))
        A = A + A.T  # Make it symmetric
        b = np.random.random(3)
        c = np.random.random()
        ansatz = PolyExpAnsatz(A, b, c)

        # Test default mode (kron)
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        z = np.array([5.0, 6.0])

        # Default mode is 'kron'
        result_default = ansatz(x, y, z)
        result_kron = ansatz(x, y, z, batch_mode="kron")
        assert np.allclose(result_default, result_kron)
        assert result_kron.shape == (1, 2, 2, 2)  # (len(x), len(y), len(z))

        # Test 'zip' mode
        result_zip = ansatz(x, y, z, batch_mode="zip")
        assert result_zip.shape == (1, 2)  # (batch_size, len(x))

        # Test error with mismatched batch sizes in zip mode
        x_long = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="all z vectors must have the same batch size"):
            ansatz(x_long, y, z, batch_mode="zip")

        # Test error with multi-dimensional arrays in kron mode
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            ansatz(x_2d, y, z, batch_mode="kron")

    def test_eval_method(self):
        """Test the eval method with various batch shapes."""
        # Create a batched ansatz
        A = np.random.random((3, 3))
        A = A + A.T  # Make it symmetric
        b = np.random.random(3)
        c = np.random.random()
        ansatz = PolyExpAnsatz([A, A, A], [b, b, b], [c, c, c], num_derived_vars=0)

        # Test with 1D batch shape
        points_1d = np.random.random((5, 3))  # 5 points, 3 variables
        result_1d = ansatz.eval(points_1d)
        assert result_1d.shape == (3, 5)  # (batch_size, num_points)

        # Test with 2D batch shape
        points_2d = np.random.random((4, 6, 3))  # 4x6 grid of points, 3 variables
        result_2d = ansatz.eval(points_2d)
        assert result_2d.shape == (3, 4, 6)  # (batch_size, *batch_shape)

        # Test with 3D batch shape
        points_3d = np.random.random((2, 3, 4, 3))  # 2x3x4 grid of points, 3 variables
        result_3d = ansatz.eval(points_3d)
        assert result_3d.shape == (3, 2, 3, 4)  # (batch_size, *batch_shape)

        # Test error with incorrect number of variables
        points_wrong = np.random.random((5, 4))  # 5 points, 4 variables (should be 3)
        with pytest.raises(ValueError, match="must equal the number of CV variables"):
            ansatz.eval(points_wrong)
