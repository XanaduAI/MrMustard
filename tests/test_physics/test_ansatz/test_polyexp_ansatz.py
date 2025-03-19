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

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_init_non_batched(self, n):
        A, b, c = Abc_triple(n)
        bargmann = PolyExpAnsatz(A, b, c)

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

        assert np.allclose(bargmann_add.A, math.stack([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.stack([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.stack([bargmann1.c, bargmann2.c], axis=0))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.random(size=(3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.random(size=(2, 2))

        bargmann3 = PolyExpAnsatz(A1, b1, c1)
        bargmann4 = PolyExpAnsatz(A2, b2, c2)

        bargmann_add2 = bargmann3 + bargmann4

        assert np.allclose(bargmann_add2.A[0], A1)
        assert np.allclose(bargmann_add2.b[0], b1)
        assert np.allclose(bargmann_add2.c[0], c1)
        assert np.allclose(bargmann_add2.A[1], A2)
        assert np.allclose(bargmann_add2.b[1], b2)
        assert np.allclose(bargmann_add2.c[1][:2, :2], c2)

    def test_add_different_poly_wires(self):
        "tests that A and b are padded correctly"
        A1 = np.random.random((1, 2, 2))
        A2 = np.random.random((1, 3, 3))
        b1 = np.random.random((1, 2))
        b2 = np.random.random((1, 3))
        c1 = np.random.random((1,))
        c2 = np.random.random((1, 11))
        ansatz1 = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz_sum = ansatz1 + ansatz2
        assert ansatz_sum.A.shape == (2, 3, 3)
        assert ansatz_sum.b.shape == (2, 3)
        assert ansatz_sum.c.shape == (2, 11)

    def test_add_error(self):
        ans1 = PolyExpAnsatz(*Abc_triple(3))
        ans2 = PolyExpAnsatz(*Abc_triple(4))

        with pytest.raises(ValueError):
            ans1 + ans2  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = PolyExpAnsatz(*triple1) & PolyExpAnsatz(*triple2)

        assert bargmann.A.shape == (2 * n, 2 * n)
        assert bargmann.b.shape == (2 * n,)
        assert bargmann.c.shape == ()

    def test_and_batched(self):
        triple1 = Abc_triple(3, (2,))
        triple2 = Abc_triple(3, (1,))
        bargmann1 = PolyExpAnsatz(*triple1)
        bargmann2 = PolyExpAnsatz(*triple2)
        bargmann = bargmann1 & bargmann2
        assert bargmann.A.shape == (2, 1, 6, 6)
        assert bargmann.b.shape == (2, 1, 6)
        assert bargmann.c.shape == (2, 1)

    def test_call(self):
        A, b, c = Abc_triple(5)
        ansatz = PolyExpAnsatz(A, b, c)

        assert np.allclose(ansatz(*math.zeros_like(b)), c)

        A, b, _ = Abc_triple(4)
        c = np.random.random(size=(1, 3, 3, 3)) + 0.0j
        ansatz = PolyExpAnsatz(A, b, c)
        z = np.random.uniform(-10, 10, size=(7, 2))
        with pytest.raises(Exception):
            ansatz(*z)

        A = math.astensor([np.array([[0.0, 1.0], [1.0, 0.0]]) + 0.0j])
        b = math.astensor([np.zeros(2) + 0.0j])
        c = np.zeros(10, dtype=complex).reshape(1, -1) + 0.0j
        c[..., -1] = 1
        ans = PolyExpAnsatz(A, b, c)

        nine_factorial = np.prod(np.arange(1, 9))
        assert np.allclose(ans(np.array([0.1 + 0.0j])), 0.1**9 / np.sqrt(nine_factorial))

    def test_partial_eval(self):
        batch = 3
        A, b, _ = Abc_triple(4, (batch,))

        c = np.random.random(size=(batch, 5, 5)) / 1000 + 0.0j

        obj = PolyExpAnsatz(A, b, c)
        z0 = [None, 2.0 + 0.0j]
        z1 = [1.0 + 0.0j]
        z2 = [1.0 + 0.0j, 2.0 + 0.0j]
        val_full = obj(*z2)
        partial = obj(*z0)
        val_partial = partial(*z1)
        assert math.allclose(val_partial, val_full)

        batch = 2
        A, b, _ = Abc_triple(4, (batch,))
        c = np.random.random(size=(2, 5)) / 1000 + 0.0j

        obj = PolyExpAnsatz(A, b, c)
        z0 = [None, 2.0 + 0.0j, None]
        z1 = [1.0 + 0.0j, 3.0 + 0.0j]
        z2 = [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]
        val_full = obj(*z2)
        partial = obj(*z0)
        val_partial = partial(*z1)
        assert math.allclose(val_partial, val_full)

    @pytest.mark.parametrize("triple", [1, 2, 3])
    def test_conj(self, triple):
        A, b, c = Abc_triple(triple)
        bargmann = PolyExpAnsatz(A, b, c).conj

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
        A, b, _ = Abc_triple(4, (1,))
        c = np.random.uniform(-1, 1, size=(1, 3, 3, 3)) + 0.0j
        ansatz = PolyExpAnsatz(A, b, c)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-1, 1, size=(1,)) + 0.0j
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (1, 2, 2))

        c2 = np.random.uniform(-1, 1, size=(1, 4)) + 0.0j
        ansatz2 = PolyExpAnsatz(A, b, c2)
        decomp_ansatz2 = ansatz2.decompose_ansatz()
        assert np.allclose(decomp_ansatz2.A, ansatz2.A)

    def test_decompose_ansatz_batch(self):
        """
        In this test the batch dimension of both ``z`` and ``Abc`` is tested.
        """
        A, b, _ = Abc_triple(4, (2,))
        c = np.random.random((2, 3, 3, 3))
        ansatz = PolyExpAnsatz(A, b, c)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.random((1,))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 2, 2))
        assert np.allclose(decomp_ansatz.b.shape, (2, 2))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9))

        A, b, _ = Abc_triple(5, (2,))
        c = np.random.random((2, 3, 3, 3))
        ansatz = PolyExpAnsatz(A, b, c)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.random((4,))
        assert np.allclose(ansatz(z, z), decomp_ansatz(z, z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 4, 4))
        assert np.allclose(decomp_ansatz.b.shape, (2, 4))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9, 9))

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [1, 2, 3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = PolyExpAnsatz(*Abc_triple(triple))
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

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [1, 2, 3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = PolyExpAnsatz(*Abc_triple(triple))
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
        A, b, _ = Abc_triple(4, (1,))
        c = np.random.random((1, 3))
        ansatz = PolyExpAnsatz(A, b, c)

        poly_dim = ansatz.num_derived_vars
        poly_shape = ansatz.shape_derived_vars
        assert np.allclose(poly_dim, 1)
        assert np.allclose(poly_shape, (3,))

    def test_reorder(self):
        triple = Abc_triple(3, (2,))
        bargmann = PolyExpAnsatz(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A, triple[0][:, [0, 2, 1], :][:, :, [0, 2, 1]])
        assert np.allclose(bargmann.b, triple[1][:, [0, 2, 1]])

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

        assert np.allclose(bargmann_add.A, math.stack([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.stack([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.stack([bargmann1.c, -bargmann2.c], axis=0))

    def test_trace(self):
        triple = Abc_triple(4)
        bargmann = PolyExpAnsatz(*triple).trace([0], [2])
        A, b, c = complex_gaussian_integral_1(triple, [0], [2])

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    def test_eval_with_scalar_inputs(self):
        """Test evaluation with scalar inputs."""
        A = np.random.random((3, 3))
        b = np.random.random(3)
        c = np.random.random(())
        F = PolyExpAnsatz(A, b, c, name="F")

        # Scalar inputs
        z0, z1, z2 = 0.4, 0.5, 0.2

        # Test eval method
        val = F.eval(z0, z1, z2)
        assert val.shape == ()

        # Test __call__ method
        val_call = F(z0, z1, z2)
        assert val_call.shape == ()

        # Verify both methods give the same result
        assert np.allclose(val, val_call)

    def test_eval_with_batched_inputs(self):
        """Test evaluation with batched inputs."""
        A = np.random.random((3, 3))
        b = np.random.random(3)
        c = np.random.random(())
        F = PolyExpAnsatz(A, b, c, name="F")

        # Batched inputs with different shapes
        z0 = np.array([0.4, 0.2])
        z1 = np.array(0.5)
        z2 = np.array([[0.3, 0.3]])

        # Test eval method
        val = F.eval(z0, z1, z2)
        assert val.shape == (2, 1, 2)

        # Test with custom batch string
        val_custom = F.eval(z0, z1, z2, batch_string="a,,ba->ab")
        assert val_custom.shape == (2, 1)

    def test_batched_ansatz_with_scalar_inputs(self):
        """Test batched ansatz with scalar inputs."""
        # Create a batched ansatz
        A = np.random.random((4, 7, 3, 3))
        b = np.random.random((4, 7, 3))
        c = np.random.random((4, 7))
        F = PolyExpAnsatz(A, b, c, name="batched")

        # Scalar inputs
        z0, z1, z2 = 0.4, 0.5, 0.2

        # Test eval method
        val = F.eval(z0, z1, z2)
        assert val.shape == (4, 7)

        # Test __call__ method
        val_call = F(z0, z1, z2)
        assert val_call.shape == (4, 7)

        # Verify both methods give the same result
        assert np.allclose(val, val_call)

    def test_batched_ansatz_with_batched_inputs(self):
        """Test batched ansatz with batched inputs."""
        # Create a batched ansatz
        A = np.random.random((4, 7, 3, 3))
        b = np.random.random((4, 7, 3))
        c = np.random.random((4, 7))
        F = PolyExpAnsatz(A, b, c, name="batched")

        # Batched inputs
        z1 = np.array([0.4, 0.2])
        z2 = np.array(0.5)
        z3 = np.array([[0.3, 0.3]])

        # Test eval method
        val = F.eval(z1, z2, z3)
        assert val.shape == (2, 1, 2, 4, 7)

        # Test with custom batch string
        val_custom = F.eval(z1, z2, z3, batch_string="a,,ba->ab")
        assert val_custom.shape == (2, 1, 4, 7)

    def test_derived_variables_with_scalar_inputs(self):
        """Test ansatz with derived variables using scalar inputs."""
        # Create ansatz with derived variables
        A = np.random.random((4, 7, 3, 3))
        b = np.random.random((4, 7, 3))
        c = np.random.random((4, 7, 5))
        F = PolyExpAnsatz(A, b, c, name="derived+batched")

        # Scalar inputs (only 2 inputs needed for 2 CV + 1 derived = 3 total variables)
        z0, z1 = 0.4, 0.5

        # Test eval method
        val = F.eval(z0, z1)
        assert val.shape == (4, 7)

        # Test __call__ method
        val_call = F(z0, z1)
        assert val_call.shape == (4, 7)

        # Verify both methods give the same result
        assert np.allclose(val, val_call)

    def test_derived_variables_with_batched_inputs(self):
        """Test ansatz with derived variables using batched inputs."""
        # Create ansatz with derived variables
        A = np.random.random((4, 7, 3, 3))
        b = np.random.random((4, 7, 3))
        c = np.random.random((4, 7, 5))
        F = PolyExpAnsatz(A, b, c, name="derived+batched")

        # Batched inputs
        z0 = np.array([0.5, 0.9])
        z1 = np.array([[0.3, 0.3]])

        # Test eval method
        val = F.eval(z0, z1)
        assert val.shape == (2, 1, 2, 4, 7)

        # Test with custom batch string
        val_custom = F.eval(z0, z1, batch_string="a,ba->ab")
        assert val_custom.shape == (2, 1, 4, 7)

    def test_singleton_dimensions(self):
        """Test ansatz with singleton dimensions in inputs."""
        # Create ansatz with derived variables
        A = np.random.random((4, 7, 3, 3))
        b = np.random.random((4, 7, 3))
        c = np.random.random((4, 7, 5))
        F = PolyExpAnsatz(A, b, c, name="derived+batched")

        # Inputs with singleton dimensions
        z0 = np.array([0.4, 0.2])[:, None]  # Shape (2, 1)
        z1 = np.array([0.5, 0.2, 0.2])[None, :]  # Shape (1, 3)

        # Test __call__ method
        val = F(z0, z1)
        assert val.shape == (2, 3, 4, 7)

    def test_partial_evaluation_with_scalar_input(self):
        """Test partial evaluation with scalar inputs."""
        # Create ansatz with derived variables
        A, b, c = Abc_triple(3, (4, 7))
        c = np.random.random((4, 7, 5))
        F = PolyExpAnsatz(A, b, c, name="derived+batched")

        # Partial evaluation with scalar input
        z0 = np.array(0.5 + 0.0j)
        partial_F = F(z0, None)

        # Verify partial_F is still a PolyExpAnsatz
        assert isinstance(partial_F, PolyExpAnsatz)

        # Complete the evaluation
        z1 = 0.3 + 0.0j
        result = partial_F(z1)

        # Verify the result matches direct evaluation
        direct_result = F(z0, z1)
        assert np.allclose(result, direct_result)

    def test_partial_evaluation_with_batched_input(self):
        """Test partial evaluation with batched inputs."""
        # Create ansatz with derived variables
        A, b, c = Abc_triple(3, (4, 7))
        c = np.random.random((4, 7, 5))
        F = PolyExpAnsatz(A, b, c, name="derived+batched")

        # Partial evaluation with scalar input
        z0 = np.array([0.5, 0.6]) + 0.0j
        partial_F = F(z0, None)

        # Verify partial_F is still a PolyExpAnsatz
        assert isinstance(partial_F, PolyExpAnsatz)

        # Complete the evaluation
        z1 = 0.3 + 0.0j
        result = partial_F(z1)

        # Verify the result matches direct evaluation
        direct_result = F(z0, z1)
        assert np.allclose(result, direct_result)
