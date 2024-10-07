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

"""This module contains tests for ``Representation`` objects."""

# pylint: disable = too-many-public-methods

from unittest.mock import patch

import numpy as np
from ipywidgets import Box, HBox, VBox, HTML, IntText, Stack, IntSlider
from plotly.graph_objs import FigureWidget
import pytest

from mrmustard import math, settings
from mrmustard.physics.gaussian_integrals import (
    contract_two_Abc,
    complex_gaussian_integral,
)
from mrmustard.physics.representations.bargmann import Bargmann
from mrmustard.physics.representations.fock import Fock

from ...random import Abc_triple

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF

# pylint: disable = missing-function-docstring


class TestBargmannRepresentation:
    r"""
    Tests the Bargmann Representation.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_non_batched(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple)

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_add(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 + bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.random(size=(1, 3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.random(size=(1, 2, 2))

        bargmann3 = Bargmann(A1, b1, c1)
        bargmann4 = Bargmann(A2, b2, c2)

        bargmann_add2 = bargmann3 + bargmann4

        assert np.allclose(bargmann_add2.A[0], A1)
        assert np.allclose(bargmann_add2.b[0], b1)
        assert np.allclose(bargmann_add2.c[0], c1[0])
        assert np.allclose(bargmann_add2.A[1], A2)
        assert np.allclose(bargmann_add2.b[1], b2)
        assert np.allclose(bargmann_add2.c[1][:2, :2], c2[0])

    def test_add_error(self):
        bargmann = Bargmann(*Abc_triple(3))
        fock = Fock(np.random.random((1, 4, 4, 4)), batched=True)

        with pytest.raises(TypeError, match="Cannot add"):
            bargmann + fock  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = Bargmann(*triple1) & Bargmann(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    def test_call(self):
        A, b, c = Abc_triple(5)
        ansatz = Bargmann(A, b, c)

        assert np.allclose(ansatz(z=math.zeros_like(b)), c)

        A, b, _ = Abc_triple(4)
        c = np.random.random(size=(1, 3, 3, 3))
        ansatz = Bargmann(A, b, c)
        z = np.random.uniform(-10, 10, size=(7, 2))
        with pytest.raises(
            Exception, match="The sum of the dimension of the argument and polynomial"
        ):
            ansatz(z)

        A = np.array([[0, 1], [1, 0]])
        b = np.zeros(2)
        c = c = np.zeros(10, dtype=complex).reshape(1, -1)
        c[0, -1] = 1
        obj1 = Bargmann(A, b, c)

        nine_factorial = np.prod(np.arange(1, 9))
        assert np.allclose(obj1(np.array([[0.1]])), 0.1**9 / np.sqrt(nine_factorial))

    def test_call_none(self):
        A1, b1, _ = Abc_triple(7)
        A2, b2, _ = Abc_triple(7)
        A3, b3, _ = Abc_triple(7)

        batch = 3
        c = np.random.random(size=(batch, 5, 5, 5)) / 1000

        obj = Bargmann([A1, A2, A3], [b1, b2, b3], c)
        z0 = np.array([[None, 2, None, 5]])
        z1 = np.array([[1, 2, 4, 5]])
        z2 = np.array([[1, 4]])
        obj_none = obj(z0)
        val1 = obj(z1)
        val2 = obj_none(z2)
        assert np.allclose(val1, val2)

        obj1 = Bargmann(A1, b1, c[0].reshape(1, 5, 5, 5))
        z0 = np.array([[None, 2, None, 5], [None, 1, None, 4]])
        z1 = np.array([[1, 2, 4, 5], [2, 1, 4, 4]])
        z2 = np.array([[1, 4], [2, 4]])
        obj1_none = obj1(z0)
        obj1_none0 = Bargmann(obj1_none.A[0], obj1_none.b[0], obj1_none.c[0].reshape(1, 5, 5, 5))
        obj1_none1 = Bargmann(obj1_none.A[1], obj1_none.b[1], obj1_none.c[1].reshape(1, 5, 5, 5))
        val1 = obj1(z1)
        val2 = np.array(
            (obj1_none0(z2[0].reshape(1, -1)), obj1_none1(z2[1].reshape(1, -1)))
        ).reshape(-1)
        assert np.allclose(val1, val2)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple).conj

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    def test_decompose_ansatz(self):
        A, b, _ = Abc_triple(4)
        c = np.random.uniform(-10, 10, size=(1, 3, 3, 3))
        ansatz = Bargmann(A, b, c)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(1, 1))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (1, 2, 2))

        c2 = np.random.uniform(-10, 10, size=(1, 4))
        ansatz2 = Bargmann(A, b, c2)
        decomp_ansatz2 = ansatz2.decompose_ansatz()
        assert np.allclose(decomp_ansatz2.A, ansatz2.A)

    def test_decompose_ansatz_batch(self):
        """
        In this test the batch dimension of both ``z`` and ``Abc`` is tested.
        """
        A1, b1, _ = Abc_triple(4)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3))
        A2, b2, _ = Abc_triple(4)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3))
        ansatz = Bargmann([A1, A2], [b1, b2], [c1, c2])

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(3, 1))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 2, 2))
        assert np.allclose(decomp_ansatz.b.shape, (2, 2))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3))
        ansatz = Bargmann([A1, A2], [b1, b2], [c1, c2])

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(3, 2))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 4, 4))
        assert np.allclose(decomp_ansatz.b.shape, (2, 4))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9, 9))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_div(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_div = bargmann1 / bargmann2

        assert np.allclose(bargmann_div.A, bargmann1.A - bargmann2.A)
        assert np.allclose(bargmann_div.b, bargmann1.b - bargmann2.b)
        assert np.allclose(bargmann_div.c, bargmann1.c / bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

    def test_eq(self):
        A, b, c = Abc_triple(5)

        ansatz = Bargmann(A, b, c)
        ansatz2 = Bargmann(2 * A, 2 * b, 2 * c)

        assert ansatz == ansatz  # pylint: disable= comparison-with-itself
        assert ansatz2 == ansatz2  # pylint: disable= comparison-with-itself
        assert ansatz != ansatz2
        assert ansatz2 != ansatz

    def test_matmul_barg_barg(self):
        triple1 = Abc_triple(3)
        triple2 = Abc_triple(3)

        res1 = Bargmann(*triple1) @ Bargmann(*triple2)
        exp1 = contract_two_Abc(triple1, triple2, [], [])
        assert np.allclose(res1.A, exp1[0])
        assert np.allclose(res1.b, exp1[1])
        assert np.allclose(res1.c, exp1[2])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mul(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_mul = bargmann1 * bargmann2

        assert np.allclose(bargmann_mul.A, bargmann1.A + bargmann2.A)
        assert np.allclose(bargmann_mul.b, bargmann1.b + bargmann2.b)
        assert np.allclose(bargmann_mul.c, bargmann1.c * bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

    def test_order_batch(self):
        ansatz = Bargmann(
            A=[np.array([[0]]), np.array([[1]])],
            b=[np.array([1]), np.array([0])],
            c=[1, 2],
        )
        ansatz._order_batch()  # pylint: disable=protected-access

        assert np.allclose(ansatz.A[0], np.array([[1]]))
        assert np.allclose(ansatz.b[0], np.array([0]))
        assert ansatz.c[0] == 2
        assert np.allclose(ansatz.A[1], np.array([[0]]))
        assert np.allclose(ansatz.b[1], np.array([1]))
        assert ansatz.c[1] == 1

    def test_polynomial_shape(self):
        A, b, _ = Abc_triple(4)
        c = np.array([[1, 2, 3]])
        ansatz = Bargmann(A, b, c)

        poly_dim, poly_shape = ansatz.polynomial_shape
        assert np.allclose(poly_dim, 1)
        assert np.allclose(poly_shape, (3,))

        A1, b1, _ = Abc_triple(4)
        c1 = np.array([[1, 2, 3]])
        ansatz1 = Bargmann(A1, b1, c1)

        A2, b2, _ = Abc_triple(4)
        c2 = np.array([[1, 2, 3]])
        ansatz2 = Bargmann(A2, b2, c2)

        ansatz3 = ansatz1 * ansatz2

        poly_dim, poly_shape = ansatz3.polynomial_shape
        assert np.allclose(poly_dim, 2)
        assert np.allclose(poly_shape, (3, 3))

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = Bargmann(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    def test_simplify(self):
        A, b, c = Abc_triple(5)

        ansatz = Bargmann(A, b, c)

        ansatz = ansatz + ansatz

        assert np.allclose(ansatz.A[0], ansatz.A[1])
        assert np.allclose(ansatz.A[0], A)
        assert np.allclose(ansatz.b[0], ansatz.b[1])
        assert np.allclose(ansatz.b[0], b)

        ansatz.simplify()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert ansatz.c == 2 * c

    def test_simplify_v2(self):
        A, b, c = Abc_triple(5)

        ansatz = Bargmann(A, b, c)

        ansatz = ansatz + ansatz

        assert math.allclose(ansatz.A[0], ansatz.A[1])
        assert math.allclose(ansatz.A[0], A)
        assert math.allclose(ansatz.b[0], ansatz.b[1])
        assert math.allclose(ansatz.b[0], b)

        ansatz.simplify_v2()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert math.allclose(ansatz.c, 2 * c)

        A, b, c = ansatz.triple

        ansatz.simplify_v2()
        assert math.allclose(ansatz.A, A)
        assert math.allclose(ansatz.b, b)
        assert math.allclose(ansatz.c, c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0))

    def test_trace(self):
        triple = Abc_triple(4)
        bargmann = Bargmann(*triple).trace([0], [2])
        A, b, c = complex_gaussian_integral(triple, [0], [2])

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    # @patch("mrmustard.physics.representations.bargmann.display")
    # def test_ipython_repr(self, mock_display):
    #     """Test the IPython repr function."""
    #     rep = Bargmann(*Abc_triple(2))
    #     rep._ipython_display_()  # pylint:disable=protected-access
    #     [box] = mock_display.call_args.args
    #     assert isinstance(box, Box)
    #     assert box.layout.max_width == "50%"

    #     # data on left, eigvals on right
    #     [data_vbox, eigs_vbox] = box.children
    #     assert isinstance(data_vbox, VBox)
    #     assert isinstance(eigs_vbox, VBox)

    #     # data forms a stack: header, ansatz, triple data
    #     [header, sub, table] = data_vbox.children
    #     assert isinstance(header, HTML)
    #     assert isinstance(sub, HBox)
    #     assert isinstance(table, HTML)

    #     # ansatz goes beside button to modify rounding
    #     [ansatz, round_w] = sub.children
    #     assert isinstance(ansatz, HTML)
    #     assert isinstance(round_w, IntText)

    #     # eigvals have a header and a unit circle plot
    #     [eig_header, unit_circle] = eigs_vbox.children
    #     assert isinstance(eig_header, HTML)
    #     assert isinstance(unit_circle, FigureWidget)

    # @patch("mrmustard.physics.representations.bargmann.display")
    # def test_ipython_repr_batched(self, mock_display):
    #     """Test the IPython repr function for a batched repr."""
    #     A1, b1, c1 = Abc_triple(2)
    #     A2, b2, c2 = Abc_triple(2)
    #     rep = Bargmann(np.array([A1, A2]), np.array([b1, b2]), np.array([c1, c2]))
    #     rep._ipython_display_()  # pylint:disable=protected-access
    #     [vbox] = mock_display.call_args.args
    #     assert isinstance(vbox, VBox)

    #     [slider, stack] = vbox.children
    #     assert isinstance(slider, IntSlider)
    #     assert slider.max == 1  # the batch size - 1
    #     assert isinstance(stack, Stack)

    #     # max_width is spot-check that this is bargmann widget
    #     assert len(stack.children) == 2
    #     assert all(box.layout.max_width == "50%" for box in stack.children)
