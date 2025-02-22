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

"""This module contains tests for ``ExpAnsatz`` objects."""

# pylint: disable = too-many-public-methods, missing-function-docstring

from unittest.mock import patch

import numpy as np
import pytest
from ipywidgets import HTML, Box, IntSlider, IntText, Stack, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz
from mrmustard.physics.ansatz.exp_ansatz import ExpAnsatz
from mrmustard.physics.gaussian_integrals import (
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
)

from ...random import Abc_triple


class TestExpAnsatz:
    r"""
    Tests the exp ansatz.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_non_batched(self, triple):
        A, b, c = triple
        bargmann = ExpAnsatz(*triple)

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_add(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = ExpAnsatz(*triple1)
        bargmann2 = ExpAnsatz(*triple2)
        bargmann_add = bargmann1 + bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0))

    def test_add_error(self):
        bargmann = ExpAnsatz(*Abc_triple(3))
        fock = ArrayAnsatz(np.random.random((1, 4, 4, 4)), batched=True)

        with pytest.raises(TypeError, match="Cannot add"):
            bargmann + fock  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = ExpAnsatz(*triple1) & ExpAnsatz(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    def test_eval(self):
        A, b, c = Abc_triple(5)
        ansatz = ExpAnsatz(A, b, c)

        assert np.allclose(ansatz(*math.zeros_like(b)), c)

        expected_1 = c * np.exp(0.5 * np.sum(A) + np.sum(b))
        assert np.allclose(ansatz(*np.ones((5,))), expected_1)

        expected_call = ansatz([1.0], [2.0], [3.0], [4.0], [5.0])
        expected_eval = ansatz._eval(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert np.allclose(expected_call, expected_eval)

    def test_partial_eval(self):
        A1, b1, _ = Abc_triple(5)
        A2, b2, _ = Abc_triple(5)
        A3, b3, _ = Abc_triple(5)

        batch = 3
        c = np.random.random(size=(batch,)) / 1000

        ansatz = ExpAnsatz([A1, A2, A3], [b1, b2, b3], c)

        ansatz_partial_call = ansatz([1.0], None, [2.0], None, None)
        ansatz_partial_eval = ansatz._partial_eval([1.0, 2.0], (0, 2))

        assert ansatz_partial_call == ansatz_partial_eval

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = ExpAnsatz(*triple).conj

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    def test_contract_barg_barg(self):
        triple1 = Abc_triple(3)
        triple2 = Abc_triple(3)

        res1 = ExpAnsatz(*triple1).contract(ExpAnsatz(*triple2))
        exp1 = complex_gaussian_integral_2(triple1, triple2, [], [])
        assert np.allclose(res1.A, exp1[0])
        assert np.allclose(res1.b, exp1[1])
        assert np.allclose(res1.c, exp1[2])
        assert res1 == ExpAnsatz(*triple1) & ExpAnsatz(*triple2)  # via tensor product

        res2 = ExpAnsatz(*triple1).contract(ExpAnsatz(*triple2), idx1=0, idx2=0)  # via contract
        exp2 = complex_gaussian_integral_2(triple1, triple2, [0], [0])
        assert np.allclose(res2.A, exp2[0])
        assert np.allclose(res2.b, exp2[1])
        assert np.allclose(res2.c, exp2[2])

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = ExpAnsatz(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

    def test_eq(self):
        A, b, c = Abc_triple(5)

        ansatz = ExpAnsatz(A, b, c)
        ansatz2 = ExpAnsatz(2 * A, 2 * b, 2 * c)

        assert ansatz == ansatz  # pylint: disable= comparison-with-itself
        assert ansatz2 == ansatz2  # pylint: disable= comparison-with-itself
        assert ansatz != ansatz2
        assert ansatz2 != ansatz

    @patch("mrmustard.physics.ansatz.exp_ansatz.display")
    def test_ipython_repr(self, mock_display):
        """Test the IPython repr function."""
        rep = ExpAnsatz(*Abc_triple(2))
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

    @patch("mrmustard.physics.ansatz.exp_ansatz.display")
    def test_ipython_repr_batched(self, mock_display):
        """Test the IPython repr function for a batched repr."""
        A1, b1, c1 = Abc_triple(2)
        A2, b2, c2 = Abc_triple(2)
        rep = ExpAnsatz(np.array([A1, A2]), np.array([b1, b2]), np.array([c1, c2]))
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
        rep = ExpAnsatz(*Abc_triple(2))
        rep._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(rep)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mul(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = ExpAnsatz(*triple1)
        bargmann2 = ExpAnsatz(*triple2)
        bargmann_mul = bargmann1 * bargmann2

        assert np.allclose(bargmann_mul.A, bargmann1.A + bargmann2.A)
        assert np.allclose(bargmann_mul.b, bargmann1.b + bargmann2.b)
        assert np.allclose(bargmann_mul.c, bargmann1.c * bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = ExpAnsatz(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

    def test_order_batch(self):
        ansatz = ExpAnsatz(
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

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = ExpAnsatz(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    def test_simplify(self):
        A, b, c = Abc_triple(5)

        ansatz = ExpAnsatz(A, b, c)

        ansatz = ansatz + ansatz

        assert np.allclose(ansatz.A[0], ansatz.A[1])
        assert np.allclose(ansatz.A[0], A)
        assert np.allclose(ansatz.b[0], ansatz.b[1])
        assert np.allclose(ansatz.b[0], b)

        ansatz.simplify()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert np.allclose(ansatz.c, 2 * c)

        A, b, c = ansatz.triple

        ansatz.simplify()
        assert np.allclose(ansatz.A, A)
        assert np.allclose(ansatz.b, b)
        assert np.allclose(ansatz.c, c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = ExpAnsatz(*triple1)
        bargmann2 = ExpAnsatz(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0))

    def test_trace(self):
        triple = Abc_triple(4)
        bargmann = ExpAnsatz(*triple).trace([0], [2])
        A, b, c = complex_gaussian_integral_1(triple, [0], [2])

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)
