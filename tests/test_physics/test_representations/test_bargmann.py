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

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple).conj()

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        temp1 = Bargmann(*triple1)
        print(temp1.A.shape)

        bargmann = Bargmann(*triple1) & Bargmann(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

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
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

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

    # def test_add_error(self):
    #     bargmann = Bargmann(*Abc_triple(3))
    #     fock = Fock(np.random.random((1, 4, 4, 4)), batched=True)

    #     with pytest.raises(ValueError):
    #         bargmann + fock  # pylint: disable=pointless-statement

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

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = Bargmann(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    def test_matmul_barg_barg(self):
        triple1 = Abc_triple(3)
        triple2 = Abc_triple(3)

        res1 = Bargmann(*triple1) @ Bargmann(*triple2)
        exp1 = contract_two_Abc(triple1, triple2, [], [])
        assert np.allclose(res1.A, exp1[0])
        assert np.allclose(res1.b, exp1[1])
        assert np.allclose(res1.c, exp1[2])

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
