# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Wires class."""

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

import pytest
from mrmustard.lab_dev.wires import Wires# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Wires class."""

# pylint: disable=protected-access

import numpy as np
import pytest

from mrmustard.lab_dev.wires import Wires

class TestWires:
    r"""
    Tests for the Wires class.
    """
    @pytest.mark.parametrize("modes_out_bra", [[0], [1, 2]])
    @pytest.mark.parametrize("modes_in_bra", [None, [1], [2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [2], [3, 4]])
    @pytest.mark.parametrize("modes_in_ket", [None, [3], [4, 5]])
    def test_init(self, modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket):
        w = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)

        modes_out_bra = modes_out_bra or []
        modes_in_bra = modes_in_bra or []
        modes_out_ket = modes_out_ket or []
        modes_in_ket = modes_in_ket or []
        modes = list(
            set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)
        )

        assert w.modes == modes
        assert w.output.bra.modes == modes_out_bra
        assert w.input.bra.modes == modes_in_bra
        assert w.output.ket.modes == modes_out_ket
        assert w.input.ket.modes == modes_in_ket

    def test_args(self):
        r"""
        Tests the ``_args`` method.
        """
        w = Wires([0], [1], [2], [3])
        assert w._args() == ((0,), (1,), (2,), (3,))

    def test_from_data(self):
        r"""
        Tests the ``_from_data`` method.
        """
        id_array = [[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 0], [0, 0, 0, 6]]
        modes = [5, 6, 7, 8]
        w = Wires._from_data(id_array, modes)

        assert np.allclose(w.id_array, id_array)
        assert w.ids == [1, 2, 3, 4, 5, 6]
        assert w.modes == modes
        
    def test_view(self):
        r"""
        Tests the ``_view`` method.
        """
        w = Wires([0], [0], [0], [0])
        assert set(w.ids) == set(w._view().ids)

    def test_view_can_edit_original(self):
        r"""
        Tests that modifications done to the ``_view`` reflect on the original.
        """
        w = Wires([0], [0], [0], [0])
        w._view().ids = [9, 99, 999, 9999]
        assert w.ids == [9, 99, 999, 9999]

    def test_wire_subsets(self):
        r"""
        Tests the methods to obtain subsets.
        """
        w = Wires([0], [1], [2], [3])
        assert w.output.bra.modes == [0]
        assert w.input.bra.modes == [1]
        assert w.output.ket.modes == [2]
        assert w.input.ket.modes == [3]

        w = Wires([10], [11], [12], [13])
        assert w[10].ids == w.output.bra.ids
        assert w[11].ids == w.input.bra.ids
        assert w[12].ids == w.output.ket.ids
        assert w[13].ids == w.input.ket.ids

    def test_id_array(self):
        r"""
        Tests the ``id_array`` property.
        """
        w = Wires([0, 1], [2], [3, 4, 5], [6])
        assert w.id_array.shape == (7, 4)

    def test_ids(self):
        r"""
        Tests the ``ids`` property and the standard order.
        """
        w = Wires([0, 1], [2], [3, 4, 5], [6])
        
        assert w.output.bra.ids == w.ids[:2]
        assert w.input.bra.ids == [w.ids[2]]
        assert w.output.ket.ids == w.ids[3:6]
        assert w.input.ket.ids == [w.ids[-1]]

    def test_ids_setter(self):
        r"""
        Tests the setter for ``ids``.
        """
        w1 = Wires([0, 1], [2], [3, 4, 5], [6])
        w2 = Wires([0, 1], [2], [3, 4, 5], [6])

        assert w1.ids != w2.ids

        w1.ids = w2.ids
        assert w1.ids == w2.ids

    def test_indices(self):
        r"""
        Tests the ``indices`` property and the standard order.
        """
        w = Wires([0, 1, 2], [3, 4, 5], [6, 7], [8])

        assert w.output.indices == [0, 1, 2, 6, 7]
        assert w.bra.indices == [0, 1, 2, 3, 4, 5]
        assert w.input.indices == [3, 4, 5, 8]
        assert w.ket.indices == [6, 7, 8]

    def test_adjoint(self):
        r"""
        Tests the ``adjoint`` method.
        """
        w = Wires([0, 1, 2], [3, 4, 5], [6, 7], [8])
        w_adj = w.adjoint

        assert w.input.ket.modes == w_adj.input.bra.modes
        assert w.output.ket.modes == w_adj.output.bra.modes
        assert w.input.bra.modes == w_adj.input.ket.modes
        assert w.output.bra.modes == w_adj.output.ket.modes

    def test_dual(self):
        r"""
        Tests the ``dual`` method.
        """
        w = Wires([0, 1, 2], [3, 4, 5], [6, 7], [8])
        w_d = w.dual

        assert w.input.ket.modes == w_d.output.ket.modes
        assert w.output.ket.modes == w_d.input.ket.modes
        assert w.input.bra.modes == w_d.output.bra.modes
        assert w.output.bra.modes == w_d.input.bra.modes

    def test_copy(self):
        r"""
        Tests the ``copy`` method.
        """
        w = Wires([0, 1, 2], [3, 4, 5], [6, 7], [8])
        w_cp = w.copy()

        assert w.input.ket.modes == w_cp.input.ket.modes
        assert w.output.ket.modes == w_cp.output.ket.modes
        assert w.input.bra.modes == w_cp.input.bra.modes
        assert w.output.bra.modes == w_cp.output.bra.modes

    def test_add(self):
        r"""
        Tests the ``__add__`` method.
        """
        w1 = Wires([0], [1], [2], [3])
        w2 = Wires([1], [2], [3], [4])
        w12 = Wires([0, 1], [1, 2], [2, 3], [3, 4])

        assert (w1 + w2).modes == w12.modes

    def test_add_error(self):
        r"""
        Tests the error raised by ``__add__`` method.
        """
        w1 = Wires([0], [1], [2], [3])
        w2 = Wires([0], [2], [3], [4])
        with pytest.raises(Exception):
            w1 + w2

    def test_bool(self):
        r"""
        Tests the ``__bool__`` method.
        """
        assert Wires([0])
        assert not Wires([0]).input

    def test_getitem(self):
        r"""
        Tests the ``__getitem__`` method.
        """
        w = Wires([0, 1], [0, 1])
        w0 = w[0]
        w1 = w[1]

        assert w0.modes == [0]
        assert w0.ids == [w.ids[0], w.ids[2]]

        assert w1.modes == [1]
        assert w1.ids == [w.ids[1], w.ids[3]]


    def test_rshift(self):
        r"""
        Tests the ``__rshift__`` method.
        """
        # contracts 1,1 on bra side
        # contracts 3,3 and 13,13 on ket side (note order doesn't matter)
        u = Wires([1, 5], [2, 6, 15], [3, 7, 13], [4, 8])
        v = Wires([0, 9, 14], [1, 10], [2, 11], [13, 3, 12])
        assert (u >> v)._args() == ((0, 5, 9, 14), (2, 6, 10, 15), (2, 7, 11), (4, 8, 12))


    def test_rshift_error(self):
        r"""
        Tests the error of the ``__rshift__`` method.
        """
        u = Wires([], [], [0], [])  # only output wire
        v = Wires([], [], [0], [])  # only output wire
        with pytest.raises(ValueError):
            u >> v
