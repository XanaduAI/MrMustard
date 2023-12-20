# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for wires.
"""

import pytest

from mrmustard.lab_dev.wires import Wires


class TestWires:
    r"""
    Tests for the ``Wires`` class.
    """

    @pytest.mark.parametrize("modes", [[0, 1], [9, 4, 10]])
    def test_init(self, modes):
        r"""
        Tests the init of ``Wires`` with unambiguous modes.
        """
        wires = Wires(modes, modes, modes)

        assert wires.modes == modes

        assert list(wires.out_bra.keys()) == modes
        assert None not in list(wires.out_bra.values())

        assert list(wires.in_bra.keys()) == modes
        assert None not in list(wires.in_bra.values())

        assert list(wires.out_ket.keys()) == modes
        assert None not in list(wires.out_ket.values())

        assert list(wires.in_ket.keys()) == modes
        assert list(wires.in_ket.values()) == [None] * len(modes)

    def test_init_ambiguous(self):
        r"""
        Tests the init of ``Wires`` with ambiguous modes.
        """
        wires = Wires([0, 2], [9, 3], [4, 5])

        with pytest.raises(ValueError, match="unambiguously"):
            wires.modes  # pylint: disable=pointless-statement

        assert list(wires.out_bra.keys()) == [0, 2, 3, 4, 5, 9]
        out_bra_values = list(wires.out_bra.values())
        assert out_bra_values[0] is not None
        assert out_bra_values[1] is not None
        assert out_bra_values[2] is None
        assert out_bra_values[3] is None
        assert out_bra_values[4] is None
        assert out_bra_values[5] is None

        assert list(wires.in_bra.keys()) == [0, 2, 3, 4, 5, 9]
        in_bra_values = list(wires.in_bra.values())
        assert in_bra_values[0] is None
        assert in_bra_values[1] is None
        assert in_bra_values[2] is not None
        assert in_bra_values[3] is None
        assert in_bra_values[4] is None
        assert in_bra_values[5] is not None

        assert list(wires.out_ket.keys()) == [0, 2, 3, 4, 5, 9]
        out_ket_values = list(wires.out_ket.values())
        assert out_ket_values[0] is None
        assert out_ket_values[1] is None
        assert out_ket_values[2] is None
        assert out_ket_values[3] is not None
        assert out_ket_values[4] is not None
        assert out_ket_values[5] is None

        assert list(wires.in_ket.keys()) == [0, 2, 3, 4, 5, 9]
        assert list(wires.in_ket.values()) == [None] * 6

    def test_modes_in_out(self):
        r"""
        Tests the methods to get modes.
        """
        wires = Wires(modes_in_ket=[2, 0])

        assert wires.modes_out_bra == []
        assert wires.modes_in_bra == []
        assert wires.modes_out_ket == []
        assert wires.modes_in_ket == [2, 0]

    def test_adjoint(self):
        r"""
        Tests the `adjoint` method.
        """
        wires = Wires([0, 2], [9, 3], [4, 5])
        wires_adj = wires.adjoint()

        assert wires.out_bra.keys() == wires_adj.out_ket.keys()
        assert wires.in_bra.keys() == wires_adj.in_ket.keys()
        assert wires.out_ket.keys() == wires_adj.out_bra.keys()
        assert wires.in_ket.keys() == wires_adj.in_bra.keys()

        assert set(wires.out_bra.values()) != set(wires_adj.out_ket.values())
        assert set(wires.in_bra.values()) != set(wires_adj.in_ket.values())
        assert set(wires.out_ket.values()) != set(wires_adj.out_bra.values())

    def test_new(self):
        r"""
        Tests the `new` method.
        """
        wires = Wires([0, 2], [9, 3], [4, 5])
        wires_new = wires.new()

        assert wires.out_bra.keys() == wires_new.out_bra.keys()
        assert wires.in_bra.keys() == wires_new.in_bra.keys()
        assert wires.out_ket.keys() == wires_new.out_ket.keys()
        assert wires.in_ket.keys() == wires_new.in_ket.keys()

        assert set(wires.out_bra.values()) != set(wires_new.out_bra.values())
        assert set(wires.in_bra.values()) != set(wires_new.in_bra.values())
        assert set(wires.out_ket.values()) != set(wires_new.out_ket.values())
        assert set(wires.in_ket.values()) == set(wires_new.in_ket.values())

    def test_get_item(self):
        r"""
        Tests the `__getitem__` method.
        """
        wires = Wires([0, 2], [9, 3], [4, 5])
        wires_slice = wires[2, 9]

        assert list(wires_slice.out_bra.keys()) == [2, 9]
        assert list(wires_slice.in_bra.keys()) == [2, 9]
        assert list(wires_slice.out_ket.keys()) == [2, 9]
        assert list(wires_slice.in_ket.keys()) == [2, 9]

        assert wires.out_bra[2] == wires_slice.out_bra[2]
        assert wires.out_bra[9] == wires_slice.out_bra[9]

        assert wires.in_bra[2] == wires_slice.in_bra[2]
        assert wires.in_bra[9] == wires_slice.in_bra[9]

        assert wires.out_ket[2] == wires_slice.out_ket[2]
        assert wires.out_ket[9] == wires_slice.out_ket[9]

        assert wires.in_ket[2] == wires_slice.in_ket[2]
        assert wires.in_ket[9] == wires_slice.in_ket[9]

    def test_list_of_types_and_modes_of_wires_for_single_wire_object_outside(self):
        r"""Tests that we can get the correct list of types and modes from wires for a single-wire object on the 'out' side."""
        wires = Wires(modes_out_ket=[5, 15])
        assert wires.modes is not None
        list_types, list_modes = wires.list_of_types_and_modes_of_wires()
        assert len(list_types) == len(list_modes)
        assert len(list_types) == 2
        assert list_types[0] == 'out_ket'
        assert list_modes[0] == 5
        assert list_types[1] == 'out_ket'
        assert list_modes[1] == 15

    def test_list_of_types_and_modes_of_wires_for_single_wire_object_inside(self):
        r"""Tests that we can get the correct list of types and modes from wires for a single-wire object on the 'in' side."""
        wires = Wires(modes_in_ket=[5, 15])
        assert wires.modes is not None
        list_types, list_modes = wires.list_of_types_and_modes_of_wires()
        assert len(list_types) == len(list_modes)
        assert len(list_types) == 2
        assert list_types[0] == 'in_ket'
        assert list_modes[0] == 5
        assert list_types[1] == 'in_ket'
        assert list_modes[1] == 15

    def test_list_of_types_and_modes_of_wires_for_2_wire_object_same_outside(self):
        r"""Tests that we can get the correct list of types and modes from wires for a 2-wire object on the same 'out' side."""
        wires = Wires(modes_out_bra=[0, 2], modes_out_ket=[0, 2])
        assert wires.modes is not None
        list_types, list_modes = wires.list_of_types_and_modes_of_wires()
        assert len(list_types) == len(list_modes)
        assert len(list_types) == 4
        assert list_types[0] == 'out_bra'
        assert list_modes[0] == 0
        assert list_types[1] == 'out_bra'
        assert list_modes[1] == 2
        assert list_types[2] == 'out_ket'
        assert list_modes[2] == 0
        assert list_types[3] == 'out_ket'
        assert list_modes[3] == 2

    def test_list_of_types_and_modes_of_wires_for_2_wire_object_oppo_side(self):
        r"""Tests that we can get the correct list of types and modes from wires for a 2-wire object on the opposite side."""
        wires = Wires(modes_out_ket=[1, 8, 10], modes_in_ket=[1, 8, 10])
        assert wires.modes is not None
        list_types, list_modes = wires.list_of_types_and_modes_of_wires()
        assert len(list_types) == len(list_modes)
        assert len(list_types) == 6
        assert list_types[0] == 'out_ket'
        assert list_modes[0] == 1
        assert list_types[1] == 'out_ket'
        assert list_modes[1] == 8
        assert list_types[2] == 'out_ket'
        assert list_modes[2] == 10
        assert list_types[3] == 'in_ket'
        assert list_modes[3] == 1
        assert list_types[4] == 'in_ket'
        assert list_modes[4] == 8
        assert list_types[5] == 'in_ket'
        assert list_modes[5] == 10

    def test_list_of_types_and_modes_of_wires_for_4_wire_object(self):
        r"""Tests that we can get the correct list of types and modes from wires for a 4-wire object."""
        wires = Wires(modes_out_bra=[0, 2], modes_out_ket=[0, 2], modes_in_bra=[0, 2], modes_in_ket=[0, 2])
        assert wires.modes is not None
        list_types, list_modes = wires.list_of_types_and_modes_of_wires()
        assert len(list_types) == len(list_modes)
        assert len(list_types) == 8
        assert list_types[0] == 'out_bra'
        assert list_modes[0] == 0
        assert list_types[2] == 'in_bra'
        assert list_modes[2] == 0
        assert list_types[4] == 'out_ket'
        assert list_modes[4] == 0
        assert list_types[6] == 'in_ket'
        assert list_modes[6] == 0

        assert list_types[1] == 'out_bra'
        assert list_modes[1] == 2
        assert list_types[3] == 'in_bra'
        assert list_modes[3] == 2
        assert list_types[5] == 'out_ket'
        assert list_modes[5] == 2
        assert list_types[7] == 'in_ket'
        assert list_modes[7] == 2
    
    def test_calculate_index_for_a_wire_on_given_mode_and_type(self):
        r"""Tests that the index of a given wire on a given mode is correctly calculated"""
        wires = Wires(modes_out_ket=[0, 4], modes_in_ket=[0, 4])
        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('out_ket', 0) == 0
        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('out_ket', 4) == 1
        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('in_ket', 0) == 2
        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('in_ket', 4) == 3

        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('in_bra', 4) is None
        assert wires.calculate_index_for_a_wire_on_given_mode_and_type('out_bra', 1007) is None
