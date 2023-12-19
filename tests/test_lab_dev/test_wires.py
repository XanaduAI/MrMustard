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
            wires.modes
        
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