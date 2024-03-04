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

# pylint: missing-function-docstring

import pytest

from mrmustard.lab_dev.wires import Wires


class TestWires:
    r"""
    Tests for the Wires class.
    """
    def test_args(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        assert w.args == ({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})

    def test_wire_subsets(self):
        w = Wires({0}, {1}, {2}, {3})
        assert w.output.bra.modes == {0}
        assert w.input.bra.modes == {1}
        assert w.output.ket.modes == {2}
        assert w.input.ket.modes == {3}

    def test_indices(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        assert w.output.indices == (0, 1, 2, 6, 7)
        assert w.bra.indices == (0, 1, 2, 3, 4, 5)
        assert w.input.indices == (3, 4, 5, 8)
        assert w.ket.indices == (6, 7, 8)

    def test_adjoint(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        w_adj = w.adjoint
        assert w.input.ket.modes == w_adj.input.bra.modes
        assert w.output.ket.modes == w_adj.output.bra.modes
        assert w.input.bra.modes == w_adj.input.ket.modes
        assert w.output.bra.modes == w_adj.output.ket.modes

    def test_dual(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        w_d = w.dual
        assert w.input.ket.modes == w_d.output.ket.modes
        assert w.output.ket.modes == w_d.input.ket.modes
        assert w.input.bra.modes == w_d.output.bra.modes
        assert w.output.bra.modes == w_d.input.bra.modes

    def test_add(self):
        w1 = Wires({0}, {1}, {2}, {3})
        w2 = Wires({1}, {2}, {3}, {4})
        w12 = Wires({0, 1}, {1, 2}, {2, 3}, {3, 4})
        assert (w1 + w2).modes == w12.modes

    def test_add_error(self):
        w1 = Wires({0}, {1}, {2}, {3})
        w2 = Wires({0}, {2}, {3}, {4})
        with pytest.raises(Exception):
            w1 + w2  # pylint: disable=pointless-statement

    def test_bool(self):
        assert Wires({0})
        assert not Wires({0}).input

    def test_getitem(self):
        w = Wires({0, 1}, {0, 1})
        w0 = w[0]
        w1 = w[1]
        assert w0.modes == {0}
        assert w0.indices == (w.indices[0], w.indices[2])
        assert w1.modes == {1}
        assert w1.indices == (w.indices[1], w.indices[3])

    def test_eq_neq(self):
        w1 = Wires({0, 1}, {2, 3}, {4, 5}, {6, 7})
        w2 = Wires({0, 1}, {2, 3}, {4, 5}, {6, 7})
        w3 = Wires(set(), {2, 3}, {4, 5}, {6, 7})
        w4 = Wires({0, 1}, set(), {4, 5}, {6, 7})
        w5 = Wires({0, 1}, {2, 3}, set(), {6, 7})
        w6 = Wires({0, 1}, {2, 3}, {4, 5}, set())

        assert w1 == w1
        assert w1 == w2
        assert w1 != w3
        assert w1 != w4
        assert w1 != w5
        assert w1 != w6

    def test_matmul(self):
        # contracts 1,1 on bra side
        # contracts 3,3 and 13,13 on ket side
        u = Wires({1, 5}, {2, 6, 15}, {3, 7, 13}, {4, 8})
        v = Wires({0, 9, 14}, {1, 10}, {2, 11}, {13, 3, 12})
        assert (u @ v).args == ({0, 5, 9, 14}, {2, 6, 10, 15}, {2, 7, 11}, {4, 8, 12})

    def test_matmul_error(self):
        u = Wires({}, {}, {0}, {})  # only output wire
        v = Wires({}, {}, {0}, {})  # only output wire
        with pytest.raises(ValueError):
            u @ v  # pylint: disable=pointless-statement
