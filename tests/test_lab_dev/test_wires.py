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
from mrmustard.lab_dev.wires import Wires


def test_wires_view_has_same_ids():
    w = Wires([0], [0], [0], [0])
    assert set(w.ids) == set(w._view().ids)


def test_view_can_edit_original():
    w = Wires([0], [0], [0], [0])
    w._view().ids = [9, 99, 999, 9999]
    assert w.ids == [9, 99, 999, 9999]


def test_wire_subsets():
    w = Wires([0], [1], [2], [3])
    assert w.output.bra.modes == [0]
    assert w.input.bra.modes == [1]
    assert w.output.ket.modes == [2]
    assert w.input.ket.modes == [3]


def test_wire_mode_subsets():
    w = Wires([10], [11], [12], [13])
    assert w[10].ids == w.output.bra.ids
    assert w[11].ids == w.input.bra.ids
    assert w[12].ids == w.output.ket.ids
    assert w[13].ids == w.input.ket.ids


def test_indices():
    w = Wires([0, 1, 2], [3, 4, 5], [6, 7], [8])
    assert w.output.indices == [0, 1, 2, 6, 7]
    assert w.bra.indices == [0, 1, 2, 3, 4, 5]
    assert w.input.indices == [3, 4, 5, 8]
    assert w.ket.indices == [6, 7, 8]


def test_setting_ids():
    w = Wires([0], [0], [0], [0])
    w.ids = [9, 99, 999, 9999]
    assert w.ids == [9, 99, 999, 9999]


def test_add_wires():
    w1 = Wires([0], [1], [2], [3])
    w2 = Wires([1], [2], [3], [4])
    w12 = Wires([0, 1], [1, 2], [2, 3], [3, 4])
    assert (w1 + w2).modes == w12.modes


def test_cant_add_overlapping_wires():
    w1 = Wires([0], [1], [2], [3])
    w2 = Wires([0], [2], [3], [4])
    with pytest.raises(Exception):
        w1 + w2


def test_args():
    w = Wires([0], [1], [2], [3])
    assert w._args() == ((0,), (1,), (2,), (3,))
