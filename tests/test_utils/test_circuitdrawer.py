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

from mrmustard.lab import BSgate, Ggate
from mrmustard.utils.circdrawer import (
    _add_grouping_symbols,
    _add_op,
    circuit_text,
    drawable_layers,
    mode_set,
)


def test_mode_set():
    op = BSgate(0.5)[3, 11]
    assert mode_set(op) == set(range(3, 12))


def test_drawable_layers_overlap():
    ops = [BSgate(0.5)[3, 11], BSgate(0.5)[3, 11], BSgate(0.5)[3, 11]]
    assert drawable_layers(ops) == {0: [ops[0]], 1: [ops[1]], 2: [ops[2]]}


def test_drawable_layers_no_overlap():
    ops = [BSgate(0.5)[3, 11], BSgate(0.5)[12, 13], BSgate(0.5)[14, 15]]
    assert drawable_layers(ops) == {0: [ops[0], ops[1], ops[2]]}


def test_drawable_layers_mix_overlap():
    ops = [BSgate(0.5)[3, 11], BSgate(0.5)[3, 11], BSgate(0.5)[12, 13], BSgate(0.5)[14, 15]]
    assert drawable_layers(ops) == {0: [ops[0]], 1: [ops[1], ops[2], ops[3]]}


def test_add_grouping_symbols_BS():
    op = BSgate(0.5)[3, 11]
    assert _add_grouping_symbols(op, ["-"] * 12) == [
        "-",
        "-",
        "-",
        "╭",
        "│",
        "│",
        "│",
        "│",
        "│",
        "│",
        "│",
        "╰",
    ]


def test_add_grouping_symbols_G():
    op = Ggate(5)[1, 2, 3, 4, 5]
    assert _add_grouping_symbols(op, ["-"] * 6) == [
        "-",
        "╭",
        "├",
        "├",
        "├",
        "╰",
    ]


def test_add_op():
    op = Ggate(5)[1, 2, 3, 4, 5]
    layer_str = _add_grouping_symbols(op, ["-"] * 6)
    decimals = None
    assert _add_op(op, layer_str, decimals) == [
        "-",
        "╭G",
        "├G",
        "├G",
        "├G",
        "╰G",
    ]


def test_circuit_text():
    from mrmustard import settings

    settings.CIRCUIT_DECIMALS = None
    ops = [BSgate(0.5)[0, 1], Ggate(4)[2, 3, 4, 5], BSgate(0.5)[7, 6]]
    decimals = None
    assert (
        circuit_text(ops, decimals)
        == "0: ─╭•──\n1: ─╰BS─\n2: ─╭G──\n3: ─├G──\n4: ─├G──\n5: ─╰G──\n6: ─╭BS─\n7: ─╰•──"
    )
