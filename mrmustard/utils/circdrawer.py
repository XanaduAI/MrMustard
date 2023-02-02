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
"""
This module contains logic for the text-based circuit drawer for MrMustard
"""


def mode_set(op):
    "includes modes in between min and max of op.modes"
    return set(range(min(op.modes), max(op.modes) + 1))


def all_modes(ops):
    "returns a set of all modes used by the operations"
    return sorted(list(set().union(*[op.modes for op in ops])))


def drawable_layers(ops):
    r"""Determine non-overlapping yet dense placement of ops into layers for drawing.
    Arguments:
        ops Iterable[op]: a list of operations

    Returns:
        list[list[op]] : At index k is a list of operations for the k-th layer
    """
    layers = [[]]
    k = 0
    for new_op in ops:
        # if there's any overlap, add to next layer
        if any(mode_set(new_op) & mode_set(op) for op in layers[k]):
            k += 1
            layers.append([])
        layers[k].append(new_op)
    return layers


def _add_grouping_symbols(op, layer_str):
    r"""Adds symbols indicating the extent of a given object."""
    S = mode_set(op)
    if len(S) > 1:
        layer_str[min(S)] = "╭"
        layer_str[max(S)] = "╰"
        for w in range(min(S) + 1, max(S)):
            layer_str[w] = "├" if w in op.modes else "│"  # other option: ┼
    return layer_str


def _add_op(op, layer_str, decimals):
    """Updates ``layer_str`` with ``op`` operation."""
    layer_str = _add_grouping_symbols(op, layer_str)
    control = []
    if op.__class__.__qualname__ in ["BSgate", "MZgate", "CZgate", "CXgate"]:
        control = [op.modes[0]]
    label = op.label(decimals)

    for w in op.modes:
        if w in control:
            layer_str[w] += "•"
        else:
            layer_str[w] += label

    return layer_str


# pylint: disable=too-many-arguments
def circuit_text(
    ops,
    decimals=None,
    max_length=100,
):
    r"""Text based diagram for a Quantum circuit.
    Arguments:
        ops: the operations and measurements to draw as a list of MrMustard operations
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        max_length (Int) : Maximum length of a individual line. After this length, the diagram will
            begin anew beneath the previous lines.
    Returns:
        str : String based graphic of the circuit.
    """
    M = all_modes(ops)
    n_modes = len(M)

    totals = [f"{mode}: " for mode in M]
    line_length = max(len(s) for s in totals)
    totals = [s.rjust(line_length, " ") for s in totals]

    filler = "─"

    for layer in drawable_layers(ops):
        layer_str = [filler] * n_modes

        for op in layer:
            layer_str = _add_op(op, layer_str, decimals)

        max_label_len = max(len(s) for s in layer_str)
        layer_str = [s.ljust(max_label_len, filler) for s in layer_str]

        line_length += max_label_len + 1  # one for the filler character

        totals = [filler.join([t, s]) for t, s in zip(totals, layer_str)]
        totals = [s + filler for s in totals]

    return "\n".join(totals)
