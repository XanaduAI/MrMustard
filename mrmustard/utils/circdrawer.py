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
from collections import defaultdict
from typing import Any, Optional

Operation = Any  # actually Operation, but circular import


def mode_set(op: Operation):
    r"""includes modes in between min and max of op.modes_in because of vertical wires."""
    return set(range(min(op.modes), max(op.modes) + 1))


def drawable_layers(ops: list[Operation]) -> dict[int, list[Operation]]:
    r"""Determine non-overlapping yet dense placement of ops into layers for drawing.
    Arguments:
        ops Iterable[op]: a list of Operations

    Returns:
        dict[int:list[op]] : At index k is a list of operations for the k-th layer
    """
    layers = defaultdict(list)
    k = 0
    for new_op in ops:
        # if there's any overlap, move to next layer
        if any(mode_set(new_op) & mode_set(op) for op in layers[k]):
            k += 1
        # add the new op to the layer
        layers[k].append(new_op)
    return layers


def add_grouping_symbols(op: Operation, layer_str: dict[int, str]) -> dict[int, str]:
    r"""Adds symbols indicating the extent of a given object."""
    S = mode_set(op)
    if len(S) > 1 and not op.parallelizable:
        layer_str[min(S)] = "╭"
        layer_str[max(S)] = "╰"
        for w in range(min(S) + 1, max(S)):
            layer_str[w] = "├" if w in op.modes_out else "│"  # other option: ┼
    return layer_str


def add_op(
    op: Operation, layer_str: dict[int, str], decimals: Optional[int] = None
) -> dict[int, str]:
    r"""Updates `layer_str` with `op` operation."""

    # add grouping symbols
    layer_str = add_grouping_symbols(op, layer_str)

    # add the operation label and parameters
    control = []
    if op.__class__.__qualname__ in ["BSgate", "MZgate", "CZgate", "CXgate"]:
        control = [list(op.modes_out)[0]]
    label = op.short_name or op.__class__.__qualname__[:2]
    if decimals is not None:
        try:
            param_string = op.param_string(decimals)
        except AttributeError:
            param_string = str(len(op.modes_out))
        label += "(" + param_string + ")"

    # add the control symbol
    for w in op.modes_out:
        layer_str[w] += "•" if w in control else label

    return layer_str


def circuit_text(
    ops,
    decimals=None,
):
    r"""Text based diagram for a Quantum circuit.
    Arguments:
        ops (List[Operation]): the operations to draw as a list of MrMustard operations
        decimals (optional(int)): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
    Returns:
        str : String based graphic of the circuit.
    """
    # get all modes used by the ops and sort them
    modes = sorted(list(set().union(*[set(op.modes_out) | set(op.modes_in) for op in ops])))
    # include all modes between min and max (need to draw over them)
    all_modes = range(min(modes), max(modes) + 1)

    totals = [f"{mode}: " for mode in all_modes]
    line_length = max(len(s) for s in totals)
    totals = [s.rjust(line_length, " ") for s in totals]
    filler = "─"

    for layer in drawable_layers(ops).values():
        layer_str = [filler] * (max(all_modes) - min(all_modes) + 1)
        for op in layer:
            layer_str = add_op(op, layer_str, decimals)

        max_label_len = max(len(s) for s in layer_str)
        layer_str = [s.ljust(max_label_len, filler) for s in layer_str]

        line_length += max_label_len + 1  # one for the filler character

        totals = [filler.join([t, s]) + filler for t, s in zip(totals, layer_str)]

    return "\n".join(totals)
