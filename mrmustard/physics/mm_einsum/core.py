# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, overload

from mrmustard import math
from mrmustard.physics.ansatz import Ansatz, ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum.contraction_path import normalize_path, validate_path
from mrmustard.physics.mm_einsum.conversions import to_fock

__all__ = ["mm_einsum"]

r"""
Einstein summation for quantum ansatzes with explicit batch and core dimension labeling.

The ``mm_einsum`` function performs Einstein summation over quantum ansatzes (PolyExpAnsatz, 
ArrayAnsatz, or raw arrays) using a three-phase process:

1. Bargmann contraction phase: Contracts PolyExpAnsatz pairs using Gaussian integrals following the provided path.
2. Fock conversion phase: Converts remaining PolyExpAnsatz to Fock-space arrays.
3. Final einsum phase: Performs the remaining array-based Einstein summation, reusing any
   unconsumed user-supplied contraction path.

Index Convention
================

The function uses a mixed-case indexing convention to distinguish dimension types:

* **UPPERCASE letters** (A-Z): Label batch dimensions
* **lowercase letters** (a-z): Label core (continuous variable) dimensions

All batch dimensions must be explicitly labeled, including eventual linear superposition axes for `PolyExpAnsatz`

Basic Usage
===========

The einsum equation follows the format: ``"input1,input2,...->output"``

Example with two operands::

    result = mm_einsum("Aab,Bbc->ABac", operand1, operand2, fock_dims={"a": 5, "b": 10, "c": 7})

This contracts over core dimension ``b`` while preserving:

* Batch dimensions ``A`` and ``B``
* Core dimensions ``a`` and ``c``

and using dimensions 5, 10, 7 for the Fock space arrays for eventual contractions in Fock and outputs.

Parentheses for Batch Grouping
===============================

Parentheses in the output string provide a simple way to group consecutive batch dimensions that
should be vectorized into a single flattened dimension:

    # Group batch dims A and B into one dimension of size A*B
    result = mm_einsum("Aa,Bb->(AB)ab", op1, op2, fock_dims={"a": 5, "b": 5})
    # Output shape: (A*B, 5, 5) instead of (A, B, 5, 5)

Rules for parentheses:

* Only batch dimensions (uppercase) can be grouped
* Groups must contain at least 2 letters
* Nested parentheses or half-open parentheses raise an error

Fock Dimensions
===============

The ``fock_dims`` parameter maps core dimension letters to their Fock space sizes::

    fock_dims = {
        "a": 5,   # Core dimension 'a' has size 5
        "b": 10,  # Core dimension 'b' has size 10
        "c": 7,   # Core dimension 'c' has size 7
    }

This parameter is required when converting PolyExpAnsatz to Fock arrays. Missing entries
will raise a ValueError during conversion. If no fock dims for a PolyExpAnsatz are provided,
it is assumed that the PolyExpAnsatz should never be converted to Fock representation.

Contraction Paths
=================

The ``contraction_path`` parameter specifies the order of contractions between ansatze.
Three path types are supported via the ``path_type`` parameter:

**Linear Assignment (LA)** - Default
    Each step `(i, j)` specifies which two operands in the **current** list to contract.
    After each contraction, the list shrinks: the two operands are removed and their
    contraction result is appended to the list.
    
    Example for 4 operands::
    
        path = [(0, 2), (1, 2), (0, 1)]
        current = [op0, op1, op2, op3]
        # Step 1: Contract current[0] and current[2] -> current = [op1, op3, op0@op2]
        # Step 2: Contract current[1] and current[2] -> current = [op1, op3@op0@op2]  
        # Step 4: return current[0]

**Static Single Assignment (SSA)**
    Each step `(i, j)` specifies which two operands in the **current** list to contract.
    After each contraction, the list grows: the contraction result is appended to the list.
    There is no actual list of ansatze that grows, this is only an indexing model.
    
    Example for 4 operands::
    
        path = [(1, 3), (0, 2), (4, 5)]
        current = [op0, op1, op2, op3]
        # Step 1: Contract current[1] and current[3] -> current = [op0, op1, op2, op3, op1@op3]
        # Step 2: Contract current[0] and current[2] -> current = [op0, op1, op2, op3, op1@op3, op0@op2]
        # Step 3: Contract current[4] and current[4] -> current = [op0, op1, op2, op3, op1@op3, op0@op2, op1@op3@op0@op2]
        # step 4: return current[-1]
        mm_einsum(eq, op0, op1, op2, op3, contraction_path=path, path_type="SSA")

**Union Assignment (UA)**
    Each step `(i, j)` specifies which **original operand IDs** to contract. The function
    tracks which original IDs have been merged together at each step. An ansatz can be referenced using
    any of its original IDs. One can think of the current list as maintaining its length, and at each step,
    all the ansatze involved in a contraction are replaced by the result.
    Also in this case this is just an indexing model, the actual computation is done efficiently.
    
    Example for 4 operands::
    
        path = [(0, 2), (1, 3), (0, 3)]  # or ...(0, 2)] or ...(1, 2)] or ...(1,3)]
        current = [op0, op1, op2, op3]
        # Step 1: Contract current[0] and current[2] -> current = [op0@op2, op1, op0@op2, op3]
        # Step 2: Contract current[1] and current[3] -> current = [op0@op2, op1@op3, op0@op2, op1@op3]
        # Step 3: Contract current[0] and current[3] -> current = [op0@op2@op1@op3, op1@op3, op0@op2, op0@op2@op1@op3]
        # Step 4: return current[0] or current[3]  # depends on the last pair in the path.
        mm_einsum(eq, op0, op1, op2, op3, contraction_path=path, path_type="UA")

If no path is provided, the function auto-contracts PolyExpAnsatz pairs that share
core dimensions. This may not be what one wants so it is recommended to provide a path.

Each type of path has its own advantages:
* The linear assignment is standard in numpy and opt_einsum.
* In the static single assignment, labels are unique throughout the contraction.
* The union assignment is the easiest to read, since you can refer to any intermediate ansatz by
  the ID of any original ansatz that was involved.

Linear Superposition
====================

When a PolyExpAnsatz has a linear superposition axis (``_lin_sup=True``), the last batch
dimension corresponds to the superposition index. This dimension is treated as follows:

* **Preserved** if its letter appears in the output string
* **Summed over** if its letter does not appear in the output string and it gets converted to an ArrayAnsatz.

Examples
========

**Basic contraction**::

    result = mm_einsum("a,a->", ansatz1, ansatz2)

**Basic contraction in Fock space**::

    result = mm_einsum("a,a->", ansatz1, ansatz2, fock_dims={"a": 10})

**Basic contraction with batch dims kroned**::

    result = mm_einsum("Aa,Ba->AB", ansatz1, ansatz2)

**Basic contraction with batch dims vectorized**::

    result = mm_einsum("Aa,Ba->(AB)", ansatz1, ansatz2)

**Basic contraction with batch dims zipped**::

    result = mm_einsum("Aa,Aa->A", ansatz1, ansatz2)

**Multiple contractions with path**::

    result = mm_einsum("Aa,Ba,ab->(AB)ab", 
                       ansatz1, ansatz2, ansatz3,
                       fock_dims={"a": 5, "b": 5},
                       contraction_path=[(0, 1), (0, 1)])

**Linear superposition handling**::

    # Preserve superposition axis L in output
    result = mm_einsum("La,b->Lab", ansatz_with_lin_sup, ansatz, 
                       fock_dims={"a": 5, "b": 5})
    
    # Sum over superposition axis L (L not in output)
    result = mm_einsum("La,b->ab", ansatz_with_lin_sup, ansatz,
                       fock_dims={"a": 5, "b": 5})
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utils
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Mimic NumPy's hardcoding for performance
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Rec:
    """Record of an ``Ansatz`` along with its batch labels and core labels at a given point."""

    ans: Ansatz
    batch_labels: list[int]
    core_labels: list[int]


def _parse_output_string(
    output_string: str, char_to_int: dict[str, int]
) -> list[int | tuple[int, ...]]:
    r"""Parses an output string in the subscript signature and returns the equivalent sublist format.

    Args:
        output_string: The output string to parse.
        char_to_int: Mapping from characters to integer labels.

    Returns:
        The list of output integer labels.

    Raises:
        ValueError: If an invalid output label is specified.
        ValueError: If an invalid parantheses is specified.
    """
    in_group, group, ret_output = False, [], []
    for char in output_string:
        _validate_parentheses(char, in_group)
        if char == "(":
            in_group = True
        elif char == ")":
            in_group = False
            if len(group) >= 2:
                ret_output.append(tuple(group))
            elif group:
                ret_output.append(group[0])
            group = []
        elif char.isalpha():
            integer_label = char_to_int.get(char)
            if integer_label is None:
                raise ValueError(f"Invalid output label {char}!")
            if in_group:
                group.append(integer_label)
            else:
                ret_output.append(integer_label)

    if in_group:
        raise ValueError("Unclosed '(' in output")

    return ret_output


def _split_by_case(s: str) -> tuple[str, str]:
    r"""Splits a string into (UPPERCASE, lowercase) letters.

    Args:
        s: The string to split.

    Returns:
        A tuple of the (UPPERCASE, lowercase) letters.
    """
    upper = "".join(c for c in s if c.isupper())
    lower = "".join(c for c in s if c.islower())
    return upper, lower


def _validate_parentheses(char: str, in_group: bool) -> None:
    r"""Validate parentheses in the output string.

    Args:
        char: The character to validate.
        in_group: Whether the character is in a parantheses group.

    Raises:
        ValueError: If the character is invalid.
    """
    if in_group:
        if char == "(":
            raise ValueError("Nested parentheses not supported")
        if char.islower():
            raise ValueError("Lowercase letters cannot be grouped")
    elif char == ")":
        raise ValueError("Unmatched ')' in output")


def build_records_from_operands(input_labels: list[int], ansatze: list[Ansatz]) -> list[Rec]:
    r"""Builds up a list of records given a list of ansatze and labels.

    Args:
        input_labels: The list of input labels.
        ansatze: The list of ansatze.

    Returns:
        The list of records.
    """
    records = []
    for ansatz, labels in zip(ansatze, input_labels, strict=True):
        batch_dims = ansatz.batch_dims
        records.append(Rec(ansatz, labels[:batch_dims], labels[batch_dims:]))
    return records


def convert_operands(
    operands: list[str | list[int | tuple[int, int]] | Ansatz], char_to_int: dict[str, int]
) -> list[Ansatz | list[int] | list[int | tuple[int, int]]]:
    r"""Converts ``operands`` to the sublist signature.

    Args:
        operands: The list of operands to convert.
        char_to_int: Mapping from characters to integer labels.

    Returns:
        The resulting list of operands in the sublist signature.

    Raises:
        ValueError: If the number of batch dimensions for an ``Ansatz`` does not match the number of labels specified.
        ValueError: If the number of CV variables for an ``Ansatz`` does not match the number of labels specified.
    """
    if not char_to_int:
        return operands

    subscripts = operands[0].replace(" ", "")  # filter whitespaces
    lhs, output_string = subscripts.split("->")
    inputs = lhs.split(",")

    ansatze = operands[1:]
    if len(ansatze) != len(inputs):
        raise ValueError("Number of inputs must match number of operands!")

    new_operands = []
    for ansatz, input_labels in zip(ansatze, inputs, strict=True):
        batch_dims = ansatz.batch_dims
        core_dims = ansatz.core_dims
        upper, lower = _split_by_case(input_labels)
        if batch_dims != len(upper):
            raise ValueError(
                f"Operand has {batch_dims} batch dims but got {len(upper)} batch letters. "
                f"All batch dimensions (including linear superposition axes) must be explicitly labeled."
            )
        if core_dims != len(lower):
            raise ValueError(f"Operand has {core_dims} CV vars but got {len(lower)} indices")
        new_operands.append(ansatz)
        new_operands.append([char_to_int[s] for s in input_labels])

    output = _parse_output_string(output_string, char_to_int)
    new_operands.append(output)

    return new_operands


def convert_fock_dims(
    fock_dims: dict[str, int] | dict[int, int] | None, char_to_int: dict[str, int]
) -> dict[int, int]:
    r"""Maps ``fock_dims`` from the subscript signature to the sublist signature.

    Args:
        fock_dims: Mapping from core labels to Fock sizes.
        char_to_int: Mapping from characters to integer labels.

    Returns:
        ``fock_dims`` in the sublist signature.
    """
    if not fock_dims:
        return {}
    if not char_to_int:
        return fock_dims
    return {char_to_int[k]: v for k, v in fock_dims.items()}


def convert_to_fock(
    records: list[Rec],
    fock_dims: dict[int, int],
    output_batch_labels: list[int],
    raise_if_missing_dims: bool,
) -> list[Rec]:
    r"""Convert ``PolyExpAnsatz`` to ``ArrayAnsatz`` if all core dimensions are in ``fock_dims`` keys.
    Preserves a linear superposition axis if it appears in the output batch, e.g.
    ``mm_einsum('Ax->Ax', ansatz, fock_dims={'x': 5})`` and ``ansatz`` is a ``PolyExpAnsatz`` with
    linear superposition axis ``A``.

    Args:
        records: The list of records.
        fock_dims: Mapping from core labels to Fock sizes.
        output_batch_labels: The output batch labels.
        raise_if_missing_dims: Whether to raise an error if a Fock dimension is missing.

    Returns:
        The resulting list of records.

    Raises:
        ValueError: If a Fock dimension is missing and ``raise_if_missing_dims`` is ``True``.
    """
    for i, rec in enumerate(records):
        ansatz = rec.ans
        if not isinstance(ansatz, PolyExpAnsatz):
            continue
        no_dims_provided = set(rec.core_labels).isdisjoint(fock_dims.keys())
        # batch_summation = any(c not in output_batch_str for c in rec.batch_str) and ansatz.num_CV_vars == 0
        # we would skip if no dims provided, but we may want to do batch summation
        if no_dims_provided:  # and not batch_summation:
            continue
        # at this point we must convert to fock
        can_convert = set(rec.core_labels).issubset(fock_dims)
        if not can_convert and raise_if_missing_dims:
            raise ValueError(
                f"Cannot convert PolyExpAnsatz {ansatz!s} to Fock: missing fock_dims for {set(rec.core_labels) - set(fock_dims.keys())}"
            )
        if can_convert:
            preserve_lin_sup = ansatz._lin_sup and rec.batch_labels[-1] in output_batch_labels
            remove_lin_sup = ansatz._lin_sup and not preserve_lin_sup
            shape = tuple(fock_dims[c] for c in rec.core_labels)
            array_ansatz = to_fock(ansatz, shape, preserve_lin_sup=preserve_lin_sup)
            # if lin sup was summed during conversion, remove the corresponding batch letter
            new_batch_labels = rec.batch_labels[:-1] if remove_lin_sup else rec.batch_labels
            records[i] = Rec(array_ansatz, new_batch_labels, rec.core_labels)
    return records


def do_leftover_polyexp_outer_product(records: list[Rec], fock_dims: dict[int, int]) -> list[Rec]:
    r"""Combine remaining ``PolyExpAnsatz`` to a single one via outer product.

    Args:
        records: The list of records.
        fock_dims: Mapping from core labels to Fock sizes.

    Returns:
        The resulting list of records.
    """
    polyexp_indices = [i for i, rec in enumerate(records) if isinstance(rec.ans, PolyExpAnsatz)]
    if len(polyexp_indices) > 1:
        all_can_convert_to_fock = all(
            rec.ans.num_CV_vars == 0 or set(rec.core_labels).issubset(set(fock_dims.keys()))
            for i, rec in enumerate(records)
            if i in polyexp_indices
        )
        if not all_can_convert_to_fock:
            while len(polyexp_indices) > 1:
                i, j = polyexp_indices[0], polyexp_indices[1]
                records[i] = _contract_polyexp_pair(records[i], records[j])
                records.pop(j)
                polyexp_indices = [
                    i for i, rec in enumerate(records) if isinstance(rec.ans, PolyExpAnsatz)
                ]
    return records


def generate_char_to_int(
    operands: list[str | list[int] | list[int | tuple[int, int]] | Ansatz],
) -> dict[str, int]:
    r"""Generate a mapping between character labels and integer labels given the subscript signature.

    Args:
        operands: The list of operands to map.

    Returns:
        A dictionary mapping character labels to integer labels.

    Raises:
        ValueError: If ``operands`` is in the subscript signature and the string is invalid.
    """
    if not isinstance(operands[0], str):
        return {}

    equation = operands[0]

    if equation.count("->") != 1:
        raise ValueError("Einsum string must contain exactly one '->' with explicit output indices")

    for s in equation:
        if s in " .,->()":
            continue
        if s not in einsum_symbols:
            raise ValueError(f"Character {s} is not a valid symbol.")

    lhs, _ = equation.split("->")
    return {char: idx for idx, char in enumerate({char for char in lhs if char not in {",", " "}})}


def parse_operands(
    operands: list[Ansatz | list[int] | list[int | tuple[int, ...]]],
) -> tuple[Ansatz, list[int], list[int], list[int], list[tuple[int, ...]]]:
    r"""Parses a list of operands in the sublist signature.

    Args:
        operands: The list of operands to parse.

    Returns:
        The ansatze, the input labels, the output batch labels, the output core labels, and the batch parantheses groups.
    """
    inputs = operands[1:-1:2]
    ansatze = operands[:-1:2]
    output = operands[-1]

    all_batch_labels = []
    all_core_labels = []
    for ansatz, input_labels in zip(ansatze, inputs, strict=True):
        batch_dim = ansatz.batch_dims
        all_batch_labels.extend(input_labels[:batch_dim])
        all_core_labels.extend(input_labels[batch_dim:])

    output_batch = []
    output_core = []
    groups = []
    for item in output:
        if isinstance(item, tuple):
            output_batch.extend([i for i in item if i in all_batch_labels])
            groups.append((output_batch.index(item[0]), output_batch.index(item[-1])))
        elif item in all_batch_labels:
            output_batch.append(item)
        elif item in all_core_labels:
            output_core.append(item)
    return ansatze, inputs, output_batch, output_core, groups


def validate_operands(
    operands: list[Ansatz | list[int] | list[int | tuple[int, ...]]],
    fock_dims: dict[int, int],
    char_to_int: dict[str, int],
):
    r"""Validates operands in the sublist signature.

    Args:
        operands: The operands to validate.
        fock_dims: Mapping from core labels to Fock sizes.
        char_to_int: Mapping from characters to integer labels.

    Raises:
        ValueError: If the number of operands is invalid.
        ValueError: If a batch label is invalid.
        ValueError: If a Fock dimension is missing.
    """
    if (len(operands) - 1) % 2:
        raise ValueError(
            "Incorrect number of operands! Ensure ansatz, inputs, and output are all provided."
        )

    all_batch_labels = []
    all_core_labels = []
    for i in range(0, len(operands) - 1, 2):
        ansatz = operands[i]
        labels = operands[i + 1]
        batch_dim = ansatz.batch_dims
        all_batch_labels.extend(labels[:batch_dim])
        all_core_labels.extend(labels[batch_dim:])

    output = [x for i in operands[-1] for x in (i if isinstance(i, tuple) else (i,))]
    core_counts = Counter(all_core_labels)
    missing_labels = []
    for label, count in core_counts.items():
        in_output = label in output
        will_contract = count > 1
        has_fock_dim = label in fock_dims

        if not in_output and not will_contract and not has_fock_dim:
            missing_labels.append(label)

    int_to_char = {v: k for k, v in char_to_int.items()}

    output_batch_count = Counter(output)
    for label, count in output_batch_count.items():
        if count > 1:
            l = int_to_char[label] if char_to_int else label
            raise ValueError(f"Invalid output batch label: {l}!")

    if missing_labels:
        if char_to_int:
            missing_labels = [int_to_char[l] for l in missing_labels]
        raise ValueError(
            f"Cannot convert PolyExpAnsatz to Fock: missing fock_dims for {set(missing_labels)}. "
            f"These core labels appear in only one input, are not in the output, "
            f"and have no fock_dims specified."
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Phase 1: Trace Out
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def phase_1_trace_out(records: list[Rec], output_core_labels: list[int]) -> list[Rec]:
    r"""The first phase where trace-out operations are applied.

    This is done for efficiency to minimize the number of indices that need to be contracted later.

    Args:
        records: A list of records.
        output_core_labels: The output core labels.

    Returns:
        A list of the updated records.
    """
    for rec in records:
        core_labels = rec.core_labels
        # if ``rec.core_labels`` contains a repeated label that is not in ``output_core_labels``, we trace out the corresponding axis
        repeated_labels = {
            c for c in core_labels if core_labels.count(c) > 1 and c not in output_core_labels
        }
        if repeated_labels:
            for label in repeated_labels:
                idxs = tuple(i for i, c in enumerate(core_labels) if c == label)
                rec.ans = rec.ans.trace(idxs[:1], idxs[1:])
                core_labels = [c for c in core_labels if c != label]

        rec.core_labels = core_labels
    return records


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Phase 2: Contract Bargmann
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _align_batch(
    ansatz: PolyExpAnsatz, current_batch_labels: list[int], target_batch_labels: list[int]
) -> PolyExpAnsatz:
    r"""Align PolyExpAnsatz batch dims by adding missing axes and permuting to match target batch letters.

    Args:
        ansatz: The ansatz.
        current_batch_labels: The current batch labels.
        target_batch_labels: The target batch labels.

    Returns:
        The resulting ``PolyExpAnsatz``.
    """
    if not target_batch_labels:
        return ansatz

    # Insert missing batch axes
    missing = [c for c in target_batch_labels if c not in current_batch_labels]
    A, b, c = ansatz.triple
    for _ in missing:
        A, b, c = math.expand_dims(A, 0), math.expand_dims(b, 0), math.expand_dims(c, 0)

    # Reorder to match target
    if current_batch_labels != target_batch_labels:
        all_letters = missing + current_batch_labels
        order = [all_letters.index(letter) for letter in target_batch_labels]
        ansatz = PolyExpAnsatz(A, b, c, lin_sup=ansatz._lin_sup).reorder_batch(order)
    else:
        ansatz = PolyExpAnsatz(A, b, c, lin_sup=ansatz._lin_sup)

    return ansatz


def _auto_contract_remaining_bargmann_pairs(records: list[Rec]) -> list[Rec]:
    r"""Auto-contract remaining Bargmann pairs when no user path was supplied.

    Args:
        records: The list of records.

    Returns:
        The resulting list of records.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                both_polyexp = isinstance(records[i].ans, PolyExpAnsatz) and isinstance(
                    records[j].ans, PolyExpAnsatz
                )
                have_common_core = any(c in records[j].core_labels for c in records[i].core_labels)
                if both_polyexp and have_common_core:
                    new_record = _contract_polyexp_pair(records[i], records[j])
                    if new_record:
                        records.pop(j)
                        records[i] = new_record
                        changed = True
                        break
            if changed:
                break

    return records


def _consume_bargmann_path_prefix(
    records: list[Rec], steps_la: list[tuple[int, int]]
) -> tuple[list[Rec], list[tuple[int, int]]]:
    r"""Consume the path prefix that can be executed purely in Bargmann space.

    Args:
        records: The list of records.
        steps_la: The list of steps in "LA" order.

    Returns:
        The resulting records and any remaining steps.

    Raises:
        ValueError: If the contraction path is invalid.
    """
    remaining_steps = list(steps_la)
    while remaining_steps:
        step = remaining_steps[0]
        if len(step) == 1:
            break
        a, b = step  # NOTE: linear assignment steps, that's why we pop largest
        try:
            new_record = _contract_polyexp_pair(records[a], records[b])
        except IndexError as err:
            raise ValueError(
                f"Invalid contraction_path for Bargmann phase: {remaining_steps[0]}"
            ) from err
        if not new_record:
            break
        records.pop(max(a, b))
        records[min(a, b)] = new_record
        remaining_steps.pop(0)
    return records, remaining_steps


def _contract_polyexp_pair(ra: Rec, rb: Rec) -> Rec | None:
    r"""Contract a pair of ``PolyExpAnsatz``.

    Args:
        ra: The first record containing a ``PolyExpAnsatz``.
        rb: The second record containing a ``PolyExpAnsatz``.

    Returns:
        The resulting record.
    """
    if not isinstance(ra.ans, PolyExpAnsatz) or not isinstance(rb.ans, PolyExpAnsatz):
        return None

    ansatz_a, ansatz_b = ra.ans, rb.ans
    target_batch = list(dict.fromkeys(ra.batch_labels + rb.batch_labels))
    result_lin_sup = ansatz_a._lin_sup or ansatz_b._lin_sup

    # If result has lin_sup, ensure the lin_sup letter ends up at the last position
    if result_lin_sup:
        lin_sup_label = None
        if ansatz_a._lin_sup and ra.batch_labels:
            lin_sup_label = ra.batch_labels[-1]
        elif ansatz_b._lin_sup and rb.batch_labels:
            lin_sup_label = rb.batch_labels[-1]

        if lin_sup_label and lin_sup_label in target_batch and target_batch[-1] != lin_sup_label:
            target_batch = [c for c in target_batch if c != lin_sup_label] + [lin_sup_label]

    ansatz_a = _align_batch(ansatz_a, ra.batch_labels, target_batch)
    ansatz_b = _align_batch(ansatz_b, rb.batch_labels, target_batch)

    # Find common core letters and perform Gaussian integral
    common = [c for c in ra.core_labels if c in rb.core_labels]
    idx1 = [ra.core_labels.index(c) for c in common]
    idx2 = [rb.core_labels.index(c) for c in common]
    A, b, log_c = math.complex_gaussian_integral_2(
        ansatz_a.A, ansatz_a.b, ansatz_b.A, ansatz_b.b, idx1, idx2
    )

    # Reorder core axes if we contracted
    if common:
        s1, s2, s3, s4 = (
            ansatz_a.num_CV_vars - len(common),
            ansatz_a.num_derived_vars,
            ansatz_b.num_CV_vars - len(common),
            ansatz_b.num_derived_vars,
        )
        order = (
            list(range(s1))
            + list(range(s1 + s2, s1 + s2 + s3))
            + list(range(s1, s1 + s2))
            + list(range(s1 + s2 + s3, s1 + s2 + s3 + s4))
        )
        A, b = math.gather(math.gather(A, order, -1), order, -2), math.gather(b, order, -1)

    # Combine c tensors
    if ansatz_a.num_derived_vars or ansatz_b.num_derived_vars:
        av = "".join(chr(97 + i) for i in range(ansatz_a.num_derived_vars))
        bv = "".join(
            chr(97 + ansatz_a.num_derived_vars + i) for i in range(ansatz_b.num_derived_vars)
        )
        c = math.exp(
            log_c + math.log(math.einsum(f"...{av},...{bv}->...{av}{bv}", ansatz_a.c, ansatz_b.c))
        )
    else:
        c = math.exp(log_c + math.log(ansatz_a.c) + math.log(ansatz_b.c))

    core_out = [c for c in ra.core_labels if c not in common] + [
        c for c in rb.core_labels if c not in common
    ]
    return Rec(PolyExpAnsatz(A, b, c, lin_sup=result_lin_sup), target_batch, core_out)


def phase_2_contract_bargmann(
    records: list[Rec], steps_la: list[tuple[int, int]] | None
) -> tuple[list[Rec], list[tuple[int, int]] | None]:
    r"""The second phase where ``PolyExpAnsatz`` are contracted.

    If ``steps_la`` is specified then only those in the path prefix will
    be consumed. Otherwise, all ``PolyExpAnsatz`` are contracted.

    Args:
        records: The list of records at this stage.
        steps_la: The list of steps in "LA" order.

    Returns:
        The resulting records and any unconsumed path suffix.
    """
    if steps_la is not None:
        return _consume_bargmann_path_prefix(records, steps_la)

    records = _auto_contract_remaining_bargmann_pairs(records)
    return records, None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Phase 3: Final Einsum
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _collapse_grouped_batch_dims_arrayansatz(
    result: ArrayAnsatz, groups: list[tuple[int, int]]
) -> ArrayAnsatz:
    r"""Collapse grouped batch dims for an ``ArrayAnsatz``.

    Args:
        result: The ``ArrayAnsatz`` to collapse.
        groups: The batch parantheses groups.

    Returns:
        The resulting ``ArrayAnsatz``.
    """
    new_batch_shape = _compute_collapsed_shape(list(result.batch_shape), groups)
    return ArrayAnsatz(
        math.reshape(result.array, new_batch_shape + list(result.core_shape)),
        batch_dims=len(new_batch_shape),
    )


def _collapse_grouped_batch_dims_polyexpansatz(
    result: PolyExpAnsatz, groups: list[tuple[int, int]]
) -> PolyExpAnsatz:
    r"""Collapse grouped batch dims for a ``PolyExpAnsatz``.

    Args:
        result: The ``PolyExpAnsatz`` to collapse.
        groups: The batch parantheses groups.

    Returns:
        The resulting ``PolyExpAnsatz``.
    """
    new_batch_shape = _compute_collapsed_shape(list(result.batch_shape), groups)
    # A and b always have shape [...batch, 2*n, 2*n] and [...batch, 2*n]
    new_A = math.reshape(result.A, new_batch_shape + list(result.A.shape[-2:]))
    new_b = math.reshape(result.b, new_batch_shape + list(result.b.shape[-1:]))
    # c has shape [...batch, ...derived], so we keep everything after the original batch dims
    new_c = math.reshape(result.c, new_batch_shape + list(result.c.shape[result.batch_dims :]))
    return PolyExpAnsatz(new_A, new_b, new_c, lin_sup=result._lin_sup)


def _compute_collapsed_shape(batch_shape: list[int], groups: list[tuple[int, int]]) -> list[int]:
    r"""Compute the new shape after collapsing grouped dimensions.

    Args:
        batch_shape: The batch shape to collapse.
        groups: The batch parantheses groups.

    Returns:
        The resulting collapsed shape.
    """
    new_shape, cursor = [], 0
    for start, end in groups:
        new_shape.extend(batch_shape[cursor:start])
        new_shape.append(int(math.prod(batch_shape[start : end + 1])))
        cursor = end + 1
    new_shape.extend(batch_shape[cursor:])
    return new_shape


def early_return_single_polyexp(
    rec: Rec,
    output_batch_labels: list[tuple[int, int] | int],
    output_core_labels: list[int],
    groups: list[tuple[int, int]],
) -> PolyExpAnsatz:
    r"""If only a single ``PolyExpAnsatz`` remains handle the batch dimensions.

    Args:
        rec: The final record.
        output_batch_labels: The output batch labels.
        output_core_labels: The output core labels.
        groups: The batch parantheses groups.

    Returns:
        The resulting ``PolyExpAnsatz``.

    Raises:
        ValueError: If a ``PolyExpAnsatz`` with remaining CV variables requires an explicit summation over batch dimensions.
    """
    result = rec.ans

    to_add = [i for i in rec.batch_labels if i not in output_batch_labels]
    if len(to_add) > 0:
        if result.num_CV_vars > 0:
            raise ValueError(
                f"For a PolyExpAnsatz result we cannot do explicit summation over batch dimensions {to_add}."
            )
        # For scalar PolyExpAnsatz, sum over batch dimensions
        c_summed = result.c
        for letter in to_add:
            axis = rec.batch_labels.index(letter)
            c_summed = math.sum(c_summed, axis=axis)
            rec.batch_labels = rec.batch_labels[:axis] + rec.batch_labels[axis + 1 :]
        A = math.zeros((*c_summed.shape, 0, 0), dtype=c_summed.dtype)
        b = math.zeros((*c_summed.shape, 0), dtype=c_summed.dtype)
        result = PolyExpAnsatz(A, b, c_summed, lin_sup=result._lin_sup)

    # Reorder batch dimensions before grouping
    if output_batch_labels and rec.batch_labels != output_batch_labels:
        order = [rec.batch_labels.index(c) for c in output_batch_labels if c in rec.batch_labels]
        if len(order) == result.batch_dims:
            result = result.reorder_batch(order)

    result = _collapse_grouped_batch_dims_polyexpansatz(result, groups) if groups else result

    if output_core_labels and rec.core_labels != output_core_labels:
        order = [rec.core_labels.index(c) for c in output_core_labels]
        result = result.reorder(order)
    return result


def phase_3_final_einsum(
    records: list[Rec],
    output_batch_labels: list[int],
    output_core_labels: list[int],
    groups: list[tuple[int, int]],
    steps_la: list[tuple[int, int]] | None,
) -> ArrayAnsatz:
    r"""The final phase where the remaining ``ArrayAnsatz`` are contracted
    and grouped batch dimensions are collapsed.

    Args:
        records: The list of records at this stage.
        output_batch_labels: The list of output batch labels.
        output_core_labels: The list of output core labels.
        groups: The batch parantheses groups.
        steps_la: The list of steps in "LA" order.

    Returns:
        The resulting ``ArrayAnsatz``.

    Raises:
        ValueError: If the final contraction path is invalid.
    """
    # Extract arrays and build equation
    arrays = [r.ans.array for r in records]
    labels = [r.batch_labels + r.core_labels for r in records]

    # Compute minimum size for each label and slice arrays
    label_min = {}
    for arr, idx in zip(arrays, labels, strict=True):
        for axis, c in enumerate(idx):
            label_min[c] = min(label_min.get(c, arr.shape[axis]), arr.shape[axis])

    sliced = []
    for arr, idx in zip(arrays, labels, strict=True):
        slices = []
        for axis, c in enumerate(idx):
            if arr.shape[axis] > label_min[c]:
                slices.append(slice(None, label_min[c]))
            else:
                slices.append(slice(None))
        sliced.append(arr[tuple(slices)])

    # Perform einsum
    operands = [x for pair in zip(sliced, labels, strict=True) for x in pair]
    operands.append(output_batch_labels + output_core_labels)

    try:
        if steps_la is None:
            result_array = math.einsum(*operands)
        else:
            explicit_steps = [(0,)] if len(records) == 1 and len(steps_la) == 0 else steps_la
            result_array = math.einsum(*operands, optimize=explicit_steps)
    except (IndexError, TypeError, ValueError) as err:
        if steps_la is None:
            raise
        raise ValueError(f"Invalid contraction_path for final einsum: {err}") from err

    # Collapse grouped batch dims
    if groups:
        result_ans = ArrayAnsatz(result_array, batch_dims=len(output_batch_labels))
        return _collapse_grouped_batch_dims_arrayansatz(result_ans, groups)
    return ArrayAnsatz(result_array, batch_dims=len(output_batch_labels))


def raise_if_any_polyexp_left(
    records: list[Rec], fock_dims: dict[int, int], char_to_int: dict[str, int]
):
    r"""Raise an error if there are any ``PolyExpAnsatz`` with core dimensions that are not in ``fock_dims``.

    Args:
        records: The list of records.
        fock_dims: Mapping from core labels to Fock sizes.
        char_to_int: Mapping from characters to integer labels.

    Raises:
        ValueError: If a Fock dimension is missing.
    """
    for rec in records:
        if isinstance(rec.ans, PolyExpAnsatz):
            missing_dims = set(rec.core_labels) - set(fock_dims.keys())
            if char_to_int:
                int_to_char = {v: k for k, v in char_to_int.items()}
                missing_dims = [int_to_char[l] for l in missing_dims]
            raise ValueError(
                f"Couldn't convert PolyExpAnsatz {rec.ans!s}: missing fock_dims for {set(missing_dims)}"
            )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mm_einsum
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@overload
def mm_einsum(
    *operands: Ansatz | list[int] | list[int | tuple[int, ...]],
    fock_dims: dict[int, int] | None,
    contraction_path: list[tuple[int, int]] | None = None,
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> Ansatz: ...


@overload
def mm_einsum(
    *operands: str | Ansatz,
    fock_dims: dict[str, int] | None,
    contraction_path: list[tuple[int, int]] | None = None,
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> Ansatz: ...


def mm_einsum(
    *operands: str | Ansatz | list[int] | list[int | tuple[int, ...]],
    fock_dims: dict[str, int] | dict[int, int] | None = None,
    contraction_path: list[tuple[int, int]] | None = None,
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> Ansatz:
    r"""Performs Einstein summation over ansatze with explicit batch and core dimension labeling.
    All batch dimensions (including an eventual linear superposition axis) must be explicitly labeled.

    Similar to ``np.einsum``, two signatures are supported:
        - Subscript style:
            The first operand is a string where batch dimensions are explicitly labeled with UPPERCASE letters,
            while core dimensions use lowercase letters. The subscripts for summation are a comma separated list
            of subscript labels with explicit output indices following a `->` indicator.
        - Sublist style:
            The operands must be in the form ``ansatz0, labels0, ansatz1, labels1, ..., output`` where ``labels``
            are a list of integers labeling indices for the preceding ansatz in the list of operands
            and output is a list of integers labeling output indices for the resulting ansatz.

    In both signatures output batch labels can be group with parentheses for batch grouping support.

    Args:
        *operands: The operands for the contraction in either subscript style or sublist style.
        fock_dims: Mapping from core labels to Fock sizes. Required if converting a ``PolyExpAnsatz`` to Fock.
        contraction_path: List of contraction steps over operand IDs. When supplied, it is used
            end-to-end: the Bargmann phase consumes the path prefix it can execute, and the final
            array/Fock einsum consumes the remaining suffix.
        path_type: Path interpretation method ("LA", "SSA", or "UA"). Default is "LA".

    Returns:
        Final ``ArrayAnsatz`` or ``PolyExpAnsatz`` depending on conversion requirements.

    Raises:
        ValueError: If the equation is invalid or the operands are not compatible.
    """
    # conversions to a single implementation
    char_to_int = generate_char_to_int(operands)
    operands = convert_operands(operands, char_to_int)
    fock_dims = convert_fock_dims(fock_dims, char_to_int)

    # validate and parse
    validate_operands(operands, fock_dims, char_to_int)
    ansatze, inputs, output_batch, output_core, groups = parse_operands(operands)

    # run
    steps = normalize_path(contraction_path, path_type)
    records = build_records_from_operands(inputs, ansatze)
    records = convert_to_fock(records, fock_dims, output_batch, raise_if_missing_dims=False)
    records = phase_1_trace_out(records, output_core)
    records, steps = phase_2_contract_bargmann(records, steps)
    records = convert_to_fock(records, fock_dims, output_batch, raise_if_missing_dims=True)
    if steps is None:
        records = do_leftover_polyexp_outer_product(records, fock_dims)
    validate_path(steps, len(records))
    if len(records) == 1 and isinstance(records[0].ans, PolyExpAnsatz):
        return early_return_single_polyexp(records[0], output_batch, output_core, groups)
    raise_if_any_polyexp_left(records, fock_dims, char_to_int)
    return phase_3_final_einsum(records, output_batch, output_core, groups, steps)
