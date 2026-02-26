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
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Literal

from numpy.typing import ArrayLike
from opt_einsum.paths import ssa_to_linear

from mrmustard import math
from mrmustard.physics.ansatz import Ansatz, ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.triples import identity_Abc

r"""
Einstein summation for quantum ansatzes with explicit batch and core dimension labeling.

The ``mm_einsum`` function performs Einstein summation over quantum ansatzes (PolyExpAnsatz, 
ArrayAnsatz, or raw arrays) using a three-phase process:

1. Bargmann contraction phase: Contracts PolyExpAnsatz pairs using Gaussian integrals following the provided path.
2. Fock conversion phase: Converts remaining PolyExpAnsatz to Fock-space arrays.
3. Final einsum phase: Performs standard array-based Einstein summation.

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

__all__ = ["mm_einsum", "to_bargmann", "to_fock"]


# ~~~~~~~
# Utility
# ~~~~~~~


@dataclass
class Rec:
    """Record of operand with its batch and core index strings."""

    ans: Ansatz | ArrayLike
    batch_str: str
    core_str: str

    def __post_init__(self):
        if self.ans.batch_dims != len(self.batch_str):
            raise ValueError(
                f"Operand has {self.ans.batch_dims} batch dims but got {len(self.batch_str)} batch letters. "
                f"All batch dimensions (including linear superposition axes) must be explicitly labeled."
            )
        if len(self.core_str) != self.ans.core_dims:
            raise ValueError(
                f"Operand has {self.ans.core_dims} CV vars but got {len(self.core_str)} indices"
            )


def _align_batch(
    ansatz: PolyExpAnsatz, current_batch_letters: str, target_batch_letters: str
) -> PolyExpAnsatz:
    """Align PolyExpAnsatz batch dims by adding missing axes and permuting to match target batch letters."""
    if not target_batch_letters:
        return ansatz

    # Insert missing batch axes
    missing = [c for c in target_batch_letters if c not in current_batch_letters]
    A, b, c = ansatz.A, ansatz.b, ansatz.c
    for _ in missing:
        A, b, c = math.expand_dims(A, 0), math.expand_dims(b, 0), math.expand_dims(c, 0)

    # Reorder to match target
    if current_batch_letters != target_batch_letters:
        all_letters = list(missing) + list(current_batch_letters)
        order = [all_letters.index(letter) for letter in target_batch_letters]
        ansatz = PolyExpAnsatz(A, b, c, lin_sup=ansatz._lin_sup).reorder_batch(order)
    else:
        ansatz = PolyExpAnsatz(A, b, c, lin_sup=ansatz._lin_sup)

    return ansatz


def _build_records_from_operands(
    input_strings: list[str], operands: list[Ansatz | ArrayLike]
) -> list[Rec]:
    """Goes from e.g. ('ABab', op) to Rec(op, 'AB', 'ab') and wraps raw arrays in ArrayAnsatz."""
    records = []
    for op_raw, string in zip(operands, input_strings):
        upper, lower = _split_by_case(string)
        op = (
            op_raw
            if isinstance(op_raw, ArrayAnsatz | PolyExpAnsatz)
            else ArrayAnsatz(math.astensor(op_raw), batch_dims=len(upper))
        )
        records.append(Rec(op, upper, lower))
    return records


def _collapse_grouped_batch_dims_arrayansatz(
    result: ArrayAnsatz, groups: list[tuple[int, int]]
) -> ArrayAnsatz:
    """Collapse grouped batch dims for ArrayAnsatz."""
    new_batch_shape = _compute_collapsed_shape(list(result.batch_shape), groups)
    return ArrayAnsatz(
        math.reshape(result.array, new_batch_shape + list(result.core_shape)),
        batch_dims=len(new_batch_shape),
    )


def _collapse_grouped_batch_dims_polyexpansatz(
    result: PolyExpAnsatz, groups: list[tuple[int, int]]
) -> PolyExpAnsatz:
    """Collapse grouped batch dims for PolyExpAnsatz."""
    new_batch_shape = _compute_collapsed_shape(list(result.batch_shape), groups)
    # A and b always have shape [...batch, 2*n, 2*n] and [...batch, 2*n]
    new_A = math.reshape(result.A, new_batch_shape + list(result.A.shape[-2:]))
    new_b = math.reshape(result.b, new_batch_shape + list(result.b.shape[-1:]))
    # c has shape [...batch, ...derived], so we keep everything after the original batch dims
    new_c = math.reshape(result.c, new_batch_shape + list(result.c.shape[result.batch_dims :]))
    return PolyExpAnsatz(new_A, new_b, new_c, lin_sup=result._lin_sup)


def _compute_collapsed_shape(batch_shape: list[int], groups: list[tuple[int, int]]) -> list[int]:
    """Compute the new shape after collapsing grouped dimensions."""
    new_shape, cursor = [], 0
    for start, end in groups:
        new_shape.extend(batch_shape[cursor:start])
        new_shape.append(int(math.prod(batch_shape[start:end])))
        cursor = end
    new_shape.extend(batch_shape[cursor:])
    return new_shape


def _contract_polyexp_pair(ra: Rec, rb: Rec) -> Rec | None:
    """Contract two PolyExpAnsatz operands via Gaussian integral."""
    if not isinstance(ra.ans, PolyExpAnsatz) or not isinstance(rb.ans, PolyExpAnsatz):
        return None

    ansatz_a, ansatz_b = ra.ans, rb.ans
    target_batch = "".join(dict.fromkeys(ra.batch_str + rb.batch_str))
    result_lin_sup = ansatz_a._lin_sup or ansatz_b._lin_sup

    # If result has lin_sup, ensure the lin_sup letter ends up at the last position
    if result_lin_sup:
        lin_sup_letter = None
        if ansatz_a._lin_sup and ra.batch_str:
            lin_sup_letter = ra.batch_str[-1]
        elif ansatz_b._lin_sup and rb.batch_str:
            lin_sup_letter = rb.batch_str[-1]

        if lin_sup_letter and lin_sup_letter in target_batch and target_batch[-1] != lin_sup_letter:
            target_batch = target_batch.replace(lin_sup_letter, "") + lin_sup_letter

    ansatz_a = _align_batch(ansatz_a, ra.batch_str, target_batch)
    ansatz_b = _align_batch(ansatz_b, rb.batch_str, target_batch)

    # Find common core letters and perform Gaussian integral
    common = [c for c in ra.core_str if c in rb.core_str]
    idx1 = [ra.core_str.index(c) for c in common]
    idx2 = [rb.core_str.index(c) for c in common]
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

    core_out = "".join(c for c in ra.core_str if c not in common) + "".join(
        c for c in rb.core_str if c not in common
    )
    return Rec(PolyExpAnsatz(A, b, c, lin_sup=result_lin_sup), target_batch, core_out)


def _convert_to_fock(
    records: list[Rec],
    fock_dims: dict[str, int],
    output_batch_str: str,
    raise_if_missing_dims: bool,
) -> list[Rec]:
    """Convert PolyExpAnsatz to ArrayAnsatz if all core dimensions are in fock_dims keys.
    preserves a linear superposition axis if it appears in the output batch, e.g.
    `mm_einsum('Ax->Ax', ansatz, fock_dims={'x': 5})` and `ansatz` is a `PolyExpAnsatz` with
    linear superposition axis `A`."""
    for i, rec in enumerate(records):
        ansatz = rec.ans
        if not isinstance(ansatz, PolyExpAnsatz):
            continue
        no_dims_provided = set(rec.core_str).isdisjoint(fock_dims.keys())
        # batch_summation = any(c not in output_batch_str for c in rec.batch_str) and ansatz.num_CV_vars == 0
        # we would skip if no dims provided, but we may want to do batch summation
        if no_dims_provided:  # and not batch_summation:
            continue
        # at this point we must convert to fock
        can_convert = set(rec.core_str).issubset(fock_dims)
        if not can_convert and raise_if_missing_dims:
            raise ValueError(
                f"Cannot convert PolyExpAnsatz {ansatz!s} to Fock: missing fock_dims for {set(rec.core_str) - set(fock_dims.keys())}"
            )
        if can_convert:
            preserve_lin_sup = ansatz._lin_sup and rec.batch_str[-1] in output_batch_str
            remove_lin_sup = ansatz._lin_sup and not preserve_lin_sup
            shape = tuple(fock_dims[c] for c in rec.core_str)
            array_ansatz = to_fock(ansatz, shape, preserve_lin_sup=preserve_lin_sup)
            # if lin sup was summed during conversion, remove the corresponding batch letter
            new_batch_str = rec.batch_str[:-1] if remove_lin_sup else rec.batch_str
            records[i] = Rec(array_ansatz, new_batch_str, rec.core_str)
    return records


def _do_leftover_polyexp_outer_product(records: list[Rec], fock_dims: dict[str, int]) -> list[Rec]:
    """Combine remaining PolyExpAnsatz to a single one via outer product."""
    polyexp_indices = [i for i, rec in enumerate(records) if isinstance(rec.ans, PolyExpAnsatz)]
    if len(polyexp_indices) > 1:
        all_can_convert_to_fock = all(
            rec.ans.num_CV_vars == 0 or set(rec.core_str).issubset(set(fock_dims.keys()))
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


def _early_return_single_polyexp(
    rec: Rec,
    out_batch_str: str,
    out_core_str: str,
    groups: list[tuple[int, int]],
) -> PolyExpAnsatz:
    """Return single PolyExpAnsatz."""
    result = rec.ans

    to_add = list(set(rec.batch_str) - set(out_batch_str))
    if len(to_add) > 0:
        if result.num_CV_vars > 0:
            raise ValueError(
                f"For a PolyExpAnsatz result we cannot do explicit summation over batch dimensions {to_add}."
            )
        # For scalar PolyExpAnsatz, sum over batch dimensions
        c_summed = result.c
        for letter in to_add:
            axis = rec.batch_str.index(letter)
            c_summed = math.sum(c_summed, axis=axis)
            rec.batch_str = rec.batch_str[:axis] + rec.batch_str[axis + 1 :]
        A = math.zeros((*c_summed.shape, 0, 0), dtype=c_summed.dtype)
        b = math.zeros((*c_summed.shape, 0), dtype=c_summed.dtype)
        result = PolyExpAnsatz(A, b, c_summed, lin_sup=result._lin_sup)

    # Reorder batch dimensions before grouping
    if rec.batch_str != out_batch_str and out_batch_str:
        order = [rec.batch_str.index(c) for c in out_batch_str if c in rec.batch_str]
        if len(order) == result.batch_dims:
            result = result.reorder_batch(order)

    result = _collapse_grouped_batch_dims_polyexpansatz(result, groups) if groups else result

    if out_core_str and rec.core_str != out_core_str:
        order = [rec.core_str.index(c) for c in out_core_str]
        result = result.reorder(order)
    return result


def _normalize_path(path: list[tuple[int, ...]] | None, path_type: str) -> list[tuple[int, int]]:
    """Normalize contraction path to linear assignment of binary steps.
    Completes the path if it's incomplete. Assumes well-formed path."""
    if not path:
        return []
    if path_type == "LA":
        steps = list(path)
    elif path_type == "SSA":
        steps = ssa_to_linear(path)
    elif path_type == "UA":  # Convert union assignment to linear assignment
        steps = ua_to_linear(path)
    return steps


def _parse_output_string(output_string: str) -> tuple[str, str, list[tuple[int, int]]]:
    """Parse output string, extracting batch/core letters and parenthesized group spans."""
    upper, lower, groups = "", "", []
    in_group, group_start = False, None

    for char in output_string:
        _validate_parentheses(char, in_group)

        if char == "(":
            in_group, group_start = True, len(upper)
        elif char == ")":
            in_group = False
            at_least_two_letters = len(upper) - group_start >= 2
            if at_least_two_letters:
                groups.append((group_start, len(upper)))
        elif char.isupper():
            upper += char
        elif char.islower():
            lower += char

    if in_group:
        raise ValueError("Unclosed '(' in output")

    return upper, lower, groups


def _phase_1_trace_out(records: list[Rec], out_core_str: str) -> list[Rec]:
    r"""
    Apply trace-out operations as the first step of the algorithm.
    This is done for efficiency to minimize the number of indices that need to be contracted later.

    Args:
        records: A list of records.
        out_core_str: The output core string.

    Returns:
        A list of the updated records.
    """
    for rec in records:
        core_str = rec.core_str
        # if ``rec.core_str`` contains a repeated letter that is not in the output core string, we trace out the corresponding axis
        repeated_letters = {c for c in core_str if core_str.count(c) > 1 and c not in out_core_str}
        if repeated_letters:
            for letter in repeated_letters:
                idxs = tuple(i for i, c in enumerate(core_str) if c == letter)
                rec.ans = rec.ans.trace(idxs[:1], idxs[1:])
                core_str = core_str.replace(letter, "")
        rec.core_str = core_str
    return records


def _phase_2_contract_bargmann(records: list[Rec], steps_la: list[tuple[int, int]]) -> list[Rec]:
    """Contract PolyExpAnsatz pairs following the provided path and continue with
    remaining pairs with shared core letters in order of appearance."""
    for a, b in steps_la:  # NOTE: linear assignment steps, that's why we pop largest
        new_record = _contract_polyexp_pair(records[a], records[b])
        if new_record:
            records.pop(max(a, b))
            records[min(a, b)] = new_record

    # Auto-contract remaining pairs with shared core letters
    changed = True
    while changed:
        changed = False
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                both_polyexp = isinstance(records[i].ans, PolyExpAnsatz) and isinstance(
                    records[j].ans, PolyExpAnsatz
                )
                have_common_core = any(c in records[j].core_str for c in records[i].core_str)
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


def _phase_3_final_einsum(
    records: list[Rec], out_batch_str: str, out_core_str: str, groups: list[tuple[int, int]]
) -> ArrayAnsatz:
    """Phase (c): Final array einsum."""
    # Extract arrays and build equation
    arrays = [r.ans.array for r in records]
    indices = [r.batch_str + r.core_str for r in records]

    # Compute minimum size for each letter and slice arrays
    letter_min = {}
    for arr, idx in zip(arrays, indices):
        for axis, c in enumerate(idx):
            letter_min[c] = min(letter_min.get(c, arr.shape[axis]), arr.shape[axis])

    sliced = []
    for arr, idx in zip(arrays, indices):
        slices = []
        for axis, c in enumerate(idx):
            if arr.shape[axis] > letter_min[c]:
                slices.append(slice(None, letter_min[c]))
            else:
                slices.append(slice(None))
        sliced.append(arr[tuple(slices)])

    # Perform einsum
    result_array = math.einsum(",".join(indices) + "->" + out_batch_str + out_core_str, *sliced)

    # Collapse grouped batch dims
    if groups:
        result_ans = ArrayAnsatz(result_array, batch_dims=len(out_batch_str))
        return _collapse_grouped_batch_dims_arrayansatz(result_ans, groups)

    return ArrayAnsatz(result_array, batch_dims=len(out_batch_str))


def _raise_if_any_polyexp_left(records: list[Rec], fock_dims: dict[str, int]):
    """Raise an error if there are any PolyExpAnsatz with core dimensions that are not in fock_dims."""
    for rec in records:
        if isinstance(rec.ans, PolyExpAnsatz):
            missing_dims = set(rec.core_str) - set(fock_dims.keys())
            raise ValueError(
                f"Couldn't convert PolyExpAnsatz {rec.ans!s}: missing fock_dims for {missing_dims}"
            )


def _split_by_case(s: str) -> tuple[str, str]:
    """Split string into (UPPERCASE, lowercase) letters."""
    upper = "".join(c for c in s if c.isupper())
    lower = "".join(c for c in s if c.islower())
    return upper, lower


def _validate_core_dimensions(
    input_strings: list[str], out_core_str: str, fock_dims: dict[str, int]
):
    """Validate that all core dimensions can be handled (contracted, kept, or converted)."""
    all_core_letters = []
    for s in input_strings:
        all_core_letters.extend(c for c in s if c.islower())
    core_counts = Counter(all_core_letters)
    missing_letters = []
    for letter, count in core_counts.items():
        in_output = letter in out_core_str
        will_contract = count > 1
        has_fock_dim = letter in fock_dims

        if not in_output and not will_contract and not has_fock_dim:
            missing_letters.append(letter)

    if missing_letters:
        raise ValueError(
            f"Cannot convert PolyExpAnsatz to Fock: missing fock_dims for {set(missing_letters)}. "
            f"These core letters appear in only one input, are not in the output, "
            f"and have no fock_dims specified."
        )


def _validate_parentheses(char: str, in_group: bool):
    """Validate parentheses in output string."""
    if in_group:
        if char == "(":
            raise ValueError("Nested parentheses not supported")
        if char.islower():
            raise ValueError("Lowercase letters cannot be grouped")
    elif char == ")":
        raise ValueError("Unmatched ')' in output")


def ua_to_linear(path: list[tuple[int, ...]]) -> list[tuple[int, int]]:
    """Convert union assignment path to linear assignment path."""
    sets = [{i} for i in sorted({j for pair in path for j in pair})]
    steps = []
    for a, b in path:
        set1, set2 = [s for s in sets if a in s or b in s]
        i, j = sorted([sets.index(set1), sets.index(set2)])
        sets.pop(j), sets.pop(i)  # must be popped in order
        sets.append(set1 | set2)
        steps.append((i, j))
    return steps


#  ~~~~~~~~~~~~~~~~~~~~
#  Conversion functions
#  ~~~~~~~~~~~~~~~~~~~~


def bargmann_to_fock(func: Callable[..., PolyExpAnsatz]) -> Callable[..., ArrayAnsatz]:
    r"""
    Decorator that wraps a function returning a ``PolyExpAnsatz``
    and returns an ``ArrayAnsatz`` instead using ``to_fock``.

    Args:
        func: A callable that returns a ``PolyExpAnsatz``.

    Returns:
        A wrapped function that returns an ``ArrayAnsatz`` constructed
        from the Bargmann triple (A, b, c) using ``to_fock``.
        The wrapped function has the same signature as the original
        function with an additional optional ``shape`` parameter.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        shape = kwargs.pop("shape", None)
        ansatz = func(*args, **kwargs)
        return to_fock(ansatz, shape=shape)

    return wrapper


def fock_to_bargmann(func: Callable[..., ArrayAnsatz]) -> Callable[..., PolyExpAnsatz]:
    r"""
    Decorator that wraps a function returning an ``ArrayAnsatz``
    and returns a ``PolyExpAnsatz`` instead using ``to_bargmann``.

    Args:
        func: A callable that returns an ``ArrayAnsatz``.

    Returns:
        A wrapped function that returns a ``PolyExpAnsatz`` constructed
        from the ``ArrayAnsatz`` using ``to_bargmann``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ansatz = func(*args, **kwargs)
        return to_bargmann(ansatz)

    return wrapper


def to_fock(
    ansatz: Ansatz, shape: tuple[int, ...], stable: bool = False, preserve_lin_sup: bool = False
) -> ArrayAnsatz:
    r"""
    Converts a PolyExpAnsatz to an ArrayAnsatz.
    If the ansatz is already an ArrayAnsatz, it reduces the shape to the given shape.

    Args:
        ansatz: The ansatz to convert.
        shape: The shape of the ArrayAnsatz.
        stable: Whether to use the stable version of the hermite_renormalized function.
        preserve_lin_sup: If True, do not sum over the linear superposition dimension.

    Returns:
        ArrayAnsatz: The converted ArrayAnsatz.
    """
    if 0 in shape:
        raise ValueError("Fock space dimension is 0.")
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)

    sum_lin_sup = ansatz._lin_sup and not preserve_lin_sup
    batch_dims = ansatz.batch_dims - int(sum_lin_sup)

    if len(shape) == 0:
        array = ansatz.scalar if sum_lin_sup else ansatz.c
    else:
        A, b, c = ansatz.triple
        # TODO: make hermite_renormalized work with num_derived_vars > 0 in sc-97587
        if ansatz.num_derived_vars == 0:
            array = math.hermite_renormalized(
                A,
                b,
                c,
                shape=shape,
                stable=stable,
            )
        else:
            G = math.hermite_renormalized(
                A,
                b,
                math.ones(ansatz.batch_shape, dtype=math.complex128),
                shape=shape + ansatz.shape_derived_vars,
                stable=stable,
            )
            G = math.reshape(G, ansatz.batch_shape + shape + (-1,))
            cs = math.reshape(c, (*ansatz.batch_shape, -1))
            core_str = "".join(
                [chr(i) for i in range(97, 97 + len(G.shape[ansatz.batch_dims :]))],
            )
            array = math.einsum(f"...{core_str},...{core_str[-1]}->...{core_str[:-1]}", G, cs)

    if sum_lin_sup and len(shape) > 0:
        array = math.sum(array, axis=ansatz.batch_dims - 1)

    return ArrayAnsatz(array, batch_dims)


def to_bargmann(ansatz: Ansatz) -> PolyExpAnsatz:
    r"""
    Converts an ArrayAnsatz to a PolyExpAnsatz.
    If the ansatz is already a PolyExpAnsatz, it returns the ansatz unchanged.

    Args:
        ansatz: The ansatz to convert.

    Returns:
        PolyExpAnsatz: The converted PolyExpAnsatz.
    """
    if isinstance(ansatz, PolyExpAnsatz):
        return ansatz
    try:
        A, b, c = ansatz._original_abc_data
    except (AttributeError, TypeError):
        A, b, _ = identity_Abc(ansatz.core_dims)
        A = math.broadcast_to(A, (*ansatz.batch_shape, 2 * ansatz.core_dims, 2 * ansatz.core_dims))
        b = math.broadcast_to(b, (*ansatz.batch_shape, 2 * ansatz.core_dims))
        c = ansatz.array
    return PolyExpAnsatz(A, b, c)


#  ~~~~~~~~~
#  mm_einsum
#  ~~~~~~~~~


def mm_einsum(
    equation: str,
    *operands: Ansatz | ArrayLike,
    fock_dims: dict[str, int] | None = None,
    contraction_path: list[tuple[int, ...]] | None = None,
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> ArrayLike | PolyExpAnsatz:
    r"""
    Performs Einstein summation over ansatzes with explicit batch and core dimension labeling.
    Batch dimensions are explicitly labeled with UPPERCASE letters, while core dimensions
    use lowercase letters. All batch dimensions (including an eventual linear superposition axis)
    must be explicitly labeled.

    Args:
        equation: Einstein notation string with explicit output indices.
        *operands: Operands in input order (PolyExpAnsatz, ArrayAnsatz, or raw arrays).
        fock_dims: Mapping from core letters to Fock sizes. Required if converting PolyExpAnsatz to Fock.
        contraction_path: List of contraction steps over operand IDs.
        path_type: Path interpretation method ("LA", "SSA", or "UA"). Default is "LA".

    Returns:
        Final ArrayAnsatz or PolyExpAnsatz depending on conversion requirements.

    Raises:
        ValueError: If the equation is invalid or the operands are not compatible.
    """
    if not fock_dims:
        fock_dims = {}

    if equation.count("->") != 1:
        raise ValueError("Einsum string must contain exactly one '->' with explicit output indices")

    lhs, output_string = equation.split("->")
    inputs = [s.strip() for s in lhs.split(",") if s.strip()]
    out_batch_str, out_core_str, groups = _parse_output_string(output_string)

    if len(inputs) != len(operands):
        raise ValueError("Number of inputs must match number of operands")

    _validate_core_dimensions(inputs, out_core_str, fock_dims)

    records = _build_records_from_operands(inputs, operands)
    records = _convert_to_fock(records, fock_dims, out_batch_str, raise_if_missing_dims=False)
    records = _phase_1_trace_out(records, out_core_str)
    steps = _normalize_path(contraction_path, path_type)
    records = _phase_2_contract_bargmann(records, steps)
    records = _convert_to_fock(records, fock_dims, out_batch_str, raise_if_missing_dims=True)
    records = _do_leftover_polyexp_outer_product(records, fock_dims)
    if len(records) == 1 and isinstance(records[0].ans, PolyExpAnsatz):
        return _early_return_single_polyexp(records[0], out_batch_str, out_core_str, groups)
    _raise_if_any_polyexp_left(records, fock_dims)
    return _phase_3_final_einsum(records, out_batch_str, out_core_str, groups)
