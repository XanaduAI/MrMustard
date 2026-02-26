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

"""
This module contains the utility functions used by the classes in ``mrmustard.physics``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import EllipsisType

import numpy as np
from numpy.typing import ArrayLike

from mrmustard import math, settings
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

__all__ = [
    "IndexerType",
    "batch_indexer_info",
    "generate_batch_str",
    "join_Abc",
    "lin_sup_batch_str",
    "outer_product_batch_str",
    "random_Abc",
    "reshape_args_to_batch_string",
    "verify_triple",
    "zip_batch_strings",
]

#  ~~~~~~~~~~~~~~~~
#  Helper functions
#  ~~~~~~~~~~~~~~~~

_IndexerTypes = int | slice | None | EllipsisType
IndexerType = _IndexerTypes | Sequence[_IndexerTypes]


def _join_Ab(
    A1: ComplexMatrix,
    b1: ComplexVector,
    A2: ComplexMatrix,
    b2: ComplexVector,
    m1: int,
    m2: int,
) -> tuple[ComplexMatrix, ComplexVector]:
    r"""
    Joins two (A, b) pairs by block-diagonal concatenation and reordering.

    This helper function creates a block-diagonal A matrix from A1 and A2, concatenates
    the b vectors, and reorders rows/columns to group core variables first, then derived variables.

    Args:
        A1: First A matrix with shape ``(*batch, n1, n1)``
        b1: First b vector with shape ``(*batch, n1)``
        A2: Second A matrix with shape ``(*batch, n2, n2)``
        b2: Second b vector with shape ``(*batch, n2)``
        m1: Number of derived variables in first triple
        m2: Number of derived variables in second triple

    Returns:
        Tuple of (A, b) with joined and reordered matrices/vectors
    """
    output_batch_shape = A1.shape[:-2]
    nA1 = A1.shape[-1]
    nA2 = A2.shape[-1]
    n1 = nA1 - m1
    n2 = nA2 - m2

    # Block-diagonal concatenation of A matrices: [[A1, 0], [0, A2]]
    zeros_top = math.zeros((*output_batch_shape, nA1, nA2), dtype=math.complex128)
    zeros_bottom = math.zeros((*output_batch_shape, nA2, nA1), dtype=math.complex128)
    A_top = math.concat([A1, zeros_top], axis=-1)
    A_bottom = math.concat([zeros_bottom, A2], axis=-1)
    A_joined = math.concat([A_top, A_bottom], axis=-2)

    # Reorder rows to group core and derived variables: [core1, core2, derived1, derived2]
    row_indices = (
        list(range(n1))  # core1
        + list(range(nA1, nA1 + n2))  # core2
        + list(range(n1, nA1))  # derived1
        + list(range(nA1 + n2, nA1 + nA2))  # derived2
    )
    A = math.gather(A_joined, row_indices, axis=-2)
    A = math.gather(A, row_indices, axis=-1)

    # Concatenate b vectors and reorder
    b_joined = math.concat([b1, b2], axis=-1)
    b = math.gather(b_joined, row_indices, axis=-1)

    return A, b


def _join_c(
    c1: ComplexTensor,
    c2: ComplexTensor,
    batch_dim1: int,
    batch_dim2: int,
    return_log: bool = False,
) -> ComplexTensor:
    r"""
    Joins two c tensors by computing their outer product.

    This helper function computes the outer product of c1 and c2 in log space for
    numerical stability, then reshapes to the final output shape.

    Args:
        c1: First c tensor with shape ``(*batch1, *poly_shape1)``
        c2: Second c tensor with shape ``(*batch2, *poly_shape2)``
        batch_dim1: Number of batch dimensions in c1
        batch_dim2: Number of batch dimensions in c2
        return_log: If ``True``, returns ``log(c)`` instead of ``c``

    Returns:
        Joined c tensor with shape ``(*batch_out, *poly_shape1, *poly_shape2)``
    """
    batch1 = c1.shape[:batch_dim1]
    batch2 = c2.shape[:batch_dim2]
    poly_shape1 = c1.shape[batch_dim1:] if c1.ndim > batch_dim1 else ()
    poly_shape2 = c2.shape[batch_dim2:] if c2.ndim > batch_dim2 else ()

    # Determine output batch shape using standard broadcasting
    output_batch_shape = np.broadcast_shapes(batch1, batch2)

    # Flatten polynomial dimensions for easier manipulation
    c1_flat_size = int(np.prod(poly_shape1)) if poly_shape1 else 1
    c2_flat_size = int(np.prod(poly_shape2)) if poly_shape2 else 1
    c1_flat = math.reshape(c1, (*batch1, c1_flat_size))
    c2_flat = math.reshape(c2, (*batch2, c2_flat_size))

    # Broadcast to output batch shape
    c1_bc = math.broadcast_to(c1_flat, (*output_batch_shape, c1_flat_size), dtype=math.complex128)
    c2_bc = math.broadcast_to(c2_flat, (*output_batch_shape, c2_flat_size), dtype=math.complex128)

    # Compute outer product in log space for numerical stability
    c1_exp = c1_bc[..., :, None]  # Add axis for outer product
    c2_exp = c2_bc[..., None, :]  # Add axis for outer product
    log_c = math.log(math.cast(c1_exp, "complex128")) + math.log(math.cast(c2_exp, "complex128"))

    # Reshape to final shape: batch + poly_shape1 + poly_shape2
    final_c_shape = output_batch_shape + poly_shape1 + poly_shape2
    log_c = math.reshape(log_c, final_c_shape)

    if return_log:
        return log_c
    return math.exp(log_c)


#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def batch_indexer_info(
    indexer: IndexerType, batch_dims: int
) -> tuple[tuple[int | slice, ...], int, int]:
    r"""
    Process the indexer passed to a __getitem__ call and return
    the batch index and the number of removed and inserted batch dimensions.

    Args:
        indexer: The indexer passed to a __getitem__ call.
        batch_dims: The number of batch dimensions of the ansatz (including eventual linear superposition dimension)

    Returns:
        A tuple containing the final index, the number of removed batch dimensions, and the number of inserted batch dimensions.

    Raises:
        TypeError: If the indexer contains any types other than int, slice, None, or Ellipsis.
        IndexError: If the indexer contains more than one Ellipsis.
        IndexError: If the indexer consumes more batch dimensions than the ansatz has.
        IndexError: If the indexer contains an Ellipsis in the middle of the indexer.
    """
    indexer = (indexer,) if not isinstance(indexer, tuple) else indexer

    def is_consuming(x: _IndexerTypes) -> bool:
        return isinstance(x, int | slice)

    if any((x is not None) and (x is not Ellipsis) and not is_consuming(x) for x in indexer):
        raise TypeError("Only int, slice, None, and Ellipsis are supported for batch indexing.")
    if indexer.count(Ellipsis) > 1:
        raise IndexError("Only a single ellipsis is allowed.")
    if indexer.count(Ellipsis) == 1 and indexer[0] != Ellipsis and indexer[-1] != Ellipsis:
        raise IndexError("Ellipsis must be at the beginning or end of the indexer.")

    explicit_consuming = sum(1 for x in indexer if is_consuming(x))
    if explicit_consuming > batch_dims:
        raise IndexError("Too many indices.")

    expanded: list[_IndexerTypes] = []
    for x in indexer:
        if x is Ellipsis:
            expanded.extend([slice(None)] * (batch_dims - explicit_consuming))
        else:
            expanded.append(x)

    return _batch_indexer_info_return_values(expanded)


def _batch_indexer_info_return_values(
    expanded: list[_IndexerTypes],
) -> tuple[list[_IndexerTypes], int, int]:
    "Helper function to return the final index, the number of removed batch dimensions, and the number of inserted batch dimensions."
    final_index: list[_IndexerTypes] = []
    removed_by_int = 0
    inserted_by_none = 0
    for x in expanded:
        if x is None:
            final_index.append(None)
            inserted_by_none += 1
        else:
            final_index.append(x)
            if isinstance(x, int):
                removed_by_int += 1

    final_index.append(Ellipsis)  # preserve core axes

    return tuple(final_index), removed_by_int, inserted_by_none


def generate_batch_str(batch_dim: int, offset: int = 0) -> str:
    r"""
    Generate a string of characters to represent the batch dimensions.

    Args:
        batch_shape: The shape of the batch dimensions.
        offset: The offset of the characters.

    Returns:
        A string of characters to represent the batch dimensions.
    """
    return "".join([chr(97 + i) for i in range(offset, offset + batch_dim)])


def join_Abc(
    A1: ComplexMatrix,
    b1: ComplexVector,
    c1: ComplexTensor,
    A2: ComplexMatrix,
    b2: ComplexVector,
    c2: ComplexTensor,
    return_log_c: bool = False,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    Joins two ``(A,b,c)`` triples into a single ``(A,b,c)`` by block-diagonal concatenation of the
    A matrices and concatenation of the b vectors. The c tensor is computed as the outer product of the
    c tensors of the two input triples.

    This function combines two Bargmann representations by placing them in a block-diagonal
    structure for the A matrices, concatenating the b vectors, and computing the outer product
    of the c arrays. Batch dimensions are automatically broadcast using standard NumPy/JAX
    broadcasting semantics.

    The function handles "derived variables" (polynomial indices in c) by reordering the
    rows and columns of A and b to keep core variables first, followed by derived variables.

    Args:
        A1: First A matrix with shape ``(*batch1, n1, n1)``
        b1: First b vector with shape ``(*batch1, n1)``
        c1: First c array with shape ``(*batch1, *poly_shape1)``
        A2: Second A matrix with shape ``(*batch2, n2, n2)``
        b2: Second b vector with shape ``(*batch2, n2)``
        c2: Second c array with shape ``(*batch2, *poly_shape2)``
        return_log_c: If ``True``, returns ``log(c)`` instead of ``c`` for numerical stability

    Returns:
        Joined ``(A, b, c)`` triple with broadcasted batch dimensions and concatenated core dimensions:
        - A: shape ``(*batch_out, n1+n2, n1+n2)``
        - b: shape ``(*batch_out, n1+n2)``
        - c: shape ``(*batch_out, *poly_shape1, *poly_shape2)``

    Examples:
        >>> # Non-batched join
        >>> A1 = math.astensor([[1, 2], [3, 4]])  # shape (2, 2)
        >>> b1 = math.astensor([5, 6])  # shape (2,)
        >>> c1 = math.astensor(7)  # shape ()
        >>> A2 = math.astensor([[8, 9], [10, 11]])  # shape (2, 2)
        >>> b2 = math.astensor([12, 13])  # shape (2,)
        >>> c2 = math.astensor(10)  # shape ()
        >>> A, b, c = join_Abc(A1, b1, c1, A2, b2, c2)
        >>> A.shape  # (4, 4) - block diagonal of two 2x2 matrices
        (4, 4)
        >>> b.shape  # (4,) - concatenation of two length-2 vectors
        (4,)
        >>> c.shape  # () - product of two scalars
        ()

        >>> # Batched join with automatic broadcasting
        >>> A1 = math.astensor([[[1, 2], [3, 4]]])  # shape (1, 2, 2)
        >>> b1 = math.astensor([[5, 6]])  # shape (1, 2)
        >>> c1 = math.astensor([7])  # shape (1,)
        >>> A2 = math.astensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])  # shape (2, 2, 2)
        >>> b2 = math.astensor([[12, 13], [14, 15]])  # shape (2, 2)
        >>> c2 = math.astensor([10, 100])  # shape (2,)
        >>> A, b, c = join_Abc(A1, b1, c1, A2, b2, c2)
        >>> A.shape  # (2, 4, 4) - broadcasts (1,) and (2,) to (2,)
        (2, 4, 4)
        >>> b.shape  # (2, 4)
        (2, 4)
        >>> c.shape  # (2,)
        (2,)
    """
    # Verify consistency of batch dimensions within each triple
    verify_triple(A1, b1, c1)
    verify_triple(A2, b2, c2)

    # Extract batch dimensions
    batch1 = A1.shape[:-2]
    batch2 = A2.shape[:-2]
    batch_dim1 = len(batch1)
    batch_dim2 = len(batch2)

    # Determine polynomial shapes (derived variables from c shape)
    poly_shape1 = c1.shape[batch_dim1:] if c1.ndim > batch_dim1 else ()
    poly_shape2 = c2.shape[batch_dim2:] if c2.ndim > batch_dim2 else ()

    # Number of derived variables (from polynomial dimensions of c)
    m1 = len(poly_shape1)
    m2 = len(poly_shape2)

    # Get dimensions for broadcasting
    nA1, mA1 = A1.shape[-2:]
    nA2, mA2 = A2.shape[-2:]

    # Determine output batch shape using standard broadcasting
    output_batch_shape = np.broadcast_shapes(batch1, batch2)

    # Broadcast A and b to common batch shape before joining
    A1_bc = math.broadcast_to(A1, (*output_batch_shape, nA1, mA1), dtype=math.complex128)
    A2_bc = math.broadcast_to(A2, (*output_batch_shape, nA2, mA2), dtype=math.complex128)
    b1_bc = math.broadcast_to(b1, (*output_batch_shape, nA1), dtype=math.complex128)
    b2_bc = math.broadcast_to(b2, (*output_batch_shape, nA2), dtype=math.complex128)

    # Join A and b using helper function
    A, b = _join_Ab(A1_bc, b1_bc, A2_bc, b2_bc, m1, m2)

    # Join c using helper function
    c = _join_c(c1, c2, batch_dim1, batch_dim2, return_log=return_log_c)

    return A, b, c


def lin_sup_batch_str(batch_str: str) -> str:
    r"""
    Given a batch string, appends the linear superposition batch dimension to the end.

    Args:
        batch_str: The batch string to append the linear superposition batch dimension to.

    Returns:
        The batch string with the linear superposition batch dimension appended to the end.
    """
    input_str, output_str = batch_str.split("->")
    inputs = input_str.split(",")
    max_char = max(ord(i) for i in batch_str)
    lin_sups = [chr(max_char + offset) for offset in range(1, len(inputs) + 1)]
    new_input = ",".join([ipt + lin_sup for ipt, lin_sup in zip(inputs, lin_sups)])
    new_output = output_str + "".join(lin_sups)
    return f"{new_input}->{new_output}"


def outer_product_batch_str(*batch_dims: int, lin_sup: tuple[int, ...] | None = None) -> str:
    r"""
    Creates the einsum string for the outer product of the given tuple of dimensions.
    E.g. for (2,1,3) it returns ab,c,def->abcdef.
    If lin_sup is provided, the linear superposition dimensions are moved to the end.
    E.g. for (2,1,3) and lin_sup=(0,1) it returns ab,c,def->adefbc, as b and c are the linear superposition dimensions
    of the 0th and 1st tensors.
    """
    strs = []
    offset = 0
    for batch_dim in batch_dims:
        strs.append(generate_batch_str(batch_dim, offset))
        offset += batch_dim

    orig_strs = strs.copy()  # keep original for input part of einsum string

    if lin_sup is not None:
        lin_sup_chars = []
        for idx in lin_sup:
            lin_sup_chars.append(strs[idx][-1])
            strs[idx] = strs[idx][:-1]
        output = "".join(strs) + "".join(lin_sup_chars)
    else:
        output = "".join(strs)

    return ",".join(orig_strs) + "->" + output


def random_Abc(
    core_vars: int,
    batch: tuple[int, ...] = (),
    derived: tuple[int, ...] = (),
    seed: int | None = None,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    Generate a random ``(A, b, c)`` triple for testing purposes.

    This function creates random complex-valued (A, b, c) triples with specified
    batch dimensions, core variables, and derived variables (polynomial dimensions).
    Useful for testing functions that operate on Fock-Bargmann triples.

    Args:
        core_vars: Number of core variables (determines the size of A and b before derived variables)
        batch: Batch shape for the triple
        derived: Derived variable shape (polynomial dimensions in c)
        seed: Random seed

    Returns:
        Tuple of (A, b, c) where:
        - A: Complex symmetric matrix of shape ``(*batch, n+m, n+m)``
        - b: Complex vector of shape ``(*batch, n+m)``
        - c: Complex tensor of shape ``(*batch, *derived)``

        where n = core_vars and m = len(derived)

    Examples:
        >>> # Simple non-batched triple with 3 core variables
        >>> A, b, c = random_Abc(3)
        >>> A.shape, b.shape, c.shape
        ((3, 3), (3,), ())

        >>> # Batched triple with derived variables
        >>> A, b, c = random_Abc(2, batch=(5,), derived=(4,))
        >>> A.shape, b.shape, c.shape
        ((5, 2, 2), (5, 2), (5, 4))
    """
    m = len(derived)
    n = core_vars
    min_magnitude = 1e-9
    max_magnitude = 1
    rng = settings.get_rng(seed)

    # Complex symmetric matrix A
    A = rng.uniform(min_magnitude, max_magnitude, (*batch, n + m, n + m)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch, n + m, n + m),
    )
    A = 0.5 * (A + np.swapaxes(A, -2, -1))  # make it symmetric

    # Complex vector b
    b = rng.uniform(min_magnitude, max_magnitude, (*batch, n + m)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch, n + m),
    )

    # Complex scalar/tensor c
    c = rng.uniform(min_magnitude, max_magnitude, (*batch, *derived)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch, *derived),
    )

    return A, b, c


def reshape_args_to_batch_string(
    args: list[ArrayLike],
    batch_string: str,
) -> tuple[list[ArrayLike], tuple[int, ...]]:
    r"""
    Reshapes arguments to match the batch string by inserting singleton dimensions where needed
    so that they are broadcastable.
    E.g. given two arrays of shape (2,7) and (3,7) and string ab,cb->abc, it reshapes them to
    shape (2,7,1) and (1,7,3).
    """
    # Parse the batch string
    input_specs, output_spec = batch_string.split("->")
    input_specs = input_specs.split(",")
    if len(input_specs) != len(args):
        raise ValueError(
            f"Number of input specifications ({len(input_specs)}) does not match number of arguments ({len(args)})",
        )

    args = [math.astensor(arg) for arg in args]

    # Determine the size of each dimension in the output
    dim_sizes = {}
    for arg, spec in zip(args, input_specs):
        for dim, label in zip(arg.shape, spec):
            if label in dim_sizes and dim_sizes[label] != dim:
                raise ValueError(
                    f"Dimension {label} has inconsistent sizes: got {dim_sizes[label]} and {dim}",
                )
            dim_sizes[label] = dim

    reshaped = []
    for arg, spec in zip(args, input_specs):
        new_shape = [dim_sizes[label] if label in spec else 1 for label in output_spec]
        reshaped.append(math.reshape(arg, new_shape))
    return reshaped


def verify_triple(
    A: ComplexMatrix,
    b: ComplexVector,
    c: ComplexTensor,
) -> None:
    r"""
    Verify that both the batch and core dimensions of the ``(A, b, c)`` triple are consistent.

    ``A`` and ``b`` must have core dimensions ``(N, N)`` and ``(N,)``, respectively.

    Args:
        A: The matrix of the quadratic form.
        b: The vector of the linear form.
        c: The scalar of the quadratic form.

    Raises:
        ValueError: If any (batch / core) dimensions of the ``(A, b, c)`` triple are inconsistent.
    """
    batch = A.shape[:-2]
    batch_dim = len(batch)

    if batch != b.shape[:batch_dim] or (len(c.shape) != 0 and batch != c.shape[:batch_dim]):
        raise ValueError(
            f"Batch dimensions of the triple ({batch}, {b.shape[:batch_dim]}, {c.shape[:batch_dim]}) are inconsistent.",
        )

    A_core_shape = A.shape[batch_dim:]
    b_core_shape = b.shape[batch_dim:]

    if A_core_shape[0] != A_core_shape[1]:
        raise ValueError(f"Core dimensions of A are inconsistent: {A_core_shape}.")

    if A_core_shape[0] != b_core_shape[0]:
        raise ValueError(
            f"Core dimensions of A and b are inconsistent: {A_core_shape} and {b_core_shape}, "
            "respectively."
        )


def zip_batch_strings(*batch_dims: int) -> str:
    r"""
    Creates a batch string for zipping over the batch dimensions.
    """
    input_str = ",".join([generate_batch_str(batch_dim) for batch_dim in batch_dims])
    return input_str + "->" + generate_batch_str(max(batch_dims))
