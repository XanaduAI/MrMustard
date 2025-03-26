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

from numpy.typing import ArrayLike

from mrmustard import math
from mrmustard.utils.typing import ComplexMatrix, ComplexVector, ComplexTensor

#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def generate_batch_str(batch_shape: tuple[int, ...], offset: int = 0) -> str:
    r"""
    Generate a string of characters to represent the batch dimensions.

    Args:
        batch_shape: The shape of the batch dimensions.
        offset: The offset of the characters.

    Returns:
        A string of characters to represent the batch dimensions.
    """
    return "".join([chr(i) for i in range(97 + offset, 97 + offset + len(batch_shape))])


def outer_product_batch_str(*ndims: int) -> str:
    r"""
    Creates the einsum string for the outer product of the given tuple of dimensions.
    E.g. for (2,1,3) it returns ab,c,def->abcdef
    """
    strs = []
    offset = 0
    for ndim in ndims:
        strs.append("".join([chr(97 + i + offset) for i in range(ndim)]))
        offset += ndim
    return ",".join(strs) + "->" + "".join(strs)


def reshape_args_to_batch_string(
    args: list[ArrayLike], batch_string: str
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
            f"Number of input specifications ({len(input_specs)}) does not match number of arguments ({len(args)})"
        )

    args = [math.astensor(arg) for arg in args]

    # Determine the size of each dimension in the output
    dim_sizes = {}
    for arg, spec in zip(args, input_specs):
        for dim, label in zip(arg.shape, spec):
            if label in dim_sizes and dim_sizes[label] != dim:
                raise ValueError(
                    f"Dimension {label} has inconsistent sizes: got {dim_sizes[label]} and {dim}"
                )
            dim_sizes[label] = dim

    reshaped = []
    for arg, spec in zip(args, input_specs):
        new_shape = [dim_sizes[label] if label in spec else 1 for label in output_spec]
        reshaped.append(math.reshape(arg, new_shape))
    return reshaped


def verify_batch_triple(
    A: ComplexMatrix | None, b: ComplexVector | None, c: ComplexTensor | None
) -> None:
    r"""
    Verify that the batch dimensions of the (A, b, c) triple are consistent.

    Args:
        A: The matrix of the quadratic form.
        b: The vector of the linear form.
        c: The scalar of the quadratic form.

    Raises:
        ValueError: If the batch dimensions of the (A, b, c) triple are inconsistent.
    """
    if A is None and b is None and c is None:
        return
    batch = A.shape[:-2]
    batch_dim = len(batch)

    if batch != b.shape[:batch_dim] or (len(c.shape) != 0 and batch != c.shape[:batch_dim]):
        raise ValueError(
            f"Batch dimensions of the first triple ({batch}, {b.shape[:batch_dim]}, {c.shape[:batch_dim]}) are inconsistent."
        )


def zip_batch_strings(*ndims: int) -> str:
    r"""
    Creates a batch string for zipping over the batch dimensions.
    """
    if len(set(ndims)) != 1:
        raise ValueError(f"Arrays must have the same number of batch dimensions, got {ndims}")
    str_ = "".join([chr(97 + i) for i in range(ndims[0])])
    return ",".join([str_] * len(ndims)) + "->" + str_
