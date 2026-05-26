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

"""This module contains the array ansatz."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
from IPython.display import display
from numpy.typing import ArrayLike

from mrmustard import math, settings, widgets
from mrmustard.physics.utils import batch_indexer_info
from mrmustard.utils.typing import Scalar, Tensor

from .base import Ansatz

__all__ = ["ArrayAnsatz"]


class ArrayAnsatz(Ansatz):
    r"""The ansatz of the Fock-Bargmann representation.

    Represents the ansatz as a multidimensional array.

    Args:
        array: A (potentially) batched array.
        batch_dims: The number of batch dimensions.

    >>> import numpy as np
    >>> from mrmustard.physics.ansatz import ArrayAnsatz
    >>> array = np.random.random((2, 4, 5))
    >>> ansatz = ArrayAnsatz(array)
    """

    def __init__(self, array: Tensor, batch_dims: int = 0):
        super().__init__()
        self._array = array
        self._batch_dims = batch_dims
        self._batch_shape = tuple(self._array.shape[:batch_dims])

    @property
    def array(self) -> Tensor:
        r"""The array of this ansatz."""
        return self._array

    @array.setter
    def array(self, value: Tensor):
        self._array = math.astensor(value)
        self._batch_shape = tuple(value.shape[: self.batch_dims])

    @property
    def batch_dims(self) -> int:
        return self._batch_dims

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def batch_size(self):
        return int(np.prod(self.batch_shape)) if self.batch_shape else 0

    @property
    def conj(self) -> ArrayAnsatz:
        return ArrayAnsatz(math.conj(self.array), self.batch_dims)

    @property
    def core_dims(self) -> int:
        r"""The number of core dimensions of this ansatz."""
        return len(self.core_shape)

    @property
    def core_shape(self) -> tuple[int, ...]:
        r"""The core dimensions of this ansatz."""
        return self.array.shape[self.batch_dims :]

    @property
    def data(self) -> Tensor:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - self.batch_dims

    @property
    def scalar(self) -> Scalar:
        r"""The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        """
        return self.array[(...,) + (0,) * self.core_dims]

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> ArrayAnsatz:
        return cls(**data)

    def concat(self, other: Ansatz, axis: int = 0) -> ArrayAnsatz:
        r"""Concatenates two ArrayAnsatz objects along the specified batch axis.

        All batch axes except the concatenation axis must agree in size.
        Core shapes must also match.

        Args:
            other: The other ArrayAnsatz to concatenate with.
            axis: The batch axis along which to concatenate (default 0).

        Returns:
            A new ArrayAnsatz with the concatenated batch dimensions.

        Raises:
            TypeError: If ``other`` is not an ``ArrayAnsatz``.
            ValueError: If batch dimensions don't match (except at axis) or if core shapes don't match.

        >>> from mrmustard.physics.ansatz import ArrayAnsatz
        >>> import numpy as np
        >>> array1 = np.random.random((2, 4, 5))
        >>> ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        >>> array2 = np.random.random((3, 4, 5))
        >>> ansatz2 = ArrayAnsatz(array2, batch_dims=1)
        >>> concatenated = ansatz1.concat(ansatz2, axis=0)
        >>> assert concatenated.batch_shape == (5,)
        >>> assert concatenated.core_shape == (4, 5)
        """
        if not isinstance(other, ArrayAnsatz):
            raise TypeError(f"Cannot concatenate with ansatz of type {type(other)}!")

        # Check that both have the same number of batch dimensions
        if self.batch_dims != other.batch_dims:
            raise ValueError(
                f"Cannot concatenate ansatze with different number of batch dimensions. "
                f"Got {self.batch_dims} and {other.batch_dims}"
            )

        # Check that core shapes match
        if any(np.asarray(self.core_shape) != np.asarray(other.core_shape)):
            raise ValueError(
                f"Cannot concatenate ansatze with different core_shape. "
                f"Got {self.core_shape} and {other.core_shape}"
            )

        # Normalize negative axis
        if axis < 0:
            axis = self.batch_dims + axis

        if axis < 0 or axis >= self.batch_dims:
            raise ValueError(f"axis {axis} is out of range for batch_dims {self.batch_dims}")

        # Check that all other batch axes (except the concatenation axis) match
        self_batch_shape = np.asarray(self.batch_shape)
        other_batch_shape = np.asarray(other.batch_shape)
        for i in range(self.batch_dims):
            if i != axis and self_batch_shape[i] != other_batch_shape[i]:
                raise ValueError(
                    f"Batch shapes must match except at axis {axis}. "
                    f"Got {self.batch_shape} and {other.batch_shape}"
                )

        # Concatenate the arrays along the batch axis
        array_concat = math.concat([self.array, other.array], axis=axis)

        return ArrayAnsatz(array_concat, batch_dims=self.batch_dims)

    def contract(
        self,
        other: Ansatz,
        idxs: tuple[Sequence[int], Sequence[int]],
    ) -> ArrayAnsatz:
        r"""Contract along specified core axes, broadcasting batch dimensions.

        Args:
            other: The other ArrayAnsatz to contract with.
            idxs: Tuple ``(idx_self, idx_other)`` of core-axis indices to contract
                (0-based relative to core dims). Negative indices allowed.

        Returns:
            The contracted ArrayAnsatz, with kept core axes ordered as
            ``[self non-contracted] + [other non-contracted]``. Batch dims are
            broadcast and preserved.

        Raises:
            TypeError: If ``other`` is not an ``ArrayAnsatz``.
            ValueError: If index sequences have incorrect length, duplicates, or out-of-range indices.

        >>> from mrmustard.physics.ansatz import ArrayAnsatz
        >>> from mrmustard import math
        >>> array1 = math.arange(20).reshape((1, 4, 5))
        >>> array2 = math.arange(72).reshape((3, 4, 6))
        >>> ansatz1 = ArrayAnsatz(array1, batch_dims=1)
        >>> ansatz2 = ArrayAnsatz(array2, batch_dims=1)
        >>> # broadcast 1 and 3 while contracting 4 and 4, leaving 3 (batch), 5 and 6:
        >>> contracted = ansatz1.contract(ansatz2, ([0], [0]))
        >>> assert contracted.array.shape == (3, 5, 6)
        """
        if not isinstance(other, ArrayAnsatz):
            raise TypeError(f"Cannot contract with ansatz of type {type(other)}!")

        idxs_self, idxs_other = idxs

        if len(idxs_self) != len(idxs_other):
            raise ValueError(
                "idxs must be sequences of equal length, "
                f"got {len(idxs_self)} and {len(idxs_other)}."
            )

        def normalize(indices: Sequence[int], core_dims: int) -> list[int]:
            normalized = [i + core_dims if i < 0 else i for i in indices]
            if any(i < 0 or i >= core_dims for i in normalized):
                raise ValueError(
                    f"At least one core index is out of range for core dims {core_dims}."
                )
            if len(set(normalized)) != len(normalized):
                raise ValueError("Repeated indices in contraction list are not allowed.")
            return normalized

        idxs_self = normalize(idxs_self, self.core_dims)
        idxs_other = normalize(idxs_other, other.core_dims)

        # Broadcast batch dims and optionally truncate mismatched contracted cores
        bshape = np.broadcast_shapes(self.batch_shape, other.batch_shape)
        a = math.broadcast_to(self.array, bshape + self.core_shape)
        b = math.broadcast_to(other.array, bshape + other.core_shape)
        if idxs_self:
            sa = [slice(None)] * a.ndim
            sb = [slice(None)] * b.ndim
            for i_s, i_o in zip(idxs_self, idxs_other):
                da = a.shape[len(bshape) + i_s]
                db = b.shape[len(bshape) + i_o]
                if da != db:
                    m = int(min(da, db))
                    sa[len(bshape) + i_s] = slice(0, m)
                    sb[len(bshape) + i_o] = slice(0, m)
            a, b = a[tuple(sa)], b[tuple(sb)]

        # Build einsum subscripts for contraction, using ellipsis for batch dims.
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        k = len(idxs_self)  # number of contracted indices
        pair = list(letters[:k])  # labels for contracted indices

        # Indices to keep (uncontracted) for self and other
        keep_self = [i for i in range(self.core_dims) if i not in idxs_self]
        keep_other = [j for j in range(other.core_dims) if j not in idxs_other]

        # Assign unique labels to kept indices for self and other
        self_keep_labels = list(letters[k : k + len(keep_self)])
        other_keep_labels = list(letters[k + len(keep_self) : k + len(keep_self) + len(keep_other)])

        # Map core indices to their einsum labels
        s_map = dict(zip(idxs_self, pair)) | dict(zip(keep_self, self_keep_labels))
        o_map = dict(zip(idxs_other, pair)) | dict(zip(keep_other, other_keep_labels))

        # Build einsum input and output strings for self, other, and result
        s_seq = "".join(s_map[i] for i in range(self.core_dims))
        o_seq = "".join(o_map[j] for j in range(other.core_dims))
        out_seq = "".join(self_keep_labels + other_keep_labels)
        subs = f"...{s_seq},...{o_seq}->...{out_seq}"

        # Perform the contraction using einsum
        out = math.einsum(subs, a, b)
        return ArrayAnsatz(out, batch_dims=len(bshape))

    def display(self):
        if widgets.IN_INTERACTIVE_SHELL or (w := widgets.fock(self)) is None:
            print(repr(self))
            return
        display(w)

    def reduce(self, shape: Sequence[int]) -> ArrayAnsatz:
        r"""Returns a new ``ArrayAnsatz`` with a sliced core shape.

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.

        >>> from mrmustard import math
        >>> from mrmustard.physics.ansatz import ArrayAnsatz
        >>> array1 = math.arange(27).reshape((3, 3, 3))
        >>> fock1 = ArrayAnsatz(array1)
        >>> fock2 = fock1.reduce((3, 3, 3))
        >>> assert fock1 == fock2
        >>> fock3 = fock1.reduce((2, 2, 2))
        >>> array3 = math.astensor([[[0, 1], [3, 4]], [[9, 10], [12, 13]]])
        >>> assert fock3 == ArrayAnsatz(array3)
        >>> fock4 = fock1.reduce((1, 3, 1))
        >>> array4 = math.astensor([[[0], [3], [6]]])
        >>> assert fock4 == ArrayAnsatz(array4)
        """
        if shape == self.core_shape:
            return self
        if len(shape) != self.core_dims:
            raise ValueError(f"Expected shape of length {self.core_dims}, got {len(shape)}.")

        if any(s > t for s, t in zip(shape, self.core_shape)):
            warn(
                "The fock array is being padded with zeros. Is this really necessary?",
                stacklevel=1,
            )
            padded = math.pad(
                self.array,
                [(0, 0)] * self.batch_dims + [(0, s - t) for s, t in zip(shape, self.core_shape)],
            )
            return ArrayAnsatz(padded, self.batch_dims)

        return ArrayAnsatz(self.array[(..., *tuple(slice(0, s) for s in shape))], self.batch_dims)

    def reorder(self, order: tuple[int, ...] | list[int]) -> ArrayAnsatz:
        order = list(range(self.batch_dims)) + [i + self.batch_dims for i in order]
        return ArrayAnsatz(math.transpose(self.array, order), self.batch_dims)

    def reorder_batch(self, order: Sequence[int]):
        if len(order) != self.batch_dims:
            raise ValueError(
                f"order must have length {self.batch_dims} (number of batch dimensions), got {len(order)}",
            )

        core_dims_indices = range(self.batch_dims, self.batch_dims + self.core_dims)
        new_array = math.transpose(self.array, list(order) + list(core_dims_indices))
        return ArrayAnsatz(new_array, self.batch_dims)

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data, "batch_dims": self.batch_dims}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> ArrayAnsatz:
        if len(idx_z) != len(idx_zconj) or not set(idx_z).isdisjoint(idx_zconj):
            raise ValueError("The idxs must be of equal length and disjoint.")
        order = (
            list(range(self.batch_dims))
            + [self.batch_dims + i for i in range(self.core_dims) if i not in idx_z + idx_zconj]
            + [self.batch_dims + i for i in idx_z]
            + [self.batch_dims + i for i in idx_zconj]
        )
        new_array = math.transpose(self.array, order)

        # truncate new_array if the z and zconj dimensions differ
        d = len(idx_z)
        z_dims = new_array.shape[-2 * d : -d]
        zconj_dims = new_array.shape[-d:]
        if z_dims != zconj_dims:
            slices = [slice(None)] * len(new_array.shape)
            for i, (z_dim, zconj_dim) in enumerate(zip(z_dims, zconj_dims)):
                if z_dim != zconj_dim:
                    min_dim = min(z_dim, zconj_dim)
                    slices[-2 * d + i] = slice(0, min_dim)
                    slices[-d + i] = slice(0, min_dim)
            new_array = new_array[tuple(slices)]

        n = math.prod(new_array.shape[-d:])
        new_array = math.reshape(new_array, (*new_array.shape[: -2 * d], n, n))
        trace = math.einsum("...ii->...", new_array)
        return ArrayAnsatz(trace, self.batch_dims)

    def __add__(self, other: Ansatz) -> ArrayAnsatz:
        r"""Adds two ArrayAnsatz together. In order to use the __add__ method, the ansatze must have
        the same batch dimensions. The shape of the core arrays must be such that one can be reduced
        to the other. Other will be reduced to the shape of self. If you want the opposite use other + self.

        Args:
            other: The other ansatz to add.

        Raises:
            TypeError: If ``other`` is not an ``ArrayAnsatz``.
            ValueError: If the batch dimensions don't match.
        """
        if not isinstance(other, ArrayAnsatz):
            raise TypeError(f"Cannot add ansatz of type {type(other)}!")

        if self.batch_dims != other.batch_dims:
            raise ValueError("Batch dimensions must match.")
        if self.core_shape != other.core_shape:
            if math.prod(self.core_shape) > math.prod(other.core_shape):
                self_array = self.array
                other_array = other.reduce(self.core_shape).array
            else:
                self_array = self.reduce(other.core_shape).array
                other_array = other.array
        else:
            self_array = self.array
            other_array = other.array
        return ArrayAnsatz(array=self_array + other_array, batch_dims=self.batch_dims)

    def __and__(self, other: Ansatz) -> ArrayAnsatz:
        if not isinstance(other, ArrayAnsatz):
            raise TypeError(f"Cannot tensor product ansatz of type {type(other)}!")
        if self.batch_shape != other.batch_shape:
            raise ValueError("Batch shapes must match.")
        batch_size = int(np.prod(self.batch_shape))
        self_reshaped = math.reshape(self.array, (batch_size, -1))
        other_reshaped = math.reshape(other.array, (batch_size, -1))
        new = math.einsum("ab,ac->abc", self_reshaped, other_reshaped)
        new = math.reshape(new, self.batch_shape + self.core_shape + other.core_shape)
        return ArrayAnsatz(array=new, batch_dims=self.batch_dims)

    def __call__(self, _: Any):
        raise AttributeError("Cannot call an ArrayAnsatz.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayAnsatz):
            return False
        if self.batch_shape != other.batch_shape:
            return False
        slices = tuple(slice(0, min(si, oi)) for si, oi in zip(self.core_shape, other.core_shape))
        return math.allclose(
            self.array[(..., *slices)],
            other.array[(..., *slices)],
            atol=settings.ATOL,
        )

    def __getitem__(self, index: Any) -> ArrayAnsatz:
        r"""Batch-only indexing. Supports integers, slices, None (newaxis), and Ellipsis.
        Indexing is restricted to the first ``batch_dims`` axes. Core axes are not indexable.

        The number of batch dimensions in the returned object is updated according to
        how many integers (remove) and Nones (insert) are used in the batch indexer.
        """
        final_index, removed_by_int, inserted_by_none = batch_indexer_info(index, self.batch_dims)
        new_array = self.array[final_index]
        return ArrayAnsatz(
            new_array, batch_dims=self.batch_dims - removed_by_int + inserted_by_none
        )

    def __mul__(self, other: Scalar) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array * other, batch_dims=self.batch_dims)

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array, batch_dims=self.batch_dims)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"ArrayAnsatz(shape={self.array.shape}, batch_dims={self.batch_dims})"

    def __truediv__(self, other: Scalar) -> ArrayAnsatz:
        # handle the case where other is a batched scalar
        shape = math.shape(other)
        if shape != ():
            delta = len(self.array.shape) - len(shape)
            other = math.reshape(
                other,
                shape + (1,) * delta,
            )
        return ArrayAnsatz(array=self.array / other, batch_dims=self.batch_dims)
