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
This module contains the array ansatz.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from warnings import warn

import numpy as np
from IPython.display import display
from numpy.typing import ArrayLike

from mrmustard import math, settings, widgets
from mrmustard.math.parameters import Variable
from mrmustard.utils.typing import Batch, Scalar, Tensor

from .base import Ansatz

__all__ = ["ArrayAnsatz"]


class ArrayAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz as a multidimensional array.

    .. code-block::

          >>> from mrmustard.physics.ansatz import ArrayAnsatz

          >>> array = np.random.random((2, 4, 5))
          >>> ansatz = ArrayAnsatz(array)

    Args:
        array: A (potentially) batched array.
        batch_dims: The number of batch dimensions.
    """

    def __init__(self, array: Batch[Tensor] | None, batch_dims: int = 0):
        super().__init__()
        self._array = array
        self._original_abc_data = None
        self._batch_dims = batch_dims
        if array is not None:
            self._batch_shape = tuple(self._array.shape[:batch_dims])

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
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
        if self._array is None:
            self._generate_ansatz()
        return self._batch_shape

    @property
    def batch_size(self):
        return int(np.prod(self.batch_shape)) if self.batch_shape else 0

    @property
    def core_dims(self) -> int:
        r"""
        The number of core dimensions of this ansatz.
        """
        return len(self.core_shape)

    @property
    def conj(self):
        return ArrayAnsatz(math.conj(self.array), self.batch_dims)

    @property
    def core_shape(self) -> tuple[int, ...] | None:
        r"""
        The core dimensions of this ansatz.
        """
        return self.array.shape[self.batch_dims :]

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - self.batch_dims

    @property
    def scalar(self) -> Scalar | ArrayLike:
        r"""
        The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        """
        return self.array[(...,) + (0,) * self.core_dims]

    @property
    def triple(self) -> tuple:
        r"""
        The data of the original PolyExpAnsatz if it exists.
        """
        if self._original_abc_data is None:
            raise AttributeError("This ArrayAnsatz does not have (A,b,c) data.")
        return self._original_abc_data

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> ArrayAnsatz:
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, batch_dims: int = 0, **kwargs: Any) -> ArrayAnsatz:
        ret = cls(None, batch_dims=batch_dims)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        (ret._kwargs,) = children
        (
            ret._batch_shape,
            ret._lin_sup,
            ret._fn,
            ret._array,
            ret._original_abc_data,
            ret._batch_dims,
        ) = aux_data
        return ret

    def contract(
        self,
        other: ArrayAnsatz,
        idx1: Sequence[str | int],
        idx2: Sequence[str | int],
        idx_out: Sequence[str | int],
    ) -> ArrayAnsatz:
        r"""Contracts this ansatz with another using einsum-style notation with labels.

        Indices are specified as sequences of labels (str or int). Batch dimensions must
        be strings, core dimensions must be integers.

        Example:
            `self.contract(other, idx1=['b', 0, 1], idx2=['b', 1, 2], idx_out=[0, 2])`
            Contracts batch label 'b' and core index 1.

        Args:
            other: The other ArrayAnsatz to contract with.
            idx1: Sequence of labels (str/int) for this ansatz's dimensions. Must match rank.
            idx2: Sequence of labels (str/int) for the other ansatz's dimensions. Must match rank.
            idx_out: Sequence of labels for the output dimensions. Must be subset of input labels.

        Returns:
            The contracted ArrayAnsatz.

        Raises:
            ValueError: If index sequences have incorrect length or invalid labels.
        """
        if len(idx1) != len(self.array.shape):
            raise ValueError(f"expected len(idx1)={self.array.ndim} got {len(idx1)}")
        if len(idx2) != len(other.array.shape):
            raise ValueError(f"expected len(idx2)={other.array.ndim} got {len(idx2)}")

        all_labels_in = set(idx1) | set(idx2)
        if not set(idx_out).issubset(all_labels_in):
            raise ValueError("Output labels must be present in input labels.")

        unique_labels = sorted(all_labels_in, key=lambda x: (isinstance(x, int), x))
        label_to_char = {label: chr(97 + i) for i, label in enumerate(unique_labels)}

        einsum_str1 = "".join([label_to_char[i] for i in idx1])
        einsum_str2 = "".join([label_to_char[i] for i in idx2])
        einsum_str_out = "".join([label_to_char[i] for i in idx_out])
        einsum_str = f"{einsum_str1},{einsum_str2}->{einsum_str_out}"

        contracted_labels = set(idx1) & set(idx2)
        array1 = self.array
        array2 = other.array
        slices1 = [slice(None)] * len(idx1)
        slices2 = [slice(None)] * len(idx2)

        for label in contracted_labels:
            pos1 = idx1.index(label)
            dim1 = array1.shape[pos1]
            pos2 = idx2.index(label)
            dim2 = array2.shape[pos2]

            if dim1 != dim2:
                min_dim = min(dim1, dim2)
                slices1[pos1] = slice(0, min_dim)
                slices2[pos2] = slice(0, min_dim)

        reduced_array1 = array1[tuple(slices1)]
        reduced_array2 = array2[tuple(slices2)]
        result = math.einsum(einsum_str, reduced_array1, reduced_array2)

        batch_dims_out = sum(1 for label in idx_out if isinstance(label, str))
        return ArrayAnsatz(result, batch_dims=batch_dims_out)

    def reduce(self, shape: Sequence[int]) -> ArrayAnsatz:
        r"""
        Returns a new ``ArrayAnsatz`` with a sliced core shape.

        .. code-block::

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

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.
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
        z_dims = new_array.shape[-2 * len(idx_z) : -len(idx_z)]
        zconj_dims = new_array.shape[-len(idx_z) :]
        slices = [slice(None)] * len(new_array.shape)
        for i, (z_dim, zconj_dim) in enumerate(zip(z_dims, zconj_dims)):
            if z_dim != zconj_dim:
                min_dim = min(z_dim, zconj_dim)
                slices[-2 * len(idx_z) + i] = slice(0, min_dim)
                slices[-len(idx_z) + i] = slice(0, min_dim)
        new_array = new_array[tuple(slices)]

        n = math.prod(new_array.shape[-len(idx_zconj) :])
        new_array = math.reshape(new_array, (*new_array.shape[: -2 * len(idx_z)], n, n))
        trace = math.trace(new_array)
        return ArrayAnsatz(trace, self.batch_dims)

    def _generate_ansatz(self):
        r"""
        Computes and sets the array given a function and its kwargs.
        """
        if self._should_regenerate():
            params = {}
            for name, param in self._kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param
            self.array = self._fn(**params)

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL or (w := widgets.fock(self)) is None:
            print(self)
            return
        display(w)

    def _should_regenerate(self):
        r"""
        Determines if the ansatz needs to be regenerated based on its current state
        and parameter types.
        """
        return self._array is None or Variable in {type(param) for param in self._kwargs.values()}

    def _tree_flatten(self):  # pragma: no cover
        children, aux_data = super()._tree_flatten()
        aux_data += (self._array, self._original_abc_data, self._batch_dims)
        return (children, aux_data)

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        r"""
        Adds two ArrayAnsatz together. In order to use the __add__ method, the ansatze must have
        the same batch dimensions. The shape of the core arrays must be such that one can be reduced
        to the other. Other will be reduced to the shape of self. If you want the opposite use other + self.
        """
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

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
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

    def __eq__(self, other: Ansatz) -> bool:
        if self.batch_shape != other.batch_shape:
            return False
        slices = tuple(slice(0, min(si, oi)) for si, oi in zip(self.core_shape, other.core_shape))
        return math.allclose(
            self.array[(..., *slices)],
            other.array[(..., *slices)],
            atol=settings.ATOL,
        )

    def __mul__(self, other: Scalar | ArrayLike) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array * other, batch_dims=self.batch_dims)

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array, batch_dims=self.batch_dims)

    def __truediv__(self, other: Scalar | ArrayLike) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array / other, batch_dims=self.batch_dims)
