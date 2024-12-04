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
from typing import Any, Callable, Sequence

from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from IPython.display import display

from mrmustard import math, widgets
from mrmustard.utils.typing import Scalar, Tensor, Vector

from .base import Ansatz
from ..batches import Batch

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
        batch_labels: The (optional) batch labels for this ansatz.

    """

    def __init__(self, array: Tensor | Batch[Tensor], batch_labels: list[str] | None = None):
        super().__init__()
        self._array = (
            Batch([array], batch_labels=batch_labels)
            if array is not None and not isinstance(array, Batch)
            else array
        )
        self._original_abc_data = None

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        return self._array

    @array.setter
    def array(self, value):
        self._array = value if isinstance(value, Batch) else Batch([value])

    @property
    def batch_size(self) -> int:
        return sum(self.array.batch_shape)

    @property
    def conj(self) -> ArrayAnsatz:
        return ArrayAnsatz(Batch(math.conj(self.array)))

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.core_shape)

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        Given that the first axis of the array is the batch axis, this is the first element of the array.
        """
        return self.array.data[(slice(None),) * len(self.array.batch_shape) + (0,) * self.num_vars]

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
        return cls(data["array"])

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> ArrayAnsatz:
        ret = cls(None)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def contract(
        self,
        other: ArrayAnsatz,
        idx1: int | tuple[int, ...] | None = None,
        idx2: int | tuple[int, ...] | None = None,
    ) -> ArrayAnsatz:
        idx1 = idx1 or ()
        idx2 = idx2 or ()
        idx1 = (idx1,) if isinstance(idx1, int) else idx1
        idx2 = (idx2,) if isinstance(idx2, int) else idx2
        for i, j in zip(idx1, idx2):
            if i >= self.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for representation with {self.num_vars} variables."
                )
            if j >= other.num_vars:
                raise IndexError(
                    f"Index {j} out of bounds for representation with {other.num_vars} variables."
                )

        n_batches_s = self.batch_size
        n_batches_o = other.batch_size

        shape_s = self.array.core_shape
        shape_o = other.array.core_shape

        new_shape_s = list(shape_s)
        new_shape_o = list(shape_o)
        for s, o in zip(idx1, idx2):
            new_shape_s[s] = min(shape_s[s], shape_o[o])
            new_shape_o[o] = min(shape_s[s], shape_o[o])

        reduced_s = self.reduce(new_shape_s)
        reduced_o = other.reduce(new_shape_o)

        axes = [list(idx1), list(idx2)]
        batched_array = []
        for i in range(n_batches_s):
            for j in range(n_batches_o):
                batched_array.append(math.tensordot(reduced_s.array[i], reduced_o.array[j], axes))
        return ArrayAnsatz(Batch(batched_array))

    def reduce(self, shape: int | Sequence[int]) -> ArrayAnsatz:
        r"""
        Returns a new ``ArrayAnsatz`` with a sliced array.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.ansatz import ArrayAnsatz

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = ArrayAnsatz(array1)

            >>> fock2 = fock1.reduce(3)
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce(2)
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == ArrayAnsatz(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == ArrayAnsatz(array4)

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.
        """
        if shape == self.array.core_shape:
            return self
        length = self.num_vars
        shape = (shape,) * length if isinstance(shape, int) else shape
        if len(shape) != length:
            msg = f"Expected shape of length {length}, "
            msg += f"given shape has length {len(shape)}."
            raise ValueError(msg)

        if any(s > t for s, t in zip(shape, self.array.core_shape)):
            warn(
                "Warning: the fock array is being padded with zeros. If possible, slice the arrays this one will contract with instead."
            )
            padded = math.pad(
                self.array,
                [(0, 0)] * len(self.array.batch_shape)
                + [(0, s - t) for s, t in zip(shape, self.array.core_shape)],
            )
            return ArrayAnsatz(Batch(padded))

        ret = self.array.data[
            (slice(None),) * len(self.array.batch_shape) + tuple(slice(0, s) for s in shape)
        ]
        return ArrayAnsatz(array=Batch(ret))

    def reorder(self, order: tuple[int, ...] | list[int]) -> ArrayAnsatz:
        return ArrayAnsatz(Batch(math.transpose(self.array, [0] + [i + 1 for i in order])))

    def sum_batch(self) -> ArrayAnsatz:
        r"""
        Sums over the batch dimension of the array. Turns an object with any batch size to a batch size of 1.

        Returns:
            The collapsed ArrayAnsatz object.
        """
        return ArrayAnsatz(Batch(math.expand_dims(math.sum(self.array, axes=[0]), 0)))

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> ArrayAnsatz:
        if len(idx_z) != len(idx_zconj) or not set(idx_z).isdisjoint(idx_zconj):
            raise ValueError("The idxs must be of equal length and disjoint.")
        order = (
            [0]
            + [i + 1 for i in range(len(self.array.core_shape)) if i not in idx_z + idx_zconj]
            + [i + 1 for i in idx_z]
            + [i + 1 for i in idx_zconj]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idx_zconj) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idx_z)] + (n, n))
        trace = math.trace(new_array)
        return ArrayAnsatz(Batch([trace] if trace.shape == () else trace))

    def _generate_ansatz(self):
        if self._array is None:
            self.array = self._fn(**self._kwargs)

    def _ipython_display_(self):
        w = widgets.fock(self)
        if w is None:
            print(repr(self))
            return
        display(w)

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        try:
            diff = sum(self.array.core_shape) - sum(other.array.core_shape)
            if diff < 0:
                new_array = [
                    a + b for a in self.reduce(other.array.core_shape).array for b in other.array
                ]
            else:
                new_array = [
                    a + b for a in self.array for b in other.reduce(self.array.core_shape).array
                ]
            return ArrayAnsatz(array=Batch(new_array))
        except Exception as e:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        new_array = [math.outer(a, b) for a in self.array for b in other.array]
        return ArrayAnsatz(array=Batch(new_array))

    def __call__(self, z: Batch[Vector]) -> Scalar:
        raise AttributeError("Cannot call this ArrayAnsatz.")

    def __eq__(self, other: Ansatz) -> bool:
        slices = (slice(None),) * len(self.array.batch_shape) + tuple(
            slice(0, min(si, oi)) for si, oi in zip(self.array.core_shape, other.array.core_shape)
        )
        return np.allclose(self.array.data[slices], other.array.data[slices], atol=1e-10)

    def __getitem__(self, idxs: int | slice | tuple[int, ...] | tuple[slice, ...]) -> ArrayAnsatz:
        return ArrayAnsatz(self.array[idxs])

    def __mul__(self, other: Scalar | ArrayAnsatz) -> ArrayAnsatz:
        if isinstance(other, ArrayAnsatz):
            try:
                diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
                if diff < 0:
                    new_array = [
                        a * b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                    ]
                else:
                    new_array = [
                        a * b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                    ]
                return ArrayAnsatz(array=Batch(new_array))
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
        else:
            ret = ArrayAnsatz(array=self.array * other)
            ret._original_abc_data = (
                tuple(i * j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array)

    def __truediv__(self, other: Scalar | ArrayAnsatz) -> ArrayAnsatz:
        if isinstance(other, ArrayAnsatz):
            try:
                diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
                if diff < 0:
                    new_array = [
                        a / b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                    ]
                else:
                    new_array = [
                        a / b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                    ]
                return ArrayAnsatz(array=Batch(new_array))
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e
        else:
            ret = ArrayAnsatz(array=self.array / other)
            ret._original_abc_data = (
                tuple(i / j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret
