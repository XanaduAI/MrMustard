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
This module contains the Fock representation.
"""

from __future__ import annotations
from typing import Any, Callable, Sequence

from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from IPython.display import display

from mrmustard import math, widgets
from mrmustard.utils.typing import Batch, Scalar, Tensor, Vector

from .base import Representation

__all__ = ["Fock"]


class Fock(Representation):
    r"""
    The Fock representation of a broad class of quantum states, transformations, measurements,
    channels, etc.

    The ansatz available in this representation is ``ArrayAnsatz``.

    This function allows for vector space operations on Fock objects including
    linear combinations, outer product (``&``), and inner product (``@``).

    .. code-block::

        >>> from mrmustard.physics.representations import Fock

        >>> # initialize Fock objects
        >>> array1 = np.random.random((5,7,8))
        >>> array2 = np.random.random((5,7,8))
        >>> array3 = np.random.random((3,5,7,8)) # where 3 is the batch.
        >>> fock1 = Fock(array1)
        >>> fock2 = Fock(array2)
        >>> fock3 = Fock(array3, batched=True)

        >>> # linear combination can be done with the same batch dimension
        >>> fock4 = 1.3 * fock1 - fock2 * 2.1

        >>> # division by a scalar
        >>> fock5 = fock1 / 1.3

        >>> # inner product by contracting on marked indices
        >>> fock6 = fock1[2] @ fock3[2]

        >>> # outer product (tensor product)
        >>> fock7 = fock1 & fock3

        >>> # conjugation
        >>> fock8 = fock1.conj

    Args:
        array: the (batched) array in Fock representation.
        batched: whether the array input has a batch dimension.

    Note: The args can be passed non-batched, as they will be automatically broadcasted to the
    correct batch shape.

    """

    def __init__(self, array: Batch[Tensor], batched=False):
        super().__init__()
        self._array = array if batched else [array]
        self._backend_array = False
        self._original_abc_data = None

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this representation.
        """
        self._generate_ansatz()
        if not self._backend_array:
            self._array = math.astensor(self._array)
            self._backend_array = True
        return self._array

    @array.setter
    def array(self, value):
        self._array = value
        self._backend_array = False

    @property
    def batch_size(self):
        return self.array.shape[0]

    @property
    def conj(self):
        ret = Fock(math.conj(self.array), batched=True)
        ret._contract_idxs = self._contract_idxs  # pylint: disable=protected-access
        return ret

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - 1

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the representation.
        I.e. the vacuum component of the Fock object, whatever it may be.
        Given that the first axis of the array is the batch axis, this is the first element of the array.
        """
        return self.array[(slice(None),) + (0,) * self.num_vars]

    @property
    def triple(self) -> tuple:
        r"""
        The data of the original Bargmann if it exists.
        """
        if self._original_abc_data is None:
            raise AttributeError(
                "This Fock object does not have an original Bargmann representation."
            )
        return self._original_abc_data

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Fock:
        return cls(data["array"], batched=True)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Fock:
        ret = cls(None, True)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def reduce(self, shape: int | Sequence[int]) -> Fock:
        r"""
        Returns a new ``Fock`` with a sliced array.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.representations import Fock

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = Fock(array1)

            >>> fock2 = fock1.reduce(3)
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce(2)
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == Fock(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == Fock(array4)

        Args:
            shape: The shape of the array of the returned ``Fock``.
        """
        if shape == self.array.shape[1:]:
            return self
        length = self.num_vars
        shape = (shape,) * length if isinstance(shape, int) else shape
        if len(shape) != length:
            msg = f"Expected shape of length {length}, "
            msg += f"given shape has length {len(shape)}."
            raise ValueError(msg)

        if any(s > t for s, t in zip(shape, self.array.shape[1:])):
            warn(
                "Warning: the fock array is being padded with zeros. If possible slice the arrays this one will contract with instead."
            )
            padded = math.pad(
                self.array,
                [(0, 0)] + [(0, s - t) for s, t in zip(shape, self.array.shape[1:])],
            )
            return Fock(padded, batched=True)

        ret = self.array[(slice(0, None),) + tuple(slice(0, s) for s in shape)]
        return Fock(array=ret, batched=True)

    def reorder(self, order: tuple[int, ...] | list[int]) -> Fock:
        return Fock(math.transpose(self.array, [0] + [i + 1 for i in order]), batched=True)

    def sum_batch(self) -> Fock:
        r"""
        Sums over the batch dimension of the array. Turns an object with any batch size to a batch size of 1.

        Returns:
            The collapsed Fock object.
        """
        return Fock(math.expand_dims(math.sum(self.array, axes=[0]), 0), batched=True)

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data}

    def trace(self, idxs1: tuple[int, ...], idxs2: tuple[int, ...]) -> Fock:
        if len(idxs1) != len(idxs2) or not set(idxs1).isdisjoint(idxs2):
            raise ValueError("idxs must be of equal length and disjoint")
        order = (
            [0]
            + [i + 1 for i in range(len(self.array.shape) - 1) if i not in idxs1 + idxs2]
            + [i + 1 for i in idxs1]
            + [i + 1 for i in idxs2]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idxs2) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idxs1)] + (n, n))
        trace = math.trace(new_array)
        return Fock([trace] if trace.shape == () else trace, batched=True)

    def _generate_ansatz(self):
        if self._array is None:
            self.array = [self._fn(**self._kwargs)]

    def _ipython_display_(self):
        w = widgets.fock(self)
        if w is None:
            print(repr(self))
            return
        display(w)

    def __add__(self, other: Fock) -> Fock:
        try:
            diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
            if diff < 0:
                new_array = [
                    a + b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                ]
            else:
                new_array = [
                    a + b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                ]
            return Fock(array=new_array, batched=True)
        except Exception as e:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: Fock) -> Fock:
        new_array = [math.outer(a, b) for a in self.array for b in other.array]
        return Fock(array=new_array, batched=True)

    def __call__(self, z: Batch[Vector]) -> Scalar:
        raise AttributeError("Cannot call Fock.")

    def __eq__(self, other: Representation) -> bool:
        slices = (slice(0, None),) + tuple(
            slice(0, min(si, oi)) for si, oi in zip(self.array.shape[1:], other.array.shape[1:])
        )
        return np.allclose(self.array[slices], other.array[slices], atol=1e-10)

    def __getitem__(self, idx: int | tuple[int, ...]) -> Fock:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for representation with {self.num_vars} variables."
                )
        ret = Fock(self.array, batched=True)
        ret._contract_idxs = idx
        return ret

    def __matmul__(self, other: Fock) -> Fock:
        idx_s = list(self._contract_idxs)
        idx_o = list(other._contract_idxs)

        # the number of batches in self and other
        n_batches_s = self.array.shape[0]
        n_batches_o = other.array.shape[0]

        # the shapes each batch in self and other
        shape_s = self.array.shape[1:]
        shape_o = other.array.shape[1:]

        new_shape_s = list(shape_s)
        new_shape_o = list(shape_o)
        for s, o in zip(idx_s, idx_o):
            new_shape_s[s] = min(shape_s[s], shape_o[o])
            new_shape_o[o] = min(shape_s[s], shape_o[o])

        reduced_s = self.reduce(new_shape_s)[idx_s]
        reduced_o = other.reduce(new_shape_o)[idx_o]

        axes = [list(idx_s), list(idx_o)]
        batched_array = []
        for i in range(n_batches_s):
            for j in range(n_batches_o):
                batched_array.append(math.tensordot(reduced_s.array[i], reduced_o.array[j], axes))
        return Fock(batched_array, batched=True)

    def __mul__(self, other: Scalar | Fock) -> Fock:
        if isinstance(other, Fock):
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
                return Fock(array=new_array, batched=True)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
        else:
            ret = Fock(array=self.array * other, batched=True)
            ret._original_abc_data = (
                tuple(i * j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret

    def __neg__(self) -> Fock:
        return Fock(array=-self.array, batched=True)

    def __truediv__(self, other: Scalar | Fock) -> Fock:
        if isinstance(other, Fock):
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
                return Fock(array=new_array, batched=True)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e
        else:
            ret = Fock(array=self.array / other, batched=True)
            ret._original_abc_data = (
                tuple(i / j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret
