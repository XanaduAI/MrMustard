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

import numba


def digits_to_int(digits):
    return sum(digit * 10**i for i, digit in enumerate(digits))


def f(shape, k, index):
    return digits_to_int(index + (k,)) if len(index) == 2 else 99


def binomial_recurrence(shape, k):
    if len(shape) == 1:
        if shape[0] >= k:
            return [(k,)]
        else:
            return []
    result = []
    for i in range(min(shape[0], k) + 1):
        for indices in binomial_recurrence(shape[1:], k - i):
            result.append((i,) + indices)
    return result


def binomial_recurrence_gen(shape, k):
    if len(shape) == 1:
        if shape[0] >= k:
            yield (k,)
        return
    for i in range(min(shape[0], k) + 1):
        for indices in binomial_recurrence(shape[1:], k - i):
            yield (i,) + indices


class BinomialData:
    def __new__(cls, shape, weight, index=()):
        if len(shape) == 1 and weight < shape[0]:
            return (weight, f(shape, weight, index))
        if weight > sum(shape) - len(shape):
            return NoneList(shape)
        if len(shape) > 0:
            return super().__new__(cls)

    def __init__(self, shape, index=()):
        self.shape = shape
        self.index = index
        self.axis = []
        self._indices = None
        for photons in range(min(shape[0], self.weight + 1)):
            # if self.weight + photons == weight:
            #     self.axis.append(f(self.shape, weight, self.index + (photons,)))
            #     break
            self.axis.append(
                BinomialData(shape[1:], self.weight - photons, self.index + (photons,))
            )

    @property
    def weight(self):
        return sum(self.index)

    @property
    def indices(self):
        r"""indexes the current manifold with sum k, of which self.index is part"""
        yield from binomial_recurrence_gen(self.shape, self.k)

    def __getitem__(self, key):
        if type(key) == int:
            return self.axis[key]
        if type(key) == tuple:
            if len(key) == 1:
                return self.axis[key[0]]
            return self.axis[key[0]][key[1:]]

    def __iter__(self):
        yield from self.axis

    def __repr__(self):
        return str(self.axis)


class NoneList:
    def __init__(self, shape: list[int]):
        self.shape = shape

    def __getitem__(self, key):
        if isinstance(key, int) and key >= self.shape[0]:
            raise IndexError
        elif isinstance(key, tuple):
            if any(k >= s for k, s in zip(key, self.shape)):
                raise IndexError
            return NoneList(self.shape[len(key) :])
        return NoneList(self.shape[1:])

    def __repr__(self):
        return "NL"


class FockDict:
    r"""A dictionary that stores the values of a tensor in Fock basis.
    Args:
        M (int): number of modes

    Example:
        >>> fock = FockDict(2)
        >>> fock[0, 0] = 1.0
        >>> fock[1, 0] = 2.0
        >>> fock[0, 1] = 3.0
        >>> fock[1, 1] = 4.0
        >>> fock[0, 0]
        1.0
        >>> fock[0, 0] = 5.0
        >>> fock[0, 0]
        5.0
        >>> fock[0, :]
        FockDict({(0, 0): 5.0, (0, 1): 3.0})
    """

    def __init__(self, M):
        self._data = numba.typed.Dict.empty(numba.types.UniTuple(numba.int64, M), numba.complex128)
        self.M = M

    def _parse_indices(self, indices):
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        elif isinstance(indices, tuple):
            if len(indices) > self.M:
                raise IndexError(f"Too many indices for FockDict with dimension {self.M}")
        else:
            raise TypeError(f"Invalid index type: {type(indices)}")

        full_indices = list(indices) + [slice(None)] * (self.M - len(indices))
        return tuple(full_indices)

    def __getitem__(self, indices):
        indices = self._parse_indices(indices)

        new = FockDict(self.M)
        for key, value in self._data.items():
            if all(
                idx == key[i] if isinstance(idx, int) else idx.start <= key[i] < idx.stop
                for i, idx in enumerate(indices)
            ):
                new._data[key] = value

        return new if len(new._data) > 1 else next(iter(new._data.values()), 0.0)

    def __setitem__(self, indices, value):
        indices = self._parse_indices(indices)

        for key in list(self._data.keys()):
            if all(
                idx == key[i] if isinstance(idx, int) else idx.start <= key[i] < idx.stop
                for i, idx in enumerate(indices)
            ):
                del self._data[key]

        if not any(isinstance(idx, slice) for idx in indices):
            self._data[tuple(indices)] = value

    def __repr__(self):
        return f"FockDict({len(self._data)} elements)"
