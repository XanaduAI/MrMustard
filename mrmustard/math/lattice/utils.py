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
        if isinstance(indices, int) or isinstance(indices, slice):
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
