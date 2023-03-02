import numba
import numpy as np
from numba import njit


@njit
def ravel_multi_index(index, shape):
    res = 0
    for i in range(len(index)):
        res += index[i] * np.prod(np.asarray(shape)[i + 1 :])
    return res


@njit
def tensor_value(tensor, index):
    return tensor.flat[ravel_multi_index(index, tensor.shape)]


class FockDict:
    def __init__(self, M):
        self._data = numba.typed.Dict.empty(numba.types.UniTuple(numba.int64, M), numba.complex128)
        self.M = M

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        if isinstance(indices, tuple):
            if all(isinstance(n, int) for n in indices) and len(indices) == self.M:
                return self._data[indices] if indices in self._data else 0.0
            slices = ()
            for index in indices:
                if isinstance(index, int):
                    slices = slices + (slice(index, index + 1),)
                elif isinstance(index, slice):
                    slices = slices + (index,)
                else:
                    raise TypeError(f"Invalid index type: {type(index)}")
            for i in range(len(indices), self.M):
                slices = slices + (slice(None),)

            new = FockDict(self.M)
            for key, value in self._data.items():
                if compatible(key, slices):
                    new._data[key] = value
            return new

        else:
            raise TypeError(f"Invalid index type: {type(indices)}")

    def __setitem__(self, indices, value):
        if isinstance(indices, tuple):
            self._data[indices] = value
        elif isinstance(indices, int):
            self._data[(indices,)] = value
        else:
            raise TypeError(f"Invalid index type: {type(indices)}")

    def __repr__(self):
        return f"FockDict({len(self._data)} elements)"


@njit
def compatible(key, slices):
    for i, s in enumerate(slices):
        if not_compatible_slice(key[i], s):
            return False
    return True


@njit
def not_compatible_slice(key_i, s):
    if (s.start is not None and key_i < s.start) or (s.stop is not None and key_i >= s.stop):
        return True
    return False
