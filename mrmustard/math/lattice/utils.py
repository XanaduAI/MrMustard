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
