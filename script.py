from numba import njit
import numpy as np

z = np.zeros(shape=(4,), dtype=np.int32)

@njit
def generator(x):
    for i in range(4):
        x[i] = 1
        yield x

for g in generator(z):
    print(g)
