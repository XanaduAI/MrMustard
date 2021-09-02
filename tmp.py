import numpy as np
from numba import njit

@njit
def all_arange(cutoffs):
    for i,c in enumerate(cutoffs):
        r = np.arange(c)
        for _ in range(i):
            r = np.expand_dims(r, axis=0)
        for _ in range(len(cutoffs)-i-1):
            r = np.expand_dims(r, axis=-1)
        yield r

if __name__=='__main__':
    print(all_arange((2,3,2)))

