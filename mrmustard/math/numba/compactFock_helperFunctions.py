"""
This module contains helper functions that are used in
compactFock_diagonal_amps.py, compactFock_diagonal_grad.py, compactFock_1leftoverMode_amps.py and compactFock_1leftoverMode_grad.py
"""

import numpy as np
from numba import njit, int64
from numba.typed import Dict
import numba

SQRT = np.sqrt(np.arange(1000))  # saving the time to recompute square roots


@njit
def repeat_twice(params):
    """
    This function is equivalent to np.repeat(params,2), but runs faster.
    Args:
        params (1D array): [a,b,c,...]
    Returns:
        (1D array): [a,a,b,b,c,c,...]
    """
    pivot = np.empty(2 * len(params), dtype=np.int64)
    for i, val in enumerate(params):
        pivot[2 * i] = val
        pivot[2 * i + 1] = val
    return pivot


@njit
def construct_dict_params(cutoffs, tuple_type, list_type):
    """
    Args:
        cutoffs (tuple): upper bounds for the number of photons in each mode
        tuple_type,list_type (numba types): numba types that need to be defined outside of numba compiled functions
    Returns:
        (typed Dict): all possible values for (a,b,c,...), grouped in lists according to their sum a+b+c+...
    """
    indices = Dict.empty(key_type=int64, value_type=list_type)
    for sum_params in range(sum(cutoffs)):
        indices[sum_params] = numba.typed.List.empty_list(tuple_type)

    for params in np.ndindex(cutoffs):
        indices[sum(params)].append(params)
    return indices
