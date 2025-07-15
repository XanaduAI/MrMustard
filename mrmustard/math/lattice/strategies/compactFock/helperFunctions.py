"""
This module contains helper functions that are used in
diagonal_amps.py, diagonal_grad.py, singleLeftoverMode_amps.py and singleLeftoverMode_grad.py
"""

import numba
import numpy as np
from numba import int64, njit
from numba.typed import Dict

SQRT = np.sqrt(np.arange(1000))  # saving the time to recompute square roots


@njit(cache=True)
def repeat_twice(params):
    """
    This function is equivalent to np.repeat(params,2), but runs faster.
    Args:
        params (1D array): [a,b,c,...]
    Returns:
        (1D array): [a,a,b,b,c,c,...]
    """
    pivot = np.zeros(2 * len(params), dtype=np.int64)
    for i, val in enumerate(params):
        pivot[2 * i] = val
        pivot[2 * i + 1] = val
    return pivot


@njit(cache=True)
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
