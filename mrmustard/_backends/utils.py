import numpy as np
from numba import njit, objmode
from numba.cpython.unsafe.tuple import tuple_setitem
from functools import lru_cache
from itertools import product
from typing import Sequence, Tuple, List, Generator

SQRT = np.sqrt(np.arange(1000))  # saving the time to recompute square roots

@lru_cache()
def Xmat(num_modes:int):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`
    Args:
        num_modes (int): positive integer
    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(num_modes)
    O = np.zeros((num_modes,num_modes))
    return np.block([[O, I], [I, O]])

@lru_cache()
def rotmat(num_modes:int):
    "Rotation matrix from quadratures to complex amplitudes"
    idl = np.identity(num_modes)
    return np.sqrt(0.5) * np.block([[idl, 1j * idl], [idl, -1j * idl]])


@lru_cache()
def J(num_modes: int):
    'Symplectic form'
    I = np.identity(num_modes)
    O = np.zeros_like(I)
    return np.block([[O, I], [-I, O]])


@lru_cache()
def fixed_cov(l, choi_r=np.arcsinh(1.0), hbar=2):
    'Construct the covariance matrix of l two-mode squeezed vacua pairing modes i and i+l'
    ch = np.diag([np.cosh(choi_r)] * l)
    sh = np.diag([np.sinh(choi_r)] * l)
    zh = np.zeros([l, l])
    return np.block(
        [[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]]
    )



# LOW-LEVEL NUMBA CODE

@lru_cache()
def partition(photons: int, max_vals: Tuple[int,...]) -> Tuple[Tuple[int,...],...]:
    "Returns a list of all the ways of putting n photons into modes that have at most (n1, n2, etc.) photons each"
    return [comb for comb in product(*(range(min(photons, i) + 1) for i in max_vals)) if sum(comb) == photons]

@njit
def remove(pattern: Tuple[int,...]) -> Generator[Tuple[int, Tuple[int,...]], None, None]:
    "returns a generator for all the possible ways to decrease elements of the given tuple by 1"
    for p, n in enumerate(pattern):
        if n > 0:
            yield p, dec(pattern, p)

@njit
def dec(tup: Tuple[int], i: int) -> Tuple[int,...]:
    "returns a copy of the given tuple of integers where the ith element has been decreased by 1"
    copy = tup[:]
    return tuple_setitem(copy, i, tup[i] - 1)


def fill_amplitudes(array, A, B, max_photons:Tuple[int,...]):
    "fills the amplitudes "
    for tot_photons in range(1, sum(max_photons) + 1):
        for idx in partition(tot_photons, max_photons):
            array = fill_amplitudes_numbaloop(array, idx, A, B)
    return array


@njit
def fill_amplitudes_numbaloop(array, idx, A, B):
    for i, val in enumerate(idx):
        if val > 0:
            break
    ki = dec(idx, i)
    u = B[i]*array[ki]
    for p, kp in remove(ki):
        u += SQRT[ki[p]] * A[i, p] * array[kp]
    array[idx] = u / SQRT[idx[i]]
    return array


def fill_gradients(dA, dB, state, A, B, max_photons:Tuple[int,...]):
    for tot_photons in range(1, sum(max_photons) + 1):
        for idx in partition(tot_photons, max_photons):
            dA, dB = fill_gradients_numbaloop(dA, dB, state, idx, A, B)
    return dA, dB


@njit
def fill_gradients_numbaloop(dA, dB, state, idx, A, B):
    for i, val in enumerate(idx):
        if val > 0:
            break
    ki = dec(idx, i)
    dudA = B[i]*dA[ki]
    dudB = B[i]*dB[ki]
    dudB[i] += state[ki]
    for p, kp in remove(ki):
        dudA += SQRT[ki[p]] * A[i, p] * dA[kp]
        dudA[i, p] -= SQRT[ki[p]] * state[kp]
        dudB += SQRT[ki[p]] * A[i, p] * dB[kp]
    dA[idx] = dudA / SQRT[idx[i]]
    dB[idx] = dudB / SQRT[idx[i]]
    return dA, dB