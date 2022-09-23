import numpy as np
from mrmustard.math.compactFockAmplitudes_diagonal import fock_representation_compact_diagonal
from mrmustard.math.compactFockAmplitudes_1leftoverMode import fock_representation_compact_1leftoverMode
from thewalrus._hafnian import input_validation

def hermite_multidimensional_diagonal(A,B,G0,cutoff,rtol=1e-05, atol=1e-08):
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    # Should I use np.real_if_close on A and B here?
    M = A.shape[0]//2

    G,G_dA,G_dB = fock_representation_compact_diagonal(A, B, G0, M, cutoff)
    G_dG0 = np.array(G / G0).astype(np.complex128)
    return G,G_dG0,G_dA,G_dB

def hermite_multidimensional_1leftoverMode(A,B,G0,cutoff, cutoff_leftoverMode,rtol=1e-05, atol=1e-08):
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    # Should I use np.real_if_close on A and B here?
    cutoff_leftoverMode = cutoff_leftoverMode.item() # tf.numpy_function() wraps this into an array, which leads to numba error if we want to iterate range(cutoff_leftoverMode)
    M = A.shape[0]//2
    if M<2:
        raise ValueError("The number of modes should be greater than 1. You might want to use hermite_multidimensional_diagonal instead.")

    G,G_dA,G_dB = fock_representation_compact_1leftoverMode(A, B, G0, M, cutoff, cutoff_leftoverMode)
    G_dG0 = np.array(G / G0).astype(np.complex128)
    return G,G_dG0,G_dA,G_dB