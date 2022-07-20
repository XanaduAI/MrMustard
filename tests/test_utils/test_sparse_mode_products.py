from hypothesis import settings, given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import tensorflow as tf

from mrmustard.math.numba import *
import thewalrus as tw

def test_numba_sparse_matvec_1_to_1():
    r"""tests that a 1-mode matrix is correctly applied to a 2-mode or 3-mode vector"""
    # 2-mode vector
    x = np.array([[1.,3],[2,4]]).reshape((1,2,2))
    # 1-mode matrix like_1 or like_0
    T = np.array([[1.,2],[3,4]]).reshape((1,1,1,2,2))
    # expected results
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[0,1], mlike_0=True), np.array([7, 15]).reshape((1,1,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[0,1], mlike_0=False), np.array([[7, 15], [2, 4]]).reshape((1,2,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[1], v_modes=[0,1], mlike_0=False), np.array([[1, 3], [10, 22]]).reshape((1,2,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[1], v_modes=[1,2], mlike_0=False), np.array([[7, 15], [2, 4]]).reshape((1,2,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[2], v_modes=[2,1], mlike_0=False), np.array([[7, 15], [2, 4]]).reshape((1,2,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[1], v_modes=[1,2], mlike_0=True), np.array([7, 15]).reshape((1,1,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[2], v_modes=[2,1], mlike_0=True), np.array([7, 15]).reshape((1,1,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[1,2], mlike_0=False), x)

    # 3-mode vector
    x = np.array([[1., 4], [2, 5], [3, 6]]).reshape((1,3,2))
    # 1-mode matrix like_1 or like_0
    T = np.array([[1., 2], [3, 4]]).reshape((1,1,1,2,2))
    # expected results
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[0,1,2], mlike_0=True), np.array([9, 19]).reshape((1,1,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[0,1,2], mlike_0=False), np.array([9, 19, 2, 5, 3, 6]).reshape((1,3,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[1], v_modes=[0,1,2], mlike_0=False), np.array([1, 4, 12, 26, 3, 6]).reshape((1,3,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[1], v_modes=[1,2,3], mlike_0=False), np.array([9, 19, 2, 5, 3, 6]).reshape((1,3,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[2], v_modes=[1,2,3], mlike_0=True), np.array([12, 26]).reshape((1,1,2)))
    assert np.allclose(numba_sparse_matvec(matrix=T, vector=x.copy(), m_modes=[0], v_modes=[1,2,3], mlike_0=False), x)
    
def test_numba_sparse_matmul_1_to_2():
    r"""tests that a batched 1-mode matrix is correctly composed with a 2-mode matrix"""
    # 2-mode matrix (batched)
    X = np.arange(4*4, dtype=np.float64).reshape([1,4,4])
    # 1-mode matrix
    T = np.arange(4, dtype=np.float64).reshape([1,2,2])
    # expanded matrices
    T0 = np.array([tw.symplectic.expand(T[i], modes=[0], N=2) for i in range(len(T))])
    T1 = np.array([tw.symplectic.expand(T[i], modes=[1], N=2) for i in range(len(T))])
    # expected results
    res = numba_sparse_matmul(matrix1=T.reshape((1,1,1,2,2)).transpose((0,1,3,2,4)), matrix2=X.reshape((1,2,2,2,2)).transpose((0,1,3,2,4)), m1_modes=[0], m2_modes=[0,1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(T0, X[0]))

    res = numba_sparse_matmul(matrix1=X, matrix2=T, m1_modes=[1,2], m2_modes=[1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(X, T0))

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[1], m2_modes=[0,1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(T1, X))

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[0], m2_modes=[0,1], m1like_0=True, m2like_0=False)
    reduced = np.matmul(T0, X)[:,np.array([0,2])][:,:,np.array([0,2])]
    assert np.allclose(res, reduced)

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[1], m2_modes=[0,1], m1like_0=True, m2like_0=False)
    reduced = np.matmul(T1, X)[:,np.array([1,3])][:,:,np.array([1,3])]
    assert np.allclose(res, reduced)

    res = numba_sparse_matmul(matrix1=X, matrix2=T, m1_modes=[0,1], m2_modes=[0], m1like_0=False, m2like_0=True)
    reduced = np.matmul(X, T0)[:,np.array([0,2])][:,:,np.array([0,2])]
    assert np.allclose(res, reduced)


def test_numba_sparse_matmul_2_to_3_1():
    r"""tests that a 2-mode matrix is correctly composed with a 3-batched 1-mode matrix"""
    # 2-mode matrix (batched)
    X = np.arange(1*16).reshape([1,4,4])
    # 1-mode matrix
    T = np.arange(3*4).reshape([3,2,2])
    # expanded matrix
    T0 = np.array([tw.symplectic.expand(T[i], modes=[0], N=2) for i in range(len(T))])
    T1 = np.array([tw.symplectic.expand(T[i], modes=[1], N=2) for i in range(len(T))])
    # expected results
    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[0], m2_modes=[0,1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(T0, X))

    res = numba_sparse_matmul(matrix1=X, matrix2=T, m1_modes=[1,2], m2_modes=[1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(X, T0))

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[1], m2_modes=[0,1], m1like_0=False, m2like_0=False)
    assert np.allclose(res, np.matmul(T1, X))

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[0], m2_modes=[0,1], m1like_0=True, m2like_0=False)
    reduced = np.matmul(T0, X)[:,np.array([0,2])][:,:,np.array([0,2])]
    assert np.allclose(res, reduced)

    res = numba_sparse_matmul(matrix1=T, matrix2=X, m1_modes=[1], m2_modes=[0,1], m1like_0=True, m2like_0=False)
    reduced = np.matmul(T1, X)[:,np.array([1,3])][:,:,np.array([1,3])]
    assert np.allclose(res, reduced)

    res = numba_sparse_matmul(matrix1=X, matrix2=T, m1_modes=[0,1], m2_modes=[0], m1like_0=False, m2like_0=True)
    reduced = np.matmul(X, T0)[:,np.array([0,2])][:,:,np.array([0,2])]
    assert np.allclose(res, reduced)




