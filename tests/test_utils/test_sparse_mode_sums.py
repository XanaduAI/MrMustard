from hypothesis import settings, given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import tensorflow as tf

from mrmustard.math.numba import *
import thewalrus as tw

from tests.random import real_array as array



@given(v1=array(1,1,2), v2=array(1,1,2))
def test_numba_sparse_vec_add_newvec1(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 1-mode vector with different modes"""
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,), modes2=(1,))
    expected = np.array([[[v1[0,0,0], v1[0,0,1]], [v2[0,0,0], v2[0,0,1]]]])
    assert np.allclose(expected, summed)


@given(v1=array(1,2,2), v2=array(1,1,2))
def test_numba_sparse_vec_add_inplace2(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 2-mode vector with mode overlap"""
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,1), modes2=(0,))
    expected = np.array([[[v1[0,0,0]+v2[0,0,0], v1[0,0,1]+v2[0,0,1]], [v1[0,1,0], v1[0,1,1]]]])
    assert np.allclose(expected, summed)


@given(v1=array(1,3,2), v2=array(1,1,2))
def test_numba_sparse_vec_add_inplace3(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 3-mode vector"""
    expected = np.array([[[v1[0,0,0]+v2[0,0,0], v1[0,0,1]+v2[0,0,1]], [v1[0,1,0], v1[0,1,1]], [v1[0,2,0], v1[0,2,1]]]])
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,1,2), modes2=(0,))

    expected = np.array([[[v1[0,0,0], v1[0,0,1]], [v1[0,1,0]+v2[0,0,0], v1[0,1,1]+v2[0,0,1]], [v1[0,2,0], v1[0,2,1]]]])
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,1,2), modes2=(1,))

    expected = np.array([[[v1[0,0,0], v1[0,0,1]], [v1[0,1,0], v1[0,1,1]], [v1[0,2,0]+v2[0,0,0], v1[0,2,1]+v2[0,0,1]]]])
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,1,2), modes2=(2,))
    assert np.allclose(expected, summed)


@given(m1=array(1,2,2,2,2), m2=array(1,1,1,2,2))
def test_numba_sparse_add(m1,m2):
    r"""tests that a 1-mode matrix is correctly added to a 2-mode matrix"""
    summed = numba_mat_add(mat1=m1.copy(), mat2=m2, modes1=(0,1), modes2=(0,), m1like_0=True, m2like_0=True)
    expected = m1
    expected[0,0,0] += m2[0,0,0]

    summed = numba_mat_add(mat1=m1.copy(), mat2=m2, modes1=(0,1), modes2=(1,), m1like_0=True, m2like_0=True)
    expected = m1
    expected[0,1,1] += m2[0,0,0]
    assert np.allclose(expected, summed)


@given(v1=array(2,2,2), v2=array(2,1,2))
def test_numba_sparse_add_vec_batched(v1, v2):
    r"""tests that a 1-mode vector is correctly added to a 2-mode vector"""
    summed = numba_vec_add(vec1=v1.copy(), vec2=v2, modes1=(0,1), modes2=(0,))
    expected = np.array([[[v1[0,0,0]+v2[0,0,0], v1[0,0,1]+v2[0,0,1]], [v1[0,1,0], v1[0,1,1]]],
                         [[v1[0,0,0]+v2[1,0,0], v1[0,0,1]+v2[1,0,1]], [v1[0,1,0], v1[0,1,1]]],
                         [[v1[1,0,0]+v2[0,0,0], v1[1,0,1]+v2[0,0,1]], [v1[1,1,0], v1[1,1,1]]],
                         [[v1[1,0,0]+v2[1,0,0], v1[1,0,1]+v2[1,0,1]], [v1[1,1,0], v1[1,1,1]]]])
    assert np.allclose(expected, summed)


@given(m1=array(2,2,2,2,2), m2=array(2,1,1,2,2))
def test_numba_sparse_add_mat_batched(m1, m2):
    r"""tests that a 1-mode matrix is correctly added to a 2-mode matrix"""
    summed = numba_mat_add(mat1=m1.copy(), mat2=m2, modes1=(0,1), modes2=(0,), m1like_0=True, m2like_0=True)
    expected = np.tile(m1, (2,1,1,1,1))
    expected[0,0,0] += m2[0,0,0]
    expected[1,0,0] += m2[1,0,0]
    expected[2,0,0] += m2[0,0,0]
    expected[3,0,0] += m2[1,0,0]
    assert np.allclose(expected, summed)

    summed = numba_mat_add(mat1=m1.copy(), mat2=m2, modes1=(0,1), modes2=(1,), m1like_0=True, m2like_0=True)
    expected = np.tile(m1, (2,1,1,1,1))
    expected[0,1,1] += m2[0,0,0]
    expected[1,1,1] += m2[1,0,0]
    expected[2,1,1] += m2[0,0,0]
    expected[3,1,1] += m2[1,0,0]
    assert np.allclose(expected, summed)


@given(m1=array(1,2,2,2,2), m2=array(1,1,1,2,2))
def test_numba_sparse_add_mat_batched_m2like_1(m1, m2):
    r"""tests that a 1-mode matrix is correctly added to a 2-mode matrix"""
    summed = numba_mat_add(mat1=m1.copy(), mat2=m2, modes1=(0,1), modes2=(0,), m1like_0=True, m2like_0=False)
    expected = m1
    expected[0,0,0] += m2[0,0,0]
    expected[0,1,1] += np.identity(2)
    assert np.allclose(expected, summed)
