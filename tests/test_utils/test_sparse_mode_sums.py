from hypothesis import settings, given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import tensorflow as tf

from mrmustard.math.numba import *
import thewalrus as tw

from tests.random import vector



@given(v1=vector(2), v2=vector(2))
def test_numba_sparse_vec_add_newvec1(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 1-mode vector with different modes"""
    summed = numba_sparse_vec_add(vec1=v1.copy()[None,:], vec2=v2[None,:], modes1=[0], modes2=[1])
    expected = np.array([[v1[0], v2[0], v1[1], v2[1]]])
    assert np.allclose(expected, summed)


@given(v1=vector(4), v2=vector(2))
def test_numba_sparse_vec_add_inplace2(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 2-mode vector"""
    summed = numba_sparse_vec_add(vec1=v1.copy()[None,:], vec2=v2[None,:], modes1=[0,1], modes2=[0])
    expected = np.array([[v1[0]+v2[0], v1[1], v1[2]+v2[1], v1[3]]])
    assert np.allclose(expected, summed)


@given(v1=vector(6), v2=vector(2))
def test_numba_sparse_vec_add_inplace3(v1,v2):
    r"""tests that a 1-mode vector is correctly added to a 3-mode vector"""
    expected = np.array([[v1[0]+v2[0], v1[1], v1[2], v1[3]+v2[1], v1[4], v1[5]]])
    summed = numba_sparse_vec_add(vec1=v1.copy()[None,:], vec2=v2[None,:], modes1=[0,1,2], modes2=[0])

    expected = np.array([[v1[0], v1[1]+v2[0], v1[2], v1[3], v1[4]+v2[1], v1[5]]])
    summed = numba_sparse_vec_add(vec1=v1.copy()[None,:], vec2=v2[None,:], modes1=[0,1,2], modes2=[1])

    expected = np.array([[v1[0], v1[1], v1[2]+v2[0], v1[3], v1[4], v1[5]+v2[1]]])
    summed = numba_sparse_vec_add(vec1=v1.copy()[None,:], vec2=v2[None,:], modes1=[0,1,2], modes2=[2])
    assert np.allclose(expected, summed)


# @given(m1=matrix((4,4)), m2=matrix((2,2)))
# def test_numba_sparse_add(m1,m2):
#     r"""tests that a 1-mode matrix is correctly added to a 2-mode matrix"""
#     pass