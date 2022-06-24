from hypothesis import strategies as st, given, assume
from hypothesis.extra.numpy import arrays
import pytest
from mrmustard.lab.states import DisplacedSqueezed
from mrmustard.utils.xptensor import XPVector, XPMatrix
import numpy as np
from tests.random import pure_state, even_vector, even_matrix



@given(even_vector())
def test_create_XPVector(even_vector):
    XPVector.from_xpxp(even_vector)

@given(even_matrix())
def test_create_XPMatrix(even_matrix):
    XPMatrix.from_xxpp(even_matrix, like_0 = True)
    XPMatrix.from_xxpp(even_matrix, like_0 = False)

def test_XPVector_inner_product_mode0():
    V1 = XPVector.from_xpxp(np.array([[1,2]]))
    V2 = XPVector.from_xpxp(np.array([[1,2,3,4]]))
    V3 = XPVector.from_xpxp(np.array([[1,2,3,4,5,6]]))
    assert np.isclose(V1 @ V1, 5)
    assert np.isclose(V1 @ V2, 5)
    assert np.isclose(V1 @ V3, 5)

def test_XPVector_inner_product_mode1_nobatch():
    V1 = XPVector.from_xpxp(np.array([1,2]), modes=[1])
    V2 = XPVector.from_xpxp(np.array([1,2,3,4]))
    V3 = XPVector.from_xpxp(np.array([1,2,3,4,5,6]))
    assert np.isclose(V1 @ V1, 5)
    assert np.isclose(V1 @ V2, 11)
    assert np.isclose(V1 @ V3, 11)

