from hypothesis import strategies as st, given, assume
from hypothesis.extra.numpy import arrays
import pytest
from mrmustard.lab.states import DisplacedSqueezed
from mrmustard.utils.xptensor import XPVector, XPMatrix
import numpy as np
from tests.random import pure_state, even_vector




@given(even_vector())
def test_create_XPVector(even_vector):
    XPVector(even_vector)

