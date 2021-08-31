from mrmustard import Coherent
from mrmustard.experimental import XPTensor
import numpy as np


def test_xp_creation():
    xp = XPTensor.from_xxpp(Coherent([0.5, 0.5], [0.4, 0.4]).cov, modes=[1, 2])


# def test_xp_multiplication():
#     a = XPTensor.from_xxpp(modes = [1,2], xxpp_matrix = Coherent([0.5, 0.5], [0.4, 0.4]).cov)
#     b = XPTensor.from_xxpp(modes = [1,2], xxpp_matrix = Coherent([0.5, 0.5], [0.4, 0.4]).cov)
#     c = a * b
#     assert np.allclose(c._tensor, np.tensordot(a._tensor, b._tensor, axes = ([2,3],[0,1])))
