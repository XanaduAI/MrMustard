from hypothesis import strategies as st, given
from mrmustard import DisplacedSqueezed
from mrmustard.experimental import XPTensor
import numpy as np
from mrmustard.tests.random import random_pure_state


# TODO: replace fixed states with random ones using hypothesis


def test_creation():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    cov, means = state.cov, state.means
    
    xp1 = XPTensor.from_xxpp(cov, like_1=True)
    xp2 = XPTensor(cov, like_1=True)
    assert np.allclose(xp1.tensor, xp2.tensor)
    
    xp1 = XPTensor.from_xxpp(means, like_1=True)
    xp2 = XPTensor(means, like_1=True)
    assert np.allclose(xp1.tensor, xp2.tensor)


def test_creation_from_rank4():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    cov, means = state.cov, state.means
    cov = np.reshape(cov, (2, 3, 2, 3))
    means = np.reshape(means, (2, 3))

    xp = XPTensor.from_tensor(cov, like_1=True)
    xp = XPTensor.from_tensor(means, like_0=True)

def xxpp_to_xpxp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    expected_cov = np.reshape(np.transpose(np.reshape(state.cov, (2,3,2,3)), (1,0,3,2)), (6,6))
    cov = XPTensor(state.cov, like_1=True).to_xpxp()
    assert np.allclose(cov, expected_cov)

def test_matmul_same_modes():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2]) 
    cov, means = state.cov, state.means
    xp1 = XPTensor(cov, like_1=True)
    expected = cov @ cov
    assert np.allclose((xp1 @ xp1).to_xxpp(), expected)

def test_matmul_different_modes():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2]) 
    cov = np.reshape(np.transpose(np.reshape(state.cov, (2,3,2,3)), (1,0,3,2)), (6,6))  #  in xpxp order
    xp1 = XPTensor.from_xpxp(cov, modes=[0,1,2], like_1=True)
    xp2 = XPTensor.from_xpxp(cov, modes=[1,2,3], like_1=True)
    cov1 = np.block([[cov, np.zeros((6,2))], [np.zeros((2,6)), np.eye(2)]])  # add one extra empty mode at the end
    cov2 = np.block([[np.eye(2), np.zeros((2,6))], [np.zeros((6,2)), cov]])  # add one extra empty mode at the beginning
    assert np.allclose((xp1 @ xp1).to_xpxp(), cov1 @ cov2)

