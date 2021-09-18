from hypothesis import strategies as st, given
from mrmustard import DisplacedSqueezed
from mrmustard.experimental import XPTensor
import numpy as np
from mrmustard.tests.random import random_pure_state


# TODO: replace fixed states with random ones using hypothesis


def test_from_xxpp_xpxp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    xxpp_cov, xxpp_means = state.cov, state.means
    
    xp1 = XPTensor.from_xxpp(xxpp_cov, like_1=True)
    xpxp_cov = np.reshape(np.transpose(np.reshape(xxpp_cov, (2,3,2,3)), (1,0,3,2)), (6,6))
    xp2 = XPTensor.from_xpxp(xpxp_cov, like_1=True)
    assert np.allclose(xp1.tensor, xp2.tensor)
    
    xp1 = XPTensor.from_xxpp(xxpp_means, like_1=True)
    xpxp_means = np.reshape(np.transpose(np.reshape(xxpp_means, (2,3)), (1,0)), (6,))
    xp2 = XPTensor.from_xpxp(xpxp_means, like_1=True)
    assert np.allclose(xp1.tensor, xp2.tensor)

def test_from_xpxp_to_xpxp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    xxpp_cov, xxpp_means = state.cov, state.means
    xpxp_cov = np.reshape(np.transpose(np.reshape(xxpp_cov, (2,3,2,3)), (1,0,3,2)), (6,6))
    xp1 = XPTensor.from_xpxp(xpxp_cov, like_1=True)
    assert np.allclose(xp1.to_xpxp(), xpxp_cov)

def test_from_xxpp_to_xxpp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    xxpp_cov = state.cov
    xp1 = XPTensor.from_xxpp(xxpp_cov, like_1=True)
    assert np.allclose(xp1.to_xxpp(), xxpp_cov)


def test_xxpp_to_xpxp_to_xxpp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    xxpp_cov, xxpp_means = state.cov, state.means
    xpxp_cov = np.reshape(np.transpose(np.reshape(xxpp_cov, (2,3,2,3)), (1,0,3,2)), (6,6))
    xp1 = XPTensor.from_xpxp(xpxp_cov, like_1=True)
    xp2 = XPTensor.from_xxpp(xp1.to_xxpp(), like_1=True)
    xp3 = XPTensor.from_xpxp(xp2.to_xpxp(), like_1=True)
    assert np.allclose(xxpp_cov, xp3.to_xxpp())

def test_xxpp_to_xpxp():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2])
    xpxp_cov = XPTensor.from_xxpp(state.cov, like_1=True).to_xpxp()
    expected_cov = np.reshape(np.transpose(np.reshape(state.cov, (2,3,2,3)), (1,0,3,2)), (6,6))
    assert np.allclose(xpxp_cov, expected_cov)

def test_matmul_same_modes():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2]) 
    cov, means = state.cov, state.means
    xp1 = XPTensor.from_xxpp(cov, like_1=True)
    expected = cov @ cov
    assert np.allclose((xp1 @ xp1).to_xxpp(), expected)

def test_matmul_different_modes():
    state = DisplacedSqueezed(r=[0.5, 0.5, 0.5], phi=[0.4, 0.4,0.4], x=[0.3, 0.3,0.3], y=[0.2, 0.2,0.2]) 
    cov = np.reshape(np.transpose(np.reshape(state.cov, (2,3,2,3)), (1,0,3,2)), (6,6))  #  in xpxp order
    xp1 = XPTensor.from_xpxp(cov, modes=[0,1,2], like_1=True)
    xp2 = XPTensor.from_xpxp(cov, modes=[1,2,3], like_1=True)
    cov1 = np.block([[cov, np.zeros((6,2))], [np.zeros((2,6)), np.eye(2)]])  # add one extra empty mode at the end
    cov2 = np.block([[np.eye(2), np.zeros((2,6))], [np.zeros((6,2)), cov]])  # add one extra empty mode at the beginning
    prod = xp1 @ xp2
    np.allclose(prod.to_xpxp(), cov1 @ cov2)

