import numpy as np
import tensorflow as tf

from mrmustard.tf import (
    Dgate,
    Sgate,
    LossChannel,
    BSgate,
    Ggate,
    Optimizer,
    Circuit,
    S2gate,
    Rgate,
    Vacuum,
)

import pytest


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    tf.random.set_seed(137)
    circ = Circuit()
    circ.append(S2gate(modes=[0, 1]))

    state_in = Vacuum(num_modes=2)

    def loss_fn():
        return -tf.abs(circ(state_in).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    circ = opt.minimize(circ, loss_fn, max_steps=0)
    prob = np.abs(circ(state_in).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2
    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(prob, expected, atol=1e-4)


def test_hong_ou_mandel_optimizer():
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip"""
    tf.random.set_seed(137)
    circ = Circuit()  # emtpy circuit with vacuum input state
    r = np.arcsinh(1.0)
    circ.append(S2gate(modes=[0, 1], r=r, r_trainable=False))
    circ.append(S2gate(modes=[2, 3], r=r, r_trainable=False))
    circ.append(BSgate(modes=[1, 2]))

    state_in = Vacuum(num_modes=4)

    def loss_fn():
        return tf.abs(circ(state_in).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.005)
    circ = opt.minimize(circ, loss_fn, max_steps=0)
    assert np.allclose(np.cos(circ.euclidean_parameters[2]) ** 2, 0.5, atol=1e-2)


def test_learning_two_mode_squeezing():
    """Finding the optimal beamsplitter transmission to make a pair of single photons"""
    tf.random.set_seed(137)
    circ = Circuit()
    circ.append(Sgate(modes=[0]))
    circ.append(Sgate(modes=[1]))
    circ.append(BSgate(modes=[0, 1]))
    tf.random.set_seed(20)

    state_in = Vacuum(num_modes=2)

    def loss_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.05)

    circ = opt.minimize(circ, loss_fn, max_steps=1000)
    assert np.allclose(-loss_fn(), 0.25, atol=2e-3)


def test_learning_two_mode_Ggate():
    """Finding the optimal Ggate to make a pair of single photons"""
    tf.random.set_seed(137)
    circ = Circuit()  # emtpy circuit with vacuum input state
    circ.append(Ggate(modes=[0, 1], displacement_trainable=False, displacement=[0, 0, 0, 0]))
    tf.random.set_seed(20)

    state_in = Vacuum(num_modes=2)

    def loss_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(symplectic_lr=0.1)

    circ = opt.minimize(circ, loss_fn, max_steps=1000)
    assert np.allclose(-loss_fn(), 0.25, atol=2e-3)
