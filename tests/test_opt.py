import numpy as np
import tensorflow as tf

from mrmustard.tf import Dgate, Sgate, LossChannel, BSgate, Ggate, Optimizer, Circuit, S2gate, Rgate

import pytest


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    circ = Circuit(num_modes=2)
    circ.add_gate(S2gate(modes=[0, 1]))

    def loss_fn():
        return -tf.abs(circ.fock_output(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    circ = opt.minimize(circ, loss_fn, max_steps=0)
    prob = np.abs(circ.fock_output(cutoffs=[n + 1, n + 1]))[n, n] ** 2
    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(prob, expected, atol=1e-4)


def test_hong_ou_mandel_optimizer():
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip"""
    tf.random.set_seed(137)
    circ = Circuit(num_modes=4)  # emtpy circuit with vacuum input state
    r = np.arcsinh(1.0)
    circ.add_gate(S2gate(modes=[0, 1], r=r, r_trainable=False))
    circ.add_gate(S2gate(modes=[2, 3], r=r, r_trainable=False))
    circ.add_gate(BSgate(modes=[1, 2]))

    def loss_fn():
        return tf.abs(circ.fock_output(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.005)
    circ = opt.minimize(circ, loss_fn, max_steps=0)
    assert np.allclose(np.cos(circ.euclidean_parameters[2]) ** 2, 0.5, atol=2e-3)
