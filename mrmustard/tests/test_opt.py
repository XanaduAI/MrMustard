import pytest
import numpy as np
import tensorflow as tf

from mrmustard import Sgate, BSgate, S2gate, Ggate, Interferometer
from mrmustard import Circuit, Optimizer
from mrmustard import Vacuum


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    tf.random.set_seed(137)
    S = S2gate(modes=[0, 1], r=abs(np.random.normal()), phi=np.random.normal())

    def cost_fn():
        return -tf.abs(S(Vacuum(2)).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[S], max_steps=300)
    
    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(-cost_fn(), expected, atol=1e-4)


def test_hong_ou_mandel_optimizer():
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip"""
    tf.random.set_seed(137)
    circ = Circuit()
    r = np.arcsinh(1.0)
    circ.append(S2gate(modes=[0, 1], r=r, r_trainable=False, phi=0.0))
    circ.append(S2gate(modes=[2, 3], r=r, r_trainable=False, phi=0.0))
    circ.append(BSgate(modes=[1, 2], theta=np.random.normal(), phi=np.random.normal()))

    state_in = Vacuum(num_modes=4)

    def cost_fn():
        return tf.abs(circ(state_in).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.cos(circ.trainable_parameters['euclidean'][2]) ** 2, 0.5, atol=1e-2)


def test_learning_two_mode_squeezing():
    """Finding the optimal beamsplitter transmission to make a pair of single photons"""
    tf.random.set_seed(137)
    circ = Circuit()
    circ.append(Sgate(modes=[0, 1], r=np.random.normal(size=(2)), phi=np.random.normal(size=(2))))
    circ.append(BSgate(modes=[0, 1], theta=np.random.normal(), phi=np.random.normal()))
    tf.random.set_seed(20)

    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.05)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=2e-3)


def test_learning_two_mode_Ggate():
    """Finding the optimal Ggate to make a pair of single photons"""
    tf.random.set_seed(137)
    G = Ggate(modes=[0, 1], displacement_trainable=False, displacement=[0, 0, 0, 0])
    tf.random.set_seed(20)

    def cost_fn():
        amps = G(Vacuum(2)).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[G], max_steps=2000)
    assert np.allclose(-cost_fn(), 0.25, atol=2e-3)


def test_learning_two_mode_Interferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    np.random.seed(11)
    circ = Circuit()  # emtpy circuit with vacuum input state
    circ.append(Sgate(modes=[0, 1], r=np.random.normal(size=(2)) ** 2, phi=np.random.normal(size=(2))))
    circ.append(Interferometer(modes=[0, 1]))

    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=2e-3)


def test_learning_four_mode_Interferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    np.random.seed(11)
    circ = Circuit()
    circ.append(Sgate(modes=[0, 1, 2, 3], r=np.random.uniform(size=4), phi=np.random.normal(size=4)))
    circ.append(Interferometer(modes=[0, 1, 2, 3]))

    state_in = Vacuum(num_modes=4)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[3, 3, 3, 3])
        return -tf.abs(tf.reduce_sum(amps[1, 1] * np.array([[0, 0, 1 / np.sqrt(2)], [0, 0, 0], [1 / np.sqrt(2), 0, 0]]))) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.0625, atol=2e-3)
