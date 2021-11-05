# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hypothesis import settings, given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import tensorflow as tf

from mrmustard.lab.gates import Sgate, BSgate, S2gate, Ggate, Interferometer
from mrmustard.lab.circuit import Circuit
from mrmustard.utils.training import Optimizer
from mrmustard.lab.states import Vacuum


@given(n=st.integers(0, 3))
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    tf.random.set_seed(137)
    S = S2gate(
        modes=[0, 1],
        r=abs(np.random.normal()),
        phi=np.random.normal(),
        r_trainable=True,
        phi_trainable=True,
    )

    def cost_fn():
        return -tf.abs(S(Vacuum(2)).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[S], max_steps=300)

    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(-cost_fn(), expected, atol=1e-3)

@given(i=st.integers(1, 5), k=st.integers(1, 5))
def test_hong_ou_mandel_optimizer(i, k):
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip"""
    tf.random.set_seed(137)
    circ = Circuit()
    r = np.arcsinh(1.0)
    circ.append(S2gate(modes=[0, 1], r=r, phi=0.0, phi_trainable=True))
    circ.append(S2gate(modes=[2, 3], r=r, phi=0.0, phi_trainable=True))
    circ.append(
        BSgate(
            modes=[1, 2],
            theta=np.arccos(np.sqrt(k/(i+k))) + 0.1 * np.random.normal(),
            phi=np.random.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )
    )
    state_in = Vacuum(num_modes=4)
    cutoff = 1 + i + k
    def cost_fn():
        return tf.abs(circ(state_in).ket(cutoffs=[cutoff, cutoff, cutoff, cutoff])[i, 1, i + k - 1, k]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.cos(circ.trainable_parameters["euclidean"][2]) ** 2, k / (i+k), atol=1e-2)


def test_squeezing_hong_ou_mandel_optimizer():
    """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
    see https://www.pnas.org/content/117/52/33107/tab-article-info
    """
    tf.random.set_seed(137)
    circ = Circuit()
    r = np.arcsinh(1.0)
    circ.append(S2gate(modes=[0, 1], r=r, phi=0.0, phi_trainable=True))
    circ.append(S2gate(modes=[2, 3], r=r, phi=0.0, phi_trainable=True))
    circ.append(S2gate(modes=[1, 2], r=1.0, phi=np.random.normal(), r_trainable=True, phi_trainable=True))
    state_in = Vacuum(num_modes=4)

    def cost_fn():
        return tf.abs(circ(state_in).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.sinh(circ.trainable_parameters["euclidean"][2]) ** 2, 1, atol=1e-2)


def test_learning_two_mode_squeezing():
    """Finding the optimal beamsplitter transmission to make a pair of single photons"""
    tf.random.set_seed(137)
    circ = Circuit()
    circ.append(
        Sgate(
            modes=[0, 1],
            r=abs(np.random.normal(size=(2))),
            phi=np.random.normal(size=(2)),
            r_trainable=True,
            phi_trainable=True,
        )
    )
    circ.append(
        BSgate(
            modes=[0, 1],
            theta=np.random.normal(),
            phi=np.random.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )
    )
    tf.random.set_seed(20)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.05)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-3)


def test_learning_two_mode_Ggate():
    """Finding the optimal Ggate to make a pair of single photons"""
    tf.random.set_seed(137)
    G = Ggate(num_modes=2, symplectic_trainable=True)
    tf.random.set_seed(20)

    def cost_fn():
        amps = G(Vacuum(2)).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[G], max_steps=2000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-3)


def test_learning_two_mode_Interferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    np.random.seed(11)
    circ = Circuit()  # emtpy circuit with vacuum input state
    circ.append(
        Sgate(
            modes=[0, 1],
            r=np.random.normal(size=(2)) ** 2,
            phi=np.random.normal(size=(2)),
            r_trainable=True,
            phi_trainable=True,
        )
    )
    circ.append(Interferometer(num_modes=2, orthogonal_trainable=True))
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-3)


def test_learning_four_mode_Interferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    np.random.seed(11)
    circ = Circuit()
    circ.append(
        Sgate(
            r=np.random.uniform(size=4),
            phi=np.random.normal(size=4),
            r_trainable=True,
            phi_trainable=True,
        )
    )
    circ.append(Interferometer(num_modes=4, orthogonal_trainable=True))
    state_in = Vacuum(num_modes=4)

    def cost_fn():
        amps = circ(state_in).ket(cutoffs=[3, 3, 3, 3])
        return -tf.abs(tf.reduce_sum(amps[1, 1] * np.array([[0, 0, 1 / np.sqrt(2)], [0, 0, 0], [1 / np.sqrt(2), 0, 0]]))) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.0625, atol=1e-3)


def test_squeezing_hong_ou_mandel_optimizer():
    """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
    see https://www.pnas.org/content/117/52/33107/tab-article-info
    """
    tf.random.set_seed(137)
    circ = Circuit()
    r = np.arcsinh(1.0)
    circ.append(S2gate(modes=[0, 1], r=r, phi=0.0, phi_trainable=True))
    circ.append(S2gate(modes=[2, 3], r=r, phi=0.0, phi_trainable=True))
    circ.append(S2gate(modes=[1, 2], r=1.0, phi=np.random.normal(), r_trainable=True, phi_trainable=True))
    state_in = Vacuum(num_modes=4)

    def cost_fn():
        return tf.abs(circ(state_in).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.sinh(circ.trainable_parameters["euclidean"][2]) ** 2, 1, atol=1e-2)
