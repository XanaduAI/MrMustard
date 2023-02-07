# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""optimization tests"""

from hypothesis import given, strategies as st

import numpy as np
import tensorflow as tf

from thewalrus.symplectic import two_mode_squeezing

from mrmustard.lab.gates import (
    Sgate,
    Rgate,
    BSgate,
    S2gate,
    Ggate,
    Interferometer,
    RealInterferometer,
)
from mrmustard.lab.circuit import Circuit
from mrmustard.training import Optimizer, Parametrized
from mrmustard.lab.states import Vacuum, SqueezedVacuum
from mrmustard.physics import fidelity
from mrmustard.physics.gaussian import trace, von_neumann_entropy
from mrmustard import settings

from mrmustard.math import Math

math = Math()


@given(n=st.integers(0, 3))
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    settings.SEED = 42
    S = S2gate(
        r=abs(settings.rng.normal()),
        phi=settings.rng.normal(),
        r_trainable=True,
        phi_trainable=True,
    )

    def cost_fn():
        return -tf.abs((Vacuum(2) >> S[0, 1]).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[S], max_steps=300)

    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(-cost_fn(), expected, atol=1e-5)


@given(i=st.integers(1, 5), k=st.integers(1, 5))
def test_hong_ou_mandel_optimizer(i, k):
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
    This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
    see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
    which lacks a square root in the right hand side.
    """
    settings.SEED = 42
    r = np.arcsinh(1.0)
    s2_0, s2_1, bs = (
        S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1],
        S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3],
        BSgate(
            theta=np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.rng.normal(),
            phi=settings.rng.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )[1, 2],
    )
    circ = Circuit([s2_0, s2_1, bs])
    state_in = Vacuum(num_modes=4)
    cutoff = 1 + i + k

    def cost_fn():
        return tf.abs((state_in >> circ).ket(cutoffs=[cutoff] * 4)[i, 1, i + k - 1, k]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.cos(bs.theta.value) ** 2, k / (i + k), atol=1e-2)


def test_learning_two_mode_squeezing():
    """Finding the optimal beamsplitter transmission to make a pair of single photons"""
    settings.SEED = 42
    ops = [
        Sgate(
            r=abs(settings.rng.normal(size=(2))),
            phi=settings.rng.normal(size=(2)),
            r_trainable=True,
            phi_trainable=True,
        ),
        BSgate(
            theta=settings.rng.normal(),
            phi=settings.rng.normal(),
            theta_trainable=True,
            phi_trainable=True,
        ),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.05)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_two_mode_Ggate():
    """Finding the optimal Ggate to make a pair of single photons"""
    settings.SEED = 42
    G = Ggate(num_modes=2, symplectic_trainable=True)

    def cost_fn():
        amps = (Vacuum(2) >> G).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[G], max_steps=500)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-4)


def test_learning_two_mode_Interferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    settings.SEED = 42
    ops = [
        Sgate(
            r=settings.rng.normal(size=(2)) ** 2,
            phi=settings.rng.normal(size=(2)),
            r_trainable=True,
            phi_trainable=True,
        ),
        Interferometer(num_modes=2, orthogonal_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_two_mode_RealInterferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    settings.SEED = 2
    ops = [
        Sgate(
            r=settings.rng.normal(size=(2)) ** 2,
            phi=settings.rng.normal(size=(2)),
            r_trainable=True,
            phi_trainable=True,
        ),
        RealInterferometer(num_modes=2, orthogonal_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[2, 2])
        return -tf.abs(amps[1, 1]) ** 2 + tf.abs(amps[0, 1]) ** 2

    opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_four_mode_Interferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    settings.SEED = 4
    ops = [
        Sgate(
            r=settings.rng.uniform(size=4),
            phi=settings.rng.normal(size=4),
            r_trainable=True,
            phi_trainable=True,
        ),
        Interferometer(num_modes=4, orthogonal_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=4)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[3, 3, 3, 3])
        return (
            -math.abs(
                math.sum(
                    amps[1, 1]
                    * np.array([[0, 0, 1 / np.sqrt(2)], [0, 0, 0], [1 / np.sqrt(2), 0, 0]])
                )
            )
            ** 2
        )

    opt = Optimizer(orthogonal_lr=0.05)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)


def test_learning_four_mode_RealInterferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    settings.SEED = 6
    ops = [
        Sgate(
            r=settings.rng.uniform(size=4),
            phi=settings.rng.normal(size=4),
            r_trainable=True,
            phi_trainable=True,
        ),
        RealInterferometer(num_modes=4, orthogonal_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=4)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[3, 3, 3, 3])
        return (
            -math.abs(
                math.sum(
                    amps[1, 1]
                    * np.array([[0, 0, 1 / np.sqrt(2)], [0, 0, 0], [1 / np.sqrt(2), 0, 0]])
                )
            )
            ** 2
        )

    opt = Optimizer()

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=400)
    assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)


def test_squeezing_hong_ou_mandel_optimizer():
    """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
    see https://www.pnas.org/content/117/52/33107/tab-article-info
    """
    settings.SEED = 42
    r = np.arcsinh(1.0)

    S_01 = S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1]
    S_23 = S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3]
    S_12 = S2gate(r=1.0, phi=settings.rng.normal(), r_trainable=True, phi_trainable=True)[1, 2]

    circ = Circuit([S_01, S_23, S_12])

    def cost_fn():
        return tf.abs((Vacuum(4) >> circ).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.sinh(S_12.r.value) ** 2, 1, atol=1e-2)


def test_parameter_passthrough():
    """Same as the test above, but with param passthrough"""
    settings.SEED = 42
    r = np.arcsinh(1.0)
    par = Parametrized(
        r=math.new_variable(r, (0.0, None), "r"),
        phi=math.new_variable(settings.rng.normal(), (None, None), "phi"),
    )
    ops = [
        S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1],
        S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3],
        S2gate(r=par.r.value, phi=par.phi.value)[1, 2],
    ]
    circ = Circuit(ops)

    def cost_fn():
        return tf.abs((Vacuum(4) >> circ).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[par], max_steps=300)
    assert np.allclose(np.sinh(par.r.value) ** 2, 1, atol=1e-2)


def test_making_thermal_state_as_one_half_two_mode_squeezed_vacuum():
    """Optimizes a Ggate on two modes so as to prepare a state with the same entropy
    and mean photon number as a thermal state"""
    settings.SEED = 42
    S_init = two_mode_squeezing(np.arcsinh(1.0), 0.0)

    nbar = 1.4

    def thermal_entropy(nbar):
        return -(nbar * np.log((nbar) / (1 + nbar)) - np.log(1 + nbar))

    G = Ggate(num_modes=2, symplectic_trainable=True, symplectic=S_init)

    def cost_fn():
        state = Vacuum(2) >> G
        cov1, _ = trace(state.cov, state.means, [0])
        mean1 = state.number_means[0]
        mean2 = state.number_means[1]
        entropy = von_neumann_entropy(cov1, settings.HBAR)
        S = thermal_entropy(nbar)
        return (mean1 - nbar) ** 2 + (entropy - S) ** 2 + (mean2 - nbar) ** 2

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=50)
    S = G.symplectic.value.numpy()
    cov = S @ S.T
    assert np.allclose(cov, two_mode_squeezing(2 * np.arcsinh(np.sqrt(nbar)), 0.0))


def test_opt_backend_param():
    """Test the optimization of a backend parameter defined outside a gate."""
    # rotated displaced squeezed state
    settings.SEED = 42
    rotation_angle = np.pi / 2
    target_state = SqueezedVacuum(r=1.0, phi=rotation_angle)

    # angle of rotation gate
    r_angle = math.new_variable(0, bounds=(0, np.pi), name="r_angle")
    # trainable squeezing
    S = Sgate(r=0.1, phi=0, r_trainable=True, phi_trainable=False)

    def cost_fn_sympl():
        state_out = Vacuum(1) >> S >> Rgate(angle=r_angle)
        return 1 - fidelity(state_out, target_state)

    opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.1)
    opt.minimize(cost_fn_sympl, by_optimizing=[S, r_angle])

    assert np.allclose(r_angle.numpy(), rotation_angle / 2, atol=1e-2)
