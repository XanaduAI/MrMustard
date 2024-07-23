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

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from thewalrus.symplectic import two_mode_squeezing

from mrmustard import math, settings
from mrmustard.lab.circuit import Circuit
from mrmustard.lab.gates import (
    BSgate,
    Dgate,
    Ggate,
    Interferometer,
    RealInterferometer,
    Rgate,
    S2gate,
    Sgate,
)
from mrmustard.lab.states import (
    DisplacedSqueezed,
    Fock,
    Gaussian,
    SqueezedVacuum,
    Vacuum,
)
from mrmustard.math.parameters import Variable, update_euclidean
from mrmustard.physics import fidelity
from mrmustard.physics.gaussian import trace, von_neumann_entropy
from mrmustard.training import Optimizer
from mrmustard.training.callbacks import Callback

from ..conftest import skip_np


@given(n=st.integers(0, 3))
def test_S2gate_coincidence_prob(n):
    """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
    skip_np()

    settings.SEED = 40
    S = S2gate(
        r=abs(settings.rng.normal(loc=1.0, scale=0.1)),
        r_trainable=True,
    )

    def cost_fn():
        return -math.abs((Vacuum(2) >> S[0, 1]).ket(cutoffs=[n + 1, n + 1])[n, n]) ** 2

    def cb(optimizer, cost, trainables, **kwargs):  # pylint: disable=unused-argument
        return {
            "cost": cost,
            "lr": optimizer.learning_rate[update_euclidean],
            "num_trainables": len(trainables),
        }

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(cost_fn, by_optimizing=[S], max_steps=300, callbacks=cb)

    expected = 1 / (n + 1) * (n / (n + 1)) ** n
    assert np.allclose(-cost_fn(), expected, atol=1e-5)

    cb_result = opt.callback_history.get("cb")
    assert {res["num_trainables"] for res in cb_result} == {1}
    assert {res["lr"] for res in cb_result} == {0.01}
    assert [res["cost"] for res in cb_result] == opt.opt_history[1:]


@given(i=st.integers(1, 5), k=st.integers(1, 5))
def test_hong_ou_mandel_optimizer(i, k):
    """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
    This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
    see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
    which lacks a square root in the right hand side.
    """
    skip_np()

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
        return (
            math.abs((state_in >> circ).ket(cutoffs=[cutoff] * 4)[i, 1, i + k - 1, k])
            ** 2
        )

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(
        cost_fn,
        by_optimizing=[circ],
        max_steps=300,
        callbacks=[Callback(tag="null_cb", steps_per_call=3)],
    )
    assert np.allclose(np.cos(bs.theta.value) ** 2, k / (i + k), atol=1e-2)
    assert "null_cb" in opt.callback_history
    assert len(opt.callback_history["null_cb"]) == (len(opt.opt_history) - 1) // 3


def test_learning_two_mode_squeezing():
    """Finding the optimal beamsplitter transmission to make a pair of single photons"""
    skip_np()

    settings.SEED = 42
    ops = [
        Sgate(
            r=abs(settings.rng.normal(size=2)),
            phi=settings.rng.normal(size=2),
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
        return -math.abs(amps[1, 1]) ** 2 + math.abs(amps[0, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.05)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_two_mode_Ggate():
    """Finding the optimal Ggate to make a pair of single photons"""
    skip_np()

    settings.SEED = 42
    G = Ggate(num_modes=2, symplectic_trainable=True)

    def cost_fn():
        amps = (Vacuum(2) >> G).ket(cutoffs=[2, 2], max_prob=0.9999)
        return -math.abs(amps[1, 1]) ** 2 + math.abs(amps[0, 1]) ** 2

    opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[G], max_steps=500)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-4)


def test_learning_two_mode_Interferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    skip_np()

    settings.SEED = 42
    ops = [
        Sgate(
            r=settings.rng.normal(size=2) ** 2,
            phi=settings.rng.normal(size=2),
            r_trainable=True,
            phi_trainable=True,
        ),
        Interferometer(num_modes=2, unitary_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[2, 2])
        return -math.abs(amps[1, 1]) ** 2 + math.abs(amps[0, 1]) ** 2

    opt = Optimizer(unitary_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_two_mode_RealInterferometer():
    """Finding the optimal Interferometer to make a pair of single photons"""
    skip_np()

    settings.SEED = 2
    ops = [
        Sgate(
            r=settings.rng.normal(size=2) ** 2,
            phi=settings.rng.normal(size=2),
            r_trainable=True,
            phi_trainable=True,
        ),
        RealInterferometer(num_modes=2, orthogonal_trainable=True),
    ]
    circ = Circuit(ops)
    state_in = Vacuum(num_modes=2)

    def cost_fn():
        amps = (state_in >> circ).ket(cutoffs=[2, 2])
        return -math.abs(amps[1, 1]) ** 2 + math.abs(amps[0, 1]) ** 2

    opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
    assert np.allclose(-cost_fn(), 0.25, atol=1e-5)


def test_learning_four_mode_Interferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    skip_np()

    settings.SEED = 4
    solution_U = np.array(
        [
            [
                -0.47541806 + 0.00045878j,
                -0.41513474 - 0.27218387j,
                -0.11065812 - 0.39556922j,
                -0.29912017 + 0.51900235j,
            ],
            [
                -0.05246398 + 0.5209089j,
                -0.29650069 - 0.40653082j,
                0.57434638 - 0.04417284j,
                0.28230532 - 0.24738672j,
            ],
            [
                0.28437557 + 0.08773767j,
                0.18377764 - 0.66496587j,
                -0.5874942 - 0.19866946j,
                0.2010813 - 0.10210844j,
            ],
            [
                -0.63173183 - 0.11057324j,
                -0.03468292 + 0.15245454j,
                -0.25390362 - 0.2244298j,
                0.18706333 - 0.64375049j,
            ],
        ]
    )
    perturbed = (
        Interferometer(num_modes=4, unitary=solution_U)
        >> BSgate(settings.rng.normal(scale=0.01), modes=[0, 1])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[2, 3])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[1, 2])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[0, 3])
    )
    X = math.cast(perturbed.XYd()[0], "complex128")
    perturbed_U = X[:4, :4] + 1j * X[4:, :4]

    ops = [
        Sgate(
            r=settings.rng.normal(loc=np.arcsinh(1.0), scale=0.01, size=4),
            r_trainable=True,
        ),
        Interferometer(unitary=perturbed_U, num_modes=4, unitary_trainable=True),
    ]
    circ = Circuit(ops)

    def cost_fn():
        amps = (Vacuum(num_modes=4) >> circ).ket(cutoffs=[3, 3, 3, 3])
        return -math.abs((amps[1, 1, 2, 0] + amps[1, 1, 0, 2]) / np.sqrt(2)) ** 2

    opt = Optimizer(unitary_lr=0.05)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=200)
    assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)


def test_learning_four_mode_RealInterferometer():
    """Finding the optimal Interferometer to make a NOON state with N=2"""
    skip_np()

    settings.SEED = 6
    solution_O = np.array(
        [
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5, -0.5],
        ]
    )
    solution_S = (np.arcsinh(1.0), np.array([0.0, np.pi / 2, -np.pi, -np.pi / 2]))
    pertubed = (
        RealInterferometer(orthogonal=solution_O, num_modes=4)
        >> BSgate(settings.rng.normal(scale=0.01), modes=[0, 1])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[2, 3])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[1, 2])
        >> BSgate(settings.rng.normal(scale=0.01), modes=[0, 3])
    )
    perturbed_O = pertubed.XYd()[0][:4, :4]

    ops = [
        Sgate(
            r=solution_S[0] + settings.rng.normal(scale=0.01, size=4),
            phi=solution_S[1] + settings.rng.normal(scale=0.01, size=4),
            r_trainable=True,
            phi_trainable=True,
        ),
        RealInterferometer(
            orthogonal=perturbed_O, num_modes=4, orthogonal_trainable=True
        ),
    ]
    circ = Circuit(ops)

    def cost_fn():
        amps = (Vacuum(num_modes=4) >> circ).ket(cutoffs=[2, 2, 3, 3])
        return -math.abs((amps[1, 1, 0, 2] + amps[1, 1, 2, 0]) / np.sqrt(2)) ** 2

    opt = Optimizer()

    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=200)
    assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)


def test_squeezing_hong_ou_mandel_optimizer():
    """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
    see https://www.pnas.org/content/117/52/33107/tab-article-info
    """
    skip_np()

    settings.SEED = 42
    r = np.arcsinh(1.0)

    S_01 = S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1]
    S_23 = S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3]
    S_12 = S2gate(
        r=1.0, phi=settings.rng.normal(), r_trainable=True, phi_trainable=True
    )[1, 2]

    circ = Circuit([S_01, S_23, S_12])

    def cost_fn():
        return math.abs((Vacuum(4) >> circ).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    assert np.allclose(np.sinh(S_12.r.value) ** 2, 1, atol=1e-2)


def test_parameter_passthrough():
    """Same as the test above, but with param passthrough"""
    skip_np()

    settings.SEED = 42
    r = np.arcsinh(1.0)
    r_var = Variable(r, "r", (0.0, None))
    phi_var = Variable(settings.rng.normal(), "phi", (None, None))

    ops = [
        S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1],
        S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3],
        S2gate(r=r_var, phi=phi_var)[1, 2],
    ]
    circ = Circuit(ops)

    def cost_fn():
        return math.abs((Vacuum(4) >> circ).ket(cutoffs=[2, 2, 2, 2])[1, 1, 1, 1]) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(cost_fn, by_optimizing=[r_var, phi_var], max_steps=300)
    assert np.allclose(np.sinh(r_var.value) ** 2, 1, atol=1e-2)


def test_making_thermal_state_as_one_half_two_mode_squeezed_vacuum():
    """Optimizes a Ggate on two modes so as to prepare a state with the same entropy
    and mean photon number as a thermal state"""
    skip_np()

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
        entropy = von_neumann_entropy(cov1)
        S = thermal_entropy(nbar)
        return (mean1 - nbar) ** 2 + (entropy - S) ** 2 + (mean2 - nbar) ** 2

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=50)
    S = math.asnumpy(G.symplectic.value)
    cov = S @ S.T
    assert np.allclose(cov, two_mode_squeezing(2 * np.arcsinh(np.sqrt(nbar)), 0.0))


def test_opt_backend_param():
    """Test the optimization of a backend parameter defined outside a gate."""
    skip_np()

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

    opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.05)
    opt.minimize(cost_fn_sympl, by_optimizing=[S, r_angle])

    assert np.allclose(math.asnumpy(r_angle), rotation_angle / 2, atol=1e-4)


def test_dgate_optimization():
    """Test that Dgate is optimized correctly."""
    skip_np()

    settings.SEED = 24

    dgate = Dgate(x_trainable=True, y_trainable=True)
    target_state = DisplacedSqueezed(r=0.0, x=0.1, y=0.2).ket(cutoffs=[40])

    def cost_fn():
        state_out = Vacuum(1) >> dgate
        return -math.abs(math.sum(math.conj(state_out.ket([40])) * target_state)) ** 2

    opt = Optimizer()
    opt.minimize(cost_fn, by_optimizing=[dgate])

    assert np.allclose(dgate.x.value, 0.1, atol=0.01)
    assert np.allclose(dgate.y.value, 0.2, atol=0.01)


def test_sgate_optimization():
    """Test that Sgate is optimized correctly."""
    skip_np()

    settings.SEED = 25

    sgate = Sgate(r=0.2, phi=0.1, r_trainable=True, phi_trainable=True)
    target_state = SqueezedVacuum(r=0.1, phi=0.2).ket(cutoffs=[40])

    def cost_fn():
        state_out = Vacuum(1) >> sgate

        return -math.abs(math.sum(math.conj(state_out.ket([40])) * target_state)) ** 2

    opt = Optimizer()
    opt.minimize(cost_fn, by_optimizing=[sgate])

    assert np.allclose(sgate.r.value, 0.1, atol=0.01)
    assert np.allclose(sgate.phi.value, 0.2, atol=0.01)


def test_bsgate_optimization():
    """Test that Sgate is optimized correctly."""
    skip_np()

    settings.SEED = 25

    G = Gaussian(2)

    bsgate = BSgate(0.05, 0.1, theta_trainable=True, phi_trainable=True)
    target_state = (G >> BSgate(0.1, 0.2)).ket(cutoffs=[40, 40])

    def cost_fn():
        state_out = G >> bsgate

        return (
            -math.abs(math.sum(math.conj(state_out.ket([40, 40])) * target_state)) ** 2
        )

    opt = Optimizer()
    opt.minimize(cost_fn, by_optimizing=[bsgate])

    assert np.allclose(bsgate.theta.value, 0.1, atol=0.01)
    assert np.allclose(bsgate.phi.value, 0.2, atol=0.01)


def test_squeezing_grad_from_fock():
    """Test that the gradient of a squeezing gate is computed from the fock representation."""
    skip_np()

    squeezing = Sgate(r=1, r_trainable=True)

    def cost_fn():
        return -(Fock(2) >> squeezing << Vacuum(1))

    opt = Optimizer(euclidean_lr=0.05)
    opt.minimize(cost_fn, by_optimizing=[squeezing], max_steps=100)


def test_displacement_grad_from_fock():
    """Test that the gradient of a displacement gate is computed from the fock representation."""
    skip_np()

    disp = Dgate(x=1.0, y=1.0, x_trainable=True, y_trainable=True)

    def cost_fn():
        return -(Fock(2) >> disp << Vacuum(1))

    opt = Optimizer(euclidean_lr=0.05)
    opt.minimize(cost_fn, by_optimizing=[disp], max_steps=100)


def test_bsgate_grad_from_fock():
    """Test that the gradient of a beamsplitter gate is computed from the fock representation."""
    skip_np()

    sq = SqueezedVacuum(r=1.0, r_trainable=True)

    def cost_fn():
        return -((sq & Fock(1)) >> BSgate(0.5) << (Vacuum(1) & Fock(1)))

    opt = Optimizer(euclidean_lr=0.05)
    opt.minimize(cost_fn, by_optimizing=[sq], max_steps=100)
