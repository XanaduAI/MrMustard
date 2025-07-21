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

"""Tests for the Optimizer class"""

import numpy as np
import pytest
import tensorflow as tf
from hypothesis import given
from hypothesis import strategies as st
from thewalrus.symplectic import two_mode_squeezing

from mrmustard import math, settings
from mrmustard.lab import (
    BSgate,
    Circuit,
    Dgate,
    DisplacedSqueezed,
    Ggate,
    GKet,
    Interferometer,
    Number,
    RealInterferometer,
    Rgate,
    S2gate,
    Sgate,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.math.parameters import Variable, update_euclidean
from mrmustard.physics.gaussian import number_means, von_neumann_entropy
from mrmustard.training import Optimizer
from mrmustard.training.callbacks import Callback


@pytest.mark.requires_backend("tensorflow")
class TestOptimizer:
    r"""
    Tests for the ``Optimizer`` class.
    """

    @given(n=st.integers(0, 3))
    def test_S2gate_coincidence_prob(self, n):
        """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
        with settings(SEED=40):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            S = TwoModeSqueezedVacuum(
                (0, 1), r=abs(settings.rng.normal(loc=1.0, scale=0.1)), r_trainable=True
            )

            def cost_fn():
                return -(math.abs(S.fock_array((n + 1, n + 1))[n, n]) ** 2)

            def cb(optimizer, cost, trainables, **kwargs):
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
    def test_hong_ou_mandel_optimizer(self, i, k):
        """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
        This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
        see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
        which lacks a square root in the right hand side.
        """
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            r = np.arcsinh(1.0)
            cutoff = 1 + i + k

            state = TwoModeSqueezedVacuum((0, 1), r=r, phi_trainable=True)
            bs = BSgate(
                (1, 2),
                theta=np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.rng.normal(),
                phi=settings.rng.normal(),
                theta_trainable=True,
                phi_trainable=True,
            )
            circ = Circuit([state, state.on((2, 3)), bs])

            def cost_fn():
                return math.abs(circ.contract().fock_array((cutoff,) * 4)[i, 1, i + k - 1, k]) ** 2

            opt = Optimizer(euclidean_lr=0.01)
            opt.minimize(
                cost_fn,
                by_optimizing=[circ],
                max_steps=300,
                callbacks=[Callback(tag="null_cb", steps_per_call=3)],
            )
            assert np.allclose(np.cos(bs.parameters.theta.value) ** 2, k / (i + k), atol=1e-2)
            assert "null_cb" in opt.callback_history
            assert len(opt.callback_history["null_cb"]) == (len(opt.opt_history) - 1) // 3

    def test_learning_two_mode_squeezing(self):
        """Finding the optimal beamsplitter transmission to make a pair of single photons"""
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            state_in = Vacuum((0, 1))
            s_gate = Sgate(
                0,
                r=abs(settings.rng.normal()),
                phi=settings.rng.normal(),
                r_trainable=True,
                phi_trainable=True,
            )
            bs_gate = BSgate(
                (0, 1),
                theta=settings.rng.normal(),
                phi=settings.rng.normal(),
                theta_trainable=True,
                phi_trainable=True,
            )
            circ = Circuit([state_in, s_gate, s_gate.on(1), bs_gate])

            def cost_fn():
                amps = circ.contract().fock_array((2, 2))
                return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

            opt = Optimizer(euclidean_lr=0.05)

            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
            assert np.allclose(-cost_fn(), 0.25, atol=1e-5)

    def test_learning_two_mode_Ggate(self):
        """Finding the optimal Ggate to make a pair of single photons"""
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            G = GKet((0, 1), symplectic_trainable=True)

            def cost_fn():
                amps = G.fock_array((2, 2))
                return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

            opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

            opt.minimize(cost_fn, by_optimizing=[G], max_steps=500)
            assert np.allclose(-cost_fn(), 0.25, atol=1e-4)

    def test_learning_two_mode_Interferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        with settings(SEED=4):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            state_in = Vacuum((0, 1))
            s_gate = Sgate(
                0,
                r=settings.rng.normal() ** 2,
                phi=settings.rng.normal(),
                r_trainable=True,
                phi_trainable=True,
            )
            interferometer = Interferometer((0, 1), unitary_trainable=True)
            circ = Circuit([state_in, s_gate, s_gate.on(1), interferometer])

            def cost_fn():
                amps = circ.contract().fock_array((2, 2))
                return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

            opt = Optimizer(unitary_lr=0.5, euclidean_lr=0.01)

            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
            assert np.allclose(-cost_fn(), 0.25, atol=1e-5)

    def test_learning_two_mode_RealInterferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        with settings(SEED=2):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            state_in = Vacuum((0, 1))
            s_gate0 = Sgate(
                0,
                r=settings.rng.normal() ** 2,
                phi=settings.rng.normal(),
                r_trainable=True,
                phi_trainable=True,
            )
            s_gate1 = Sgate(
                1,
                r=settings.rng.normal() ** 2,
                phi=settings.rng.normal(),
                r_trainable=True,
                phi_trainable=True,
            )
            r_inter = RealInterferometer((0, 1), orthogonal_trainable=True)

            circ = Circuit([state_in, s_gate0, s_gate1, r_inter])

            def cost_fn():
                amps = circ.contract().fock_array((2, 2))
                return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

            opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=1000)
            assert np.allclose(-cost_fn(), 0.25, atol=1e-5)

    def test_learning_four_mode_Interferometer(self):
        """Finding the optimal Interferometer to make a NOON state with N=2"""
        with settings(SEED=4):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

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
                ],
            )
            perturbed = (
                Interferometer((0, 1, 2, 3), unitary=solution_U)
                >> BSgate((0, 1), settings.rng.normal(scale=0.01))
                >> BSgate((2, 3), settings.rng.normal(scale=0.01))
                >> BSgate((1, 2), settings.rng.normal(scale=0.01))
                >> BSgate((0, 3), settings.rng.normal(scale=0.01))
            )
            X = perturbed.symplectic
            perturbed_U = X[:4, :4] + 1j * X[4:, :4]

            state_in = Vacuum((0, 1, 2, 3))
            s_gate = Sgate(
                0,
                r=settings.rng.normal(loc=np.arcsinh(1.0), scale=0.01),
                r_trainable=True,
            )
            interferometer = Interferometer(
                (0, 1, 2, 3),
                unitary=perturbed_U,
                unitary_trainable=True,
            )

            circ = Circuit(
                [state_in, s_gate, s_gate.on(1), s_gate.on(2), s_gate.on(3), interferometer],
            )

            def cost_fn():
                amps = circ.contract().fock_array((3, 3, 3, 3))
                return -(math.abs((amps[1, 1, 2, 0] + amps[1, 1, 0, 2]) / np.sqrt(2)) ** 2)

            opt = Optimizer(unitary_lr=0.05)
            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=200)
            assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)

    def test_learning_four_mode_RealInterferometer(self):
        """Finding the optimal Interferometer to make a NOON state with N=2"""
        with settings(SEED=6):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            solution_O = np.array(
                [
                    [0.5, -0.5, 0.5, 0.5],
                    [-0.5, -0.5, -0.5, 0.5],
                    [0.5, 0.5, -0.5, 0.5],
                    [0.5, -0.5, -0.5, -0.5],
                ],
            )
            pertubed = (
                RealInterferometer((0, 1, 2, 3), orthogonal=solution_O)
                >> BSgate((0, 1), settings.rng.normal(scale=0.01))
                >> BSgate((2, 3), settings.rng.normal(scale=0.01))
                >> BSgate((1, 2), settings.rng.normal(scale=0.01))
                >> BSgate((0, 3), settings.rng.normal(scale=0.01))
            )
            perturbed_O = pertubed.symplectic[:4, :4]

            state_in = Vacuum((0, 1, 2, 3))
            s_gate0 = Sgate(
                0,
                r=np.arcsinh(1.0) + settings.rng.normal(scale=0.01),
                phi=settings.rng.normal(scale=0.01),
                r_trainable=True,
                phi_trainable=True,
            )
            s_gate1 = Sgate(
                1,
                r=np.arcsinh(1.0) + settings.rng.normal(scale=0.01),
                phi=(np.pi / 2) + settings.rng.normal(scale=0.01),
                r_trainable=True,
                phi_trainable=True,
            )
            s_gate2 = Sgate(
                2,
                r=np.arcsinh(1.0) + settings.rng.normal(scale=0.01),
                phi=-np.pi + settings.rng.normal(scale=0.01),
                r_trainable=True,
                phi_trainable=True,
            )
            s_gate3 = Sgate(
                3,
                r=np.arcsinh(1.0) + settings.rng.normal(scale=0.01),
                phi=(-np.pi / 2) + settings.rng.normal(scale=0.01),
                r_trainable=True,
                phi_trainable=True,
            )
            r_inter = RealInterferometer(
                (0, 1, 2, 3),
                orthogonal=perturbed_O,
                orthogonal_trainable=True,
            )

            circ = Circuit([state_in, s_gate0, s_gate1, s_gate2, s_gate3, r_inter])

            def cost_fn():
                amps = circ.contract().fock_array((2, 2, 3, 3))
                return -(math.abs((amps[1, 1, 0, 2] + amps[1, 1, 2, 0]) / np.sqrt(2)) ** 2)

            opt = Optimizer()

            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=200)
            assert np.allclose(-cost_fn(), 0.0625, atol=1e-5)

    def test_squeezing_hong_ou_mandel_optimizer(self):
        """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
        see https://www.pnas.org/content/117/52/33107/tab-article-info
        """
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            r = np.arcsinh(1.0)

            state_in = Vacuum((0, 1, 2, 3))
            S_01 = S2gate((0, 1), r=r, phi=0.0, phi_trainable=True)
            S_23 = S2gate((2, 3), r=r, phi=0.0, phi_trainable=True)
            S_12 = S2gate(
                (1, 2),
                r=1.0,
                phi=settings.rng.normal(),
                r_trainable=True,
                phi_trainable=True,
            )

            circ = Circuit([state_in, S_01, S_23, S_12])

            def cost_fn():
                return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

            opt = Optimizer(euclidean_lr=0.001)
            opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
            assert np.allclose(np.sinh(S_12.parameters.r.value) ** 2, 1, atol=1e-2)

    def test_parameter_passthrough(self):
        """Same as the test above, but with param passthrough"""
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            r = np.arcsinh(1.0)
            r_var = Variable(r, "r", (0.0, None))
            phi_var = Variable(settings.rng.normal(), "phi", (None, None))

            state_in = Vacuum((0, 1, 2, 3))
            s2_gate0 = S2gate((0, 1), r=r, phi=0.0, phi_trainable=True)
            s2_gate1 = S2gate((2, 3), r=r, phi=0.0, phi_trainable=True)
            s2_gate2 = S2gate((1, 2), r=r_var, phi=phi_var)

            circ = Circuit([state_in, s2_gate0, s2_gate1, s2_gate2])

            def cost_fn():
                return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

            opt = Optimizer(euclidean_lr=0.001)
            opt.minimize(cost_fn, by_optimizing=[r_var, phi_var], max_steps=300)
            assert np.allclose(np.sinh(r_var.value) ** 2, 1, atol=1e-2)

    def test_making_thermal_state_as_one_half_two_mode_squeezed_vacuum(self):
        """Optimizes a Ggate on two modes so as to prepare a state with the same entropy
        and mean photon number as a thermal state"""
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            def thermal_entropy(nbar):
                return -(nbar * np.log((nbar) / (1 + nbar)) - np.log(1 + nbar))

            nbar = 1.4
            S_init = two_mode_squeezing(np.arcsinh(1.0), 0.0)
            S = thermal_entropy(nbar)

            G = Ggate((0, 1), symplectic=S_init, symplectic_trainable=True)

            def cost_fn():
                state = Vacuum((0, 1)) >> G

                state0 = state[0]
                state1 = state[1]

                cov0, mean0, _ = state0.phase_space(s=0)
                cov1, mean1, _ = state1.phase_space(s=0)

                num_mean0 = number_means(cov0, mean0)[0]
                num_mean1 = number_means(cov1, mean1)[0]

                entropy = von_neumann_entropy(cov0)
                return (num_mean0 - nbar) ** 2 + (entropy - S) ** 2 + (num_mean1 - nbar) ** 2

            opt = Optimizer(symplectic_lr=0.1)
            opt.minimize(cost_fn, by_optimizing=[G], max_steps=50)
            S = math.asnumpy(G.parameters.symplectic.value)
            cov = S @ S.T
            assert np.allclose(cov, two_mode_squeezing(2 * np.arcsinh(np.sqrt(nbar)), 0.0))

    def test_opt_backend_param(self):
        """Test the optimization of a backend parameter defined outside a gate."""
        # rotated displaced squeezed state
        with settings(SEED=42):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            rotation_angle = np.pi / 2
            target_state = SqueezedVacuum(0, r=1.0, phi=rotation_angle)

            # angle of rotation gate
            r_angle = math.new_variable(0, bounds=(0, np.pi), name="r_angle")
            # trainable squeezing
            S = Sgate(0, r=0.1, r_trainable=True)

            def cost_fn_sympl():
                state_out = Vacuum(0) >> S >> Rgate(0, theta=r_angle)
                return 1 - math.abs((state_out >> target_state.dual) ** 2)

            opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.05)
            opt.minimize(cost_fn_sympl, by_optimizing=[S, r_angle])

            assert np.allclose(math.asnumpy(r_angle), rotation_angle / 2, atol=1e-4)

    def test_dgate_optimization(self):
        """Test that Dgate is optimized correctly."""
        with settings(SEED=24):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            dgate = Dgate(0, x_trainable=True, y_trainable=True)
            target_state = DisplacedSqueezed(0, r=0.0, x=0.1, y=0.2).fock_array((40,))

            def cost_fn():
                state_out = Vacuum(0) >> dgate
                return -(
                    math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2
                )

            opt = Optimizer()
            opt.minimize(cost_fn, by_optimizing=[dgate])

            assert np.allclose(dgate.parameters.x.value, 0.1, atol=0.01)
            assert np.allclose(dgate.parameters.y.value, 0.2, atol=0.01)

    def test_sgate_optimization(self):
        """Test that Sgate is optimized correctly."""
        with settings(SEED=25):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            sgate = Sgate(0, r=0.2, phi=0.1, r_trainable=True, phi_trainable=True)
            target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))

            def cost_fn():
                state_out = Vacuum(0) >> sgate

                return -(
                    math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2
                )

            opt = Optimizer()
            opt.minimize(cost_fn, by_optimizing=[sgate])

            assert np.allclose(sgate.parameters.r.value, 0.1, atol=0.01)
            assert np.allclose(sgate.parameters.phi.value, 0.2, atol=0.01)

    def test_bsgate_optimization(self):
        """Test that BSgate is optimized correctly."""
        with settings(SEED=25):
            rng = tf.random.get_global_generator()
            rng.reset_from_seed(settings.SEED)

            bsgate = BSgate((0, 1), 0.05, 0.1, theta_trainable=True, phi_trainable=True)
            target_gate = BSgate((0, 1), 0.1, 0.2).fock_array(40)

            def cost_fn():
                return -(math.abs(math.sum(math.conj(bsgate.fock_array(40)) * target_gate)) ** 2)

            opt = Optimizer()
            opt.minimize(cost_fn, by_optimizing=[bsgate])

            assert np.allclose(bsgate.parameters.theta.value, 0.1, atol=0.01)
            assert np.allclose(bsgate.parameters.phi.value, 0.2, atol=0.01)

    def test_squeezing_grad_from_fock(self):
        """Test that the gradient of a squeezing gate is computed from the fock representation."""
        squeezing = Sgate(0, r=1.0, r_trainable=True)
        og_r = math.asnumpy(squeezing.parameters.r.value)

        def cost_fn():
            return -((Number(0, 2) >> squeezing >> Vacuum(0).dual) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        opt.minimize(cost_fn, by_optimizing=[squeezing], max_steps=100)

        assert squeezing.parameters.r.value != og_r

    def test_displacement_grad_from_fock(self):
        """Test that the gradient of a displacement gate is computed from the fock representation."""

        disp = Dgate(0, x=1.0, y=0.5, x_trainable=True, y_trainable=True)
        og_x = math.asnumpy(disp.parameters.x.value)
        og_y = math.asnumpy(disp.parameters.y.value)

        def cost_fn():
            return -((Number(0, 2) >> disp >> Vacuum(0).dual) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        opt.minimize(cost_fn, by_optimizing=[disp], max_steps=100)
        assert og_x != disp.parameters.x.value
        assert og_y != disp.parameters.y.value

    def test_bsgate_grad_from_fock(self):
        """Test that the gradient of a beamsplitter gate is computed from the fock representation."""
        sq = SqueezedVacuum(0, r=1.0, r_trainable=True)
        og_r = math.asnumpy(sq.parameters.r.value)

        def cost_fn():
            return -(
                (sq >> Number(1, 1) >> BSgate((0, 1), 0.5) >> (Vacuum(0) >> Number(1, 1)).dual) ** 2
            )

        opt = Optimizer(euclidean_lr=0.05)
        opt.minimize(cost_fn, by_optimizing=[sq], max_steps=100)

        assert og_r != sq.parameters.r.value
