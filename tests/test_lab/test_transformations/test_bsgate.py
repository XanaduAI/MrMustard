# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``BSgate`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.transformations import BSgate


class TestBSgate:
    r"""
    Tests for the ``BSgate`` class.
    """

    modes = [(0, 8), (1, 2), (7, 9)]
    theta = [1, 1, 2]
    phi = [3, 4, 4]

    def test_init(self):
        gate = BSgate((0, 1), 2, 3)

        assert gate.name == "BSgate"
        assert gate.modes == (0, 1)
        assert gate.parameters.theta.value == 2
        assert gate.parameters.phi.value == 3

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        theta = math.broadcast_to(0.1, batch_shape)
        phi = math.broadcast_to(0.2, batch_shape)
        rep1 = BSgate((0, 1), theta, phi).ansatz
        A_exp = [
            [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
            [0.0, 0, 0.0978434 + 0.01983384j, 0.99500417],
            [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
            [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
        ]
        assert math.allclose(rep1.A, A_exp)
        assert math.allclose(rep1.b, math.zeros((4,)))
        assert math.allclose(rep1.c, 1)

        theta = math.broadcast_to(0.1, batch_shape)
        rep2 = BSgate((0, 1), theta).ansatz
        A_exp = [
            [0, 0, 9.95004165e-01, -9.98334166e-02],
            [0.0, 0, 9.98334166e-02, 9.95004165e-01],
            [9.95004165e-01, 9.98334166e-02, 0, 0],
            [-9.98334166e-02, 9.95004165e-01, 0, 0],
        ]
        assert math.allclose(rep2.A, A_exp)
        assert math.allclose(rep2.b, math.zeros((4,)))
        assert math.allclose(rep2.c, 1)

    def test_trainable_parameters(self):
        gate1 = BSgate((0, 1), 1, 1)
        gate2 = BSgate((0, 1), 1, 1, theta_trainable=True, theta_bounds=(-2, 2))
        gate3 = BSgate((0, 1), 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.theta.value = 3

        gate2.parameters.theta.value = 2
        assert gate2.parameters.theta.value == 2

        gate3.parameters.phi.value = 2
        assert gate3.parameters.phi.value == 2

    def test_fock_representation(self):
        gate = BSgate((0, 1), 2, 3)
        gate_fock = gate.fock_array(5, method="vanilla")  # int shape
        gate_fock2 = gate.fock_array((5, 5, 5, 5), method="schwinger")  # tuple shape
        assert math.allclose(gate_fock, gate_fock2)

        with pytest.raises(ValueError):
            gate.fock_array((5, 5, 5))  # wrong shape

        bs_with_batch = BSgate((0, 1), math.astensor([[1, 2]]), math.astensor([[3], [4], [5]]))
        assert bs_with_batch.fock_array(5, method="stable").shape == (3, 2, 5, 5, 5, 5)

    def test_to_fock_lin_sup(self):
        bsgate = (BSgate((0, 1), 2, 3) + BSgate((0, 1), -2, -3)).to_fock(5)
        assert bsgate.ansatz.batch_dims == 0
        assert bsgate.ansatz.batch_shape == ()
        assert bsgate.ansatz.array.shape == (5, 5, 5, 5)
