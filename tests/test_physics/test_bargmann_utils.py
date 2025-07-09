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

"""Tests for the bargmann_utils.py file."""

import numpy as np

from mrmustard.lab import DM, Channel, Dgate, Ket
from mrmustard.physics.bargmann_utils import (
    XY_of_channel,
    au2Symplectic,
    symplectic2Au,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
)


def test_wigner_to_bargmann_psi():
    """Test that the Bargmann representation of a ket is correct"""
    G = Ket.random((0, 1)) >> Dgate(0, 0.1, 0.2) >> Dgate(1, 0.2, 0.4)
    cov, means, coeff = G.phase_space(s=0)
    A_exp, b_exp, c_exp = wigner_to_bargmann_psi(cov, means)
    A, b, c = G.bargmann_triple()
    assert np.allclose(A, A_exp)
    assert np.allclose(b, b_exp)
    assert np.allclose(np.abs(c), np.abs(c_exp))


def test_wigner_to_bargmann_rho():
    """Test that the Bargmann representation of a dm is correct"""
    G = DM.random((0, 1)) >> Dgate(0, 0.1, 0.2) >> Dgate(1, 0.2, 0.4)
    cov, means, coeff = G.phase_space(s=0)
    A, b, c = wigner_to_bargmann_rho(cov, means)
    A_exp, b_exp, c_exp = wigner_to_bargmann_rho(cov, means)
    assert np.allclose(A, A_exp)
    assert np.allclose(b, b_exp)
    assert np.allclose(c, c_exp)


def test_au2Symplectic():
    """Tests the Au -> symplectic code; we check two simple examples"""
    # Beam splitter example
    V = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]])

    Au = np.block([[np.zeros_like(V), V], [np.transpose(V), np.zeros_like(V)]])
    S = au2Symplectic(Au)
    S_by_hand = np.block([[V, np.zeros_like(V)], [np.zeros_like(V), np.conjugate(V)]])
    transformation = (
        1
        / np.sqrt(2)
        * np.block(
            [
                [np.eye(2), np.eye(2)],
                [-1j * np.eye(2), 1j * np.eye(2)],
            ],
        )
    )
    S_by_hand = transformation @ S_by_hand @ np.conjugate(np.transpose(transformation))
    assert np.allclose(S, S_by_hand)

    # squeezing example
    r = 2
    Au = np.array([[-np.tanh(r), 1 / np.cosh(r)], [1 / np.cosh(r), np.tanh(r)]])
    S = au2Symplectic(Au)
    S_by_hand = np.array([[np.cosh(r), -np.sinh(r)], [-np.sinh(r), np.cosh(r)]])
    transformation = 1 / np.sqrt(2) * np.array([[1, 1], [-1j, 1j]])
    S_by_hand = transformation @ S_by_hand @ np.conjugate(np.transpose(transformation))
    assert np.allclose(S, S_by_hand)


def test_symplectic2Au():
    """Tests the Symplectic -> Au code"""

    # here we consider the circuit of two-mode squeezing

    r = 2  # some squeezing parameter

    S_bs = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]]) / np.sqrt(2)

    S_sq = np.array(
        [
            [np.cosh(r), 0, -np.sinh(r), 0],
            [0, np.cosh(r), 0, np.sinh(r)],
            [-np.sinh(r), 0, np.cosh(r), 0],
            [0, np.sinh(r), 0, np.cosh(r)],
        ],
    )

    S = S_bs @ S_sq

    m = S.shape[-1]
    m = m // 2
    # the following lines of code transform the quadrature symplectic matrix to
    # the annihilation one
    transformation = np.block(
        [[np.eye(m), np.eye(m)], [-1j * np.eye(m), 1j * np.eye(m)]],
    ) / np.sqrt(2)
    S = transformation @ S @ np.conjugate(np.transpose(transformation))
    A = symplectic2Au(S)

    W = S_bs[:2, :2]
    T = np.diag([np.tanh(r), -np.tanh(r)])
    C = np.diag([np.cosh(r), np.cosh(r)])
    Sec = np.linalg.pinv(C)
    A_by_hand = np.block(
        [[-W @ T @ np.transpose(W), W @ Sec], [Sec @ np.transpose(W), np.conjugate(T)]],
    )

    assert np.allclose(A, A_by_hand)


def test_XY_of_channel():
    r"""
    Tests the function X_of_channel.
    """

    X, Y = XY_of_channel(Channel.random([0]).ansatz.A)
    omega = np.array([[0, 1j], [-1j, 0]])
    channel_check = X @ omega @ X.T / 2 - omega / 2 + Y
    assert np.all([mu > 0 for mu in np.linalg.eigvals(channel_check)])
