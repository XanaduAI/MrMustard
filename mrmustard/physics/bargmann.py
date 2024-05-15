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

r"""
This module contains functions for performing calculations on objects in the Bargmann representations.

The Bargmann representation of the state can be obtained as projecting the state on to the Bargmann basis.

The Bargmann basis :math:`|z\rangle_b` can be defined from the coherent state basis :math:`|z\rangle_c` 

.. math:: 
    |z\rangle_b = e^\frac{|z|^2}{2}|z\rangle_c = \sum_n \frac{z^n}{\sqrt{n!}}|n\rangle_f.

Any Gaussian objects :math:`O` can be written in the Bargmann basis as a Gaussian exponential function, parametrized by a matrix :math:`A`, a vector :math:`b` and a scalar :math:`c`, which is called ``triples`` through all the documentations in MM:

.. math::
    \langle\vec{\alpha}|O\rangle = c \exp\left( \frac12 \vec{\alpha}^T A \vec{\alpha} + \vec{\alpha}^T b \right).

The objects in Bargmann representation uses the :class:`~mrmustard.physics.ansatze.PolyExpAnsatz` and the information is stored in the triple (A,b,c).

The expression :math:`\langle\vec{\alpha}|O\rangle` is vectorized the variables vector :math:`\vec{\alpha}`, which is different for different quantum objects. 
As for a ``n``-mode pure Gaussian state :math:`\langle\vec{\alpha}|\psi\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_{n-1})`.

    .. code-block::

        ╔═══════╗
        ║       ║─────▶ alpha^*_0
        ║ |psi> ║─────▶ alpha^*_1
        ║       ║...
        ║       ║─────▶ alpha^*_(n-1)   
        ╚═══════╝    

All the wires in ths mixed Gaussian state :math:`\langle\vec{\alpha}|\rho|\vec{\beta}\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_{n-1}, \beta_0, \beta_1,..., \beta_{n-1})`.

    .. code-block::

        ╔═══════╗
        ║       ║─────▶ alpha^*_0
        ║       ║─────▶ alpha^*_1
        ║       ║─────▶ ...
        ║       ║─────▶ alpha^*_(n-1)  
        ║  rho  ║─────▶ beta_0_1
        ║       ║─────▶ beta_1
        ║       ║─────▶ ...
        ║       ║─────▶ beta_(n-1)    
        ╚═══════╝ 

The wires in the diagram below correspond to the `out_bra` wires (:math:`\alpha^*`) and the `out_ket` wires (:math:`\beta`) in :class:`~mrmustard.lab_dev.wires.Wires`.

As for a ``n``-mode Gaussian unitary :math:`\langle\vec{\alpha}|U|\vec{\beta}\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_{n-1})`.

    .. code-block::

                        ╔═══════╗
        beta_0    ─────▶║       ║─────▶ alpha^*_0
        beta_1    ─────▶║   U   ║─────▶ alpha^*_1
                     ...║       ║...
        beta_(n-1)─────▶║       ║─────▶ alpha^*_(n-1)       
                        ╚═══════╝    

The wires in the diagram below correspond to the `out_ket` wires (:math:`\alpha^*`) and the `in_ket` wires (:math:`\beta`) in :class:`~mrmustard.lab_dev.wires.Wires`.
                    
As for a ``n``-mode Gaussian Channel :math:`\langle \vec{\alpha}|\Psi(|\vec{\gamma}\rangle\langle\vec{\delta}|)|\vec{\beta}`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_{n-1}, \beta_0, \beta_1,..., \beta_{n-1}, \delta^*_0, \delta^*_1, ..., \delta^*_{n-1}, \gamma_0, \gamma_1,..., \gamma_{n-1})`.

    .. code-block::

                             ╔═══════╗
        delta^*_0      ─────▶║       ║─────▶ alpha^*_0
        delta^*_1      ─────▶║       ║─────▶ alpha^*_1
                          ...║       ║...
        delta^*_(n-1)  ─────▶║  Phi  ║─────▶ alpha^*_(n-1)    
        gamma_0        ─────▶║       ║─────▶ beta_0
        gamma_1        ─────▶║       ║─────▶ beta_1
                          ...║       ║...
        gamma_(n-1)    ─────▶║       ║─────▶ beta_(n-1)  
                             ╚═══════╝    

The wires in the diagram below correspond to the `out_bra` wires (:math:`\alpha^*`), the `in_bra` wires (:math:`\delta^*`), `out_ket` wires (:math:`\beta`) and the `in_ket` wires (:math:`\gamma`) in :class:`~mrmustard.lab_dev.wires.Wires`.

The computation of quantum circuits with Bargmann representation can be considered as the inner product of two Bargmann representations (which can be realized by Gaussian integrals for all Gaussian objects computation), such as applying the unitary on a state, contracting two unitaries, applying the channel on a state, and etc.

For example, applying a single-mode unitary :math:`U` on a single-mode pure state :math:`|\psi\rangle` is to multiply the Bargmann representation of the unitary and the state and then to integral the variables on the common wire between then:

    .. math::
        U|\psi\rangle = \int d^2 \alpha |\beta\rangle \langle\beta|U|\alpha\rangle \langle\alpha|\psi\rangle.

    .. code-block::

        ╔═══════╗                                ╔═════╗
        ║ |psi> ║─────▶ alpha^*_0   alpha_0─────▶║  U  ║─────▶ beta^*_0
        ╚═══════╝                                ╚═════╝
            |
            | integral on alpha
            |
        ╔════════╗
        ║ |psi'> ║─────▶ beta^*_0
        ╚════════╝

Another example, applying a single-mode Gaussian channel :math:`\Phi` on a single-mode pure state, one needs to add the ``adjoint`` of the state and contract with the input of the channel:

    .. code-block::

                                                 ╔═══════╗
                                    alpha_0─────▶║       ║─────▶ beta_0
                                                 ║  Phi  ║
        ╔═══════╗                                ║       ║
        ║ |psi> ║─────▶ alpha^*_0 alpha^*_0─────▶║       ║─────▶ beta^*_0
        ╚═══════╝                                ╚═══════╝ 
            |
            | Add the ajoint part
            |
        ╔═══════╗                                ╔═══════╗
        ║ <psi| ║─────▶ alpha_0     alpha_0─────▶║       ║─────▶ beta_0
        ╚═══════╝                                ║  Phi  ║
        ╔═══════╗                                ║       ║
        ║ |psi> ║─────▶ alpha^*_0 alpha^*_0─────▶║       ║─────▶ beta^*_0
        ╚═══════╝                                ╚═══════╝ 
            |
            | integral on alpha
            |
        ╔═══════╗
        ║       ║─────▶ beta_0
        ║ rho'  ║
        ║       ║
        ║       ║─────▶ beta^*_0
        ╚═══════╝ 

        
"""
import numpy as np

from mrmustard import math, settings
from mrmustard.physics.husimi import pq_to_aadag, wigner_to_husimi


def cayley(X, c):
    r"""Returns the Cayley transform of a matrix:
    :math:`cay(X) = (X - cI)(X + cI)^{-1}`

    Args:
        c (float): the parameter of the Cayley transform
        X (Tensor): a matrix

    Returns:
        Tensor: the Cayley transform of X
    """
    I = math.eye(X.shape[0], dtype=X.dtype)
    return math.solve(X + c * I, X - c * I)


def wigner_to_bargmann_rho(cov, means):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a density matrix (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    The order of the rows/columns of A and B corresponds to a density matrix with the usual ordering of the indices.

    Note that here A and B are defined with respect to the literature.
    """
    N = cov.shape[-1] // 2
    A = math.matmul(math.Xmat(N), cayley(pq_to_aadag(cov), c=0.5))
    Q, beta = wigner_to_husimi(cov, means)
    b = math.solve(Q, beta)
    B = math.conj(b)
    num_C = math.exp(-0.5 * math.sum(math.conj(beta) * b))
    detQ = math.det(Q)
    den_C = math.sqrt(detQ, dtype=num_C.dtype)
    C = num_C / den_C
    return A, B, C


def wigner_to_bargmann_psi(cov, means):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann A,B,C triple
    for a Hilbert vector (i.e. for M modes, A has shape M x M and B has shape M).
    """
    N = cov.shape[-1] // 2
    A, B, C = wigner_to_bargmann_rho(cov, means)
    return A[N:, N:], B[N:], math.sqrt(C)
    # NOTE: c for th psi is to calculated from the global phase formula.


def wigner_to_bargmann_Choi(X, Y, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a channel (i.e. for M modes, A has shape 4M x 4M and B has shape 4M)."""
    N = X.shape[-1] // 2
    I2 = math.eye(2 * N, dtype=X.dtype)
    XT = math.transpose(X)
    xi = 0.5 * (I2 + math.matmul(X, XT) + 2 * Y / settings.HBAR)
    detxi = math.det(xi)
    xi_inv = math.inv(xi)
    A = math.block(
        [
            [I2 - xi_inv, math.matmul(xi_inv, X)],
            [math.matmul(XT, xi_inv), I2 - math.matmul(math.matmul(XT, xi_inv), X)],
        ]
    )
    I = math.eye(N, dtype="complex128")
    o = math.zeros_like(I)
    R = math.block(
        [[I, 1j * I, o, o], [o, o, I, -1j * I], [I, -1j * I, o, o], [o, o, I, 1j * I]]
    ) / np.sqrt(2)
    A = math.matmul(math.matmul(R, A), math.dagger(R))
    A = math.matmul(math.Xmat(2 * N), A)
    b = math.matvec(xi_inv, d)
    B = math.matvec(math.conj(R), math.concat([b, -math.matvec(XT, b)], axis=-1)) / math.sqrt(
        settings.HBAR, dtype=R.dtype
    )
    C = math.exp(-0.5 * math.sum(d * b) / settings.HBAR) / math.sqrt(detxi, dtype=b.dtype)
    # now A and B have order [out_r, in_r out_l, in_l].
    return A, B, math.cast(C, "complex128")


def wigner_to_bargmann_U(X, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    """
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    return A[2 * N :, 2 * N :], B[2 * N :], math.sqrt(C)
