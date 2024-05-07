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
    <\vec{\alpha}|O\rangle = c \exp\left( \frac12 \vec{\alpha}^T A \vec{\alpha} + \vec{\alpha}^T b \right),

Note that:

*The objects in Bargmann representation uses the :class:`~mrmustard.physics.ansatze.PolyExpAnsatz` and the information is stored in the triple (A,b,c).
*The expression :math:`<\vec{\alpha}|O\rangle` is vectorized the variables vector :math:`\vec{\alpha}`, which is different for different quantum objects. 
As for a ``n``-mode pure Gaussian state :math:`\langle\vec{\alpha}|\psi\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_n)`.
.. code-block::

    ╔═══════╗
    ║       ║─────▶ :math:`\alpha^*_0`
    ║ |psi> ║─────▶ :math:`\alpha^*_1`
    ║       ║...
    ║       ║─────▶ :math:`\alpha^*_n`     
    ╚═══════╝    
All the wires in the diagram below correspond to the `out_ket` wires in :class:`~mrmustard.lab_dev.wires.Wires`.
    
As for a ``n``-mode mixed Gaussian state :math:`\langle\vec{\alpha}|\rho|\vec{\beta}\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_n, \beta_0, \beta_1,..., \beta_n)`.
.. code-block::

    ╔═══════╗
    ║       ║─────▶ :math:`\alpha^*_0`
    ║       ║─────▶ :math:`\alpha^*_1`
    ║       ║─────▶ ...
    ║       ║─────▶ :math:`\alpha^*_n`
    ║  rho  ║─────▶ :math:`\beta_0_1`
    ║       ║─────▶ :math:`\beta_1`
    ║       ║─────▶ ...
    ║       ║─────▶ :math:`\beta_n`     
    ╚═══════╝    
The wires in the diagram below correspond to the `out_ket` wires (:math:`\beta`) and the `out_bra` wires ((:math:`\alpha^*`) in :class:`~mrmustard.lab_dev.wires.Wires`.
 
As for a ``n``-mode Gaussian unitary :math:`\langle\vec{\alpha}|U|\vec{\beta}\rangle`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_n)`.

As for a ``n``-mode Gaussian Channel :math:`\langle \vec{\alpha}|\Psi(|\vec{\gamma}\rangle\langle\vec{\delta}|)|\vec{\beta}`, the variable vector denotes :math:`\vec{\alpha} = (\alpha^*_0, \alpha^*_1, ..., \alpha^*_n, \beta_0, \beta_1,..., \beta_n, \delta^*_0, \delta^*_1, ..., \delta^*_n, \gamma_0, \gamma_1,..., \gamma_n)`.


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
