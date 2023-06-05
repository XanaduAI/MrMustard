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


import numpy as np

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.math.caching import tensor_int_cache
from mrmustard.typing import Tensor, RealVector, Sequence
from mrmustard.lab.representations import WignerKet, WignerDM, FockKet, FockDM, BargmannKet, BargmannDM, WavefunctionQKet, WavaefunctionQDM

math = Math()

#This python file contains all the transitions between different nodes.
#With the language of Graph, here we represent all edges in it.

####################################################
#Now we have MVP chain here:
#  Wigner ---> Bargmann ---> Fock ---> WavefunctionQ
#####################################################

########################################################################
###                    From Wigner to Husimi                         ###
########################################################################
#!!!! Husimi is not included now! It is only considered as a intermiate step to bargmann.
def pq_to_aadag(X):
    r"""maps a matrix or vector from the q/p basis to the a/adagger basis"""
    N = X.shape[0] // 2
    R = math.rotmat(N)
    if X.ndim == 2:
        return math.matmul(math.matmul(R, X / settings.HBAR), math.dagger(R))
    elif X.ndim == 1:
        return math.matvec(R, X / math.sqrt(settings.HBAR, dtype=X.dtype))
    else:
        raise ValueError("Input to complexify must be a matrix or vector")


def wigner_to_husimi(cov, means):
    r"Returns the husimi complex covariance matrix and means vector."
    N = cov.shape[-1] // 2
    sigma = pq_to_aadag(cov)
    beta = pq_to_aadag(means)
    Q = sigma + 0.5 * math.eye(2 * N, dtype=sigma.dtype)
    return Q, beta


########################################################################
###                    From Wigner to Bargmann                       ###
########################################################################


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


def wignerdm_to_bargmanndm(wignerdm: WignerDM) -> BargmannDM:
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a density matrix (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    The order of the rows/columns of A and B corresponds to a density matrix with the usual ordering of the indices.

    Note that here A and B are defined with inverted blocks with respect to the literature,
    otherwise the density matrix would have the left and the right indices swapped once we convert to Fock.
    By inverted blocks we mean that if A is normally defined as `A = [[A_00, A_01], [A_10, A_11]]`,
    here we define it as `A = [[A_11, A_10], [A_01, A_00]]`. For `B` we have `B = [B_0, B_1] -> B = [B_1, B_0]`.
    """
    N = wignerdm.data.cov.shape[-1] // 2
    A = math.matmul(
        cayley(pq_to_aadag(wignerdm.data.cov), c=0.5), math.Xmat(N)
    )  # X on the right, so the index order will be rho_{left,right}:
    Q, beta = wigner_to_husimi(wignerdm.data.cov, wignerdm.data.means)
    B = math.solve(Q, beta)  # no conjugate, so that the index order will be rho_{left,right}
    C = math.exp(-0.5 * math.sum(math.conj(beta) * B)) / math.sqrt(math.det(Q))
    return BargmannDM(A, B, C)


def wignerket_to_bargmannket(wignerket: WignerKet) -> BargmannKet:
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann A,B,C triple
    for a Hilbert vector (i.e. for M modes, A has shape M x M and B has shape M).
    """
    N = wignerket.data.cov.shape[-1] // 2
    bargmanndm = wignerdm_to_bargmanndm(wignerket.data.cov, wignerket.data.means)
    # NOTE: with A_rho and B_rho defined with inverted blocks, we now keep the first half rather than the second
    return BargmannKet(bargmanndm.data.A[:N, :N], bargmanndm.data.B[:N], math.sqrt(bargmanndm.data.C))


#DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
def wigner_to_bargmann_Choi(X, Y, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a channel (i.e. for M modes, A has shape 4M x 4M and B has shape 4M).
    We have freedom to choose the order of the indices of the Choi matrix by rearranging the `MxM` blocks of A and the M-subvectors of B.
    Here we choose the order `[out_l, in_l out_r, in_r]` (`in_l` and `in_r` to be contracted with the left and right indices of the density matrix)
    so that after the contraction the result has the right order `[out_l, out_r]`."""
    N = X.shape[-1] // 2
    I2 = math.eye(2 * N, dtype=X.dtype)
    XT = math.transpose(X)
    xi = 0.5 * (I2 + math.matmul(X, XT) + 2 * Y / settings.HBAR)
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
    A = math.matmul(A, math.Xmat(2 * N))  # yes: X on the right
    b = math.matvec(xi_inv, d)
    B = math.matvec(math.conj(R), math.concat([b, -math.matvec(XT, b)], axis=-1)) / math.sqrt(
        settings.HBAR, dtype=R.dtype
    )
    B = math.concat([B[2 * N :], B[: 2 * N]], axis=-1)  # yes: opposite order
    C = math.exp(-0.5 * math.sum(d * b) / settings.HBAR) / math.sqrt(math.det(xi), dtype=b.dtype)
    # now A and B have order [out_l, in_l out_r, in_r].
    return A, B, math.cast(C, "complex128")


#DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
def wigner_to_bargmann_U(X, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    """
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    # NOTE: with A_Choi and B_Choi defined with inverted blocks, we now keep the first half rather than the second
    return A[: 2 * N, : 2 * N], B[: 2 * N], math.sqrt(C)


########################################################################
###                     From Wigner to Fock                         ###
########################################################################


def wignerket_to_fockket(
    wignerket: WignerKet,
    max_prob: float = 1.0,
    max_photons: Optional[int] = None,
) -> FockKet:
    r"""Returns the Fock representation of a Gaussian state in ket form.

    Args:
        wignerket(WignerKet): the WignerKet object. 
        max_prob: the maximum probability of a the state (applies only if the ket is returned)
        max_photons: the maximum number of photons in the state (applies only if the ket is returned)

    Returns:
        FockKet: the fock representation of the ket.
    """

    A, B, C = wignerket_to_bargmannket(wignerket)
    if max_photons is None:
        max_photons = sum(wignerket.data.shape) - len(wignerket.data.shape)
    if max_prob < 1.0 or max_photons < sum(wignerket.data.shape) - len(wignerket.data.shape):
        return math.hermite_renormalized_binomial(
            A, B, C, shape=wignerket.data.shape, max_l2=max_prob, global_cutoff=max_photons + 1
        )
    return math.hermite_renormalized(A, B, C, shape=tuple(wignerket.data.shape))


def wignerdm_to_fockdm(
    wignerdm: WignerDM,
) -> FockDM:
    r"""Returns the Fock representation of a Gaussian state in density matrix form.

    Args:
        wignerdm(WignerDM): the WignerDM object. 
        max_prob: the maximum probability of a the state (applies only if the ket is returned)
        max_photons: the maximum number of photons in the state (applies only if the ket is returned)

    Returns:
        Tensor: the fock representation
    """

    A, B, C = wignerdm_to_bargmanndm(wignerdm)
    return math.hermite_renormalized(A, B, C, shape=wignerdm.data.shape)


#DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
def wigner_to_fock_U(X, d, shape):
    r"""Returns the Fock representation of a Gaussian unitary transformation.
    The index order is out_l, in_l, where in_l is to be contracted with the indices of a ket,
    or with the left indices of a density matrix.

    Arguments:
        X: the X matrix
        d: the d vector
        shape: the shape of the tensor

    Returns:
        Tensor: the fock representation of the unitary transformation
    """
    A, B, C = wigner_to_bargmann_U(X, d)
    return math.hermite_renormalized(A, B, C, shape=tuple(shape))


#DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
def wigner_to_fock_Choi(X, Y, d, shape):
    r"""Returns the Fock representation of a Gaussian Choi matrix.
    The order of choi indices is :math:`[\mathrm{out}_l, \mathrm{in}_l, \mathrm{out}_r, \mathrm{in}_r]`
    where :math:`\mathrm{in}_l` and :math:`\mathrm{in}_r` are to be contracted with the left and right indices of a density matrix.

    Arguments:
        X: the X matrix
        Y: the Y matrix
        d: the d vector
        shape: the shape of the tensor

    Returns:
        Tensor: the fock representation of the Choi matrix
    """
    A, B, C = wigner_to_bargmann_Choi(X, Y, d)
    return math.hermite_renormalized(A, B, C, shape=tuple(shape))


########################################################################
###                     From Bargmann to Fock                        ###
########################################################################
###!!! Just the math.hermite_renormalized


########################################################################
###                     From Fock to Wavefunction                    ###
########################################################################

@tensor_int_cache
def oscillator_eigenstates(q: RealVector, cutoff: int) -> Tensor:
    r"""Harmonic oscillator eigenstate wavefunctions `\psi_n(q) = <q|n>` for n = 0, 1, 2, ..., cutoff-1.

    Args:
        q (Vector): a vector containing the q points at which the function is evaluated (units of \sqrt{\hbar})
        cutoff (int): maximum number of photons

    Returns:
        Tensor: a tensor of shape ``(cutoff, len(q))``. The entry with index ``[n, j]`` represents the eigenstate evaluated
            with number of photons ``n`` evaluated at position ``q[j]``, i.e., `\psi_n(q_j) = <q_j|n>`.

    .. details::

        .. admonition:: Definition
            :class: defn

        The q-quadrature eigenstates are defined as

        .. math::

            \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
                \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

        where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
    """
    omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
    x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

    # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
    prefactor = (omega_over_hbar / np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

    # Renormalized physicist hermite polys: Hn / sqrt(n!)
    R = np.array([[2 + 0j]])  # to get the physicist polys

    def f_hermite_polys(xi):
        poly = math.hermite_renormalized(R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff)
        return math.cast(poly, "float64")

    hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

    # (real) wavefunction
    psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
    return psi


def fockket_to_wavefunctionqket(self, fockket: FockKet, qs: Optional[Sequence[Sequence[float]]] = None) -> WavefunctionQKet:
    r"""Returns the position wavefunction of the Fock ket state at a vector of positions.

    Args:
        fockket (FockKet): a Fock ket object.
        qs (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
            If ``None``, a set of positions is automatically generated.

    Returns:
        ComplexFunctionND: the wavefunction at the given positions wrapped in a
        :class:`~.ComplexFunctionND` object.
    """
    krausses = [math.transpose(oscillator_eigenstates(q, c)) for q, c in zip(qs, self.cutoffs)]

    ket = fockket.data.array
    for i, h_n in enumerate(krausses):
        ket = fockket.apply_kraus_to_ket(h_n, ket, [i])
    return ket  # now in q basis


def fockket_to_wavefunctionqdm(self, fockdm: FockDM, qs: Optional[Sequence[Sequence[float]]] = None) -> WavaefunctionQDM:
    r"""Returns the position wavefunction of the Fock density matrix at a vector of positions.

    Args:
        fockdm (FockDM): a Fock density matrix object.
        qs (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
            If ``None``, a set of positions is automatically generated.

    Returns:
        ComplexFunctionND: the wavefunction at the given positions wrapped in a
        :class:`~.ComplexFunctionND` object.
    """
    krausses = [math.transpose(oscillator_eigenstates(q, c)) for q, c in zip(qs, self.cutoffs)]

    dm = fockdm.data.array
    for i, h_n in enumerate(krausses):
        dm = fockdm.apply_kraus_to_dm(h_n, dm, [i])
    return dm