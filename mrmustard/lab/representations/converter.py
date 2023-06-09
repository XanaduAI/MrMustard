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

from __future__ import annotations
import networkx as nx
import numpy as np
from re import sub
from typing import List, Optional, Tuple, Union
from mrmustard import settings
from mrmustard.math import Math
from mrmustard.math.caching import tensor_int_cache
from mrmustard.typing import Batch, Matrix, RealVector, Scalar, Tensor, Vector
from mrmustard.lab.representations import (
    BargmannKet, BargmannDM,
    FockKet, FockDM,
    WavefunctionQKet, WavaefunctionQDM
    WignerKet, WignerDM
)   

math = Math()

class Converter():

    def __init__(self) -> None:
        r"""
        Class for an object allowing conversion between a given source representation and a desired
          destination one. It relies on the representation transition graph which we detail below.
        
        The transition graph describes transitions between representations. Nodes are the names 
        -as strings- of the Representation object. Edges are functions corresponding to the 
        transitions between two Representation objects
        Note that this graph is:
            - Finite
            - Directed
            - Disconnected, with 2 connected components
            - Acyclic
            - Unweighted
            - Order : 8
            - Size : 6
            - Degree 1 for all nodes
        Currently of the form: Wigner --> Bargmann --> Fock --> WavefunctionQ
        """
        ### DEFINE NODES - REPRESENTATION NAMES

        # Ket part of the graph
        b_K = "BargmannKet"
        f_K = "FockKet"
        wq_K = "WavefunctionQKet"
        w_K = "WignerKet"

        # DM component of the graph
        b_DM = "BargmannDM"
        f_DM = "FockDM"
        wq_DM = "WavefunctionQDM"
        w_DM = "WignerDM"


        ### DEFINE EDGES - CONNEXIONS

        # Ket component of the graph
        w2b_K = (w_K, b_K)
        w2f_K = (w_K, f_K)
        f2wq_K = (f_K, wq_K)

        # DM component of the graph
        w2b_DM = (w_DM, b_DM)
        w2f_DM = (w_DM, f_DM)
        f2wq_DM = (f_DM, wq_DM)


        ### DEFINE EDGE LABELS - FORMULAS

        # Ket component of the graph
        f_w2b_K = self._wignerket_to_bargmannket
        f_w2f_K = self._wignerket_to_fockket
        f_f2wq_K = self._fockket_to_wavefunctionqket

        # DM component of the graph
        f_w2b_DM = self._wignerdm_to_bargmanndm
        f_w2f_DM = self._wignerdm_to_fockdm
        f_f2wq_DM = self._fockdm_to_wavefunctionqdm


        ### DEFINE GRAPH

        edges = [w2b_K, w2f_K, f2wq_K, w2b_DM, w2f_DM, f2wq_DM]

        transition_formulas = {
            w2b_K: {"f": f_w2b_K},
            w2f_K: {"f": f_w2f_K},
            f2wq_K: {"f": f_f2wq_K},
            w2b_DM: {"f": f_w2b_DM},
            w2f_DM: {"f": f_w2f_DM},
            f2wq_DM: {"f": f_f2wq_DM}
        }

        self.g = nx.DiGraph()
        
        self.g.add_edges_from(edges)

        nx.set_edge_attributes(g, transition_formulas)
        


    def _find_target_node_name(self, source:str, destination:str) -> str:
        r""""

        Args:
            source (str)        : the class name of the source Representation, containing the 
                                  Ket/DM suffix
            destination (str)   : the name of the target Representation without the Ket/DM suffix

        Returns:
            The string of the target representation concatenated with eith ket or DM depending on 
            the source.
        """

        suffix = sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', source).split()[-1]

        return destination + suffix
  

    def convert(self, source:Representation, destination:str) -> Representation:
        r""" 
        Converts from a source Representation to target Representation, using the representations 
        graph g.

        .. code-block::
            # assuming we have some State object s
            c = Converter()
            new_s = c.convert(source=s, destination="Fock")

        Args:
            source (State)      : the state which representation must be transformed
            destination (str)   : the name of the target prepresentation, 
                                this name must NOT include ket/DM


        Returns:
            The target representation

        """

        try:
            s_name = source.__class__.__name__
            d_name = self._find_target_node_name(source=source, destination=destination)
            f = self.g[s_name][d_name]
            return f(source)
        
        except KeyError as e:
            raise ValueError(f"{destination} is not a valid target name") from e
        

    
    def shortest_path(self, source:Representation, destination:Representation):
        raise NotImplementedError # TODO : implement


    def add_edge(self):
        raise NotImplementedError # TODO : implement



    def show(self) -> None:
        raise NotImplementedError # TODO : implement
    

         # TODO : for the whole following code double-check / add to the doc
         # TODO : for the whole following code complete the signatures with type hints
    ########################################################################
    ###                    From Wigner to Husimi                         ###
    ########################################################################
    def _pq_to_aadag(self, X:Union[Batch[Matrix], Batch[Vector]]
                     ) -> Union[Batch[Matrix], Batch[Vector]]:
        r"""
        Maps a matrix or vector from the q/p basis to the a/adagger basis

        Args:
            X (Union[Batch[Matrix], Batch[Vector]]) : A matrix or vector in the Q/P basis

        Returns:
            The input matrix/vector in the A/A^\dagger basis
        """

        N = X.shape[0] // 2
        R = math.rotmat(N)

        if X.ndim == 2:
            return math.matmul(math.matmul(R, X / settings.HBAR), math.dagger(R))
        
        elif X.ndim == 1:
            return math.matvec(R, X / math.sqrt(settings.HBAR, dtype=X.dtype))
        
        else:
            raise ValueError("Input to complexify must be a matrix or vector")


    def _wigner_to_husimi(self, cov:Batch[Matrix], means:Batch[Vector]
                          ) -> Tuple[Batch[Matrix], Batch[Vector]]:
        r"""
        Converts from a Wigner covariance and means matrix to Husimi ones.

        Args:
            cov (Batch[Matrix])     :
            means (Batch[Vector])   :

        Returns:
            The Husimi representation's complex covariance and means vector
        """

        N = cov.shape[-1] // 2
        sigma = self._pq_to_aadag(cov)
        beta = self._pq_to_aadag(means)
        Q = sigma + 0.5 * math.eye(2 * N, dtype=sigma.dtype)

        return Q, beta


    ########################################################################
    ###                    From Wigner to Bargmann                       ###
    ########################################################################


    def _cayley(self, X:Tensor, c:float) -> Tensor:
        r"""
        Returns the self._cayley transform of a matrix:
        :math:`cay(X) = (X - cI)(X + cI)^{-1}`

        Args:
            c (float): the parameter of the self._cayley transform
            X (Tensor): a matrix

        Returns:
            The self._cayley transform of X tensor
        """

        I = math.eye(X.shape[0], dtype=X.dtype)

        return math.solve(X + c * I, X - c * I) # TODO : solve the argument nb issue


    def _wignerdm_to_bargmanndm(self, wigner_dm: WignerDM) -> BargmannDM:
        r"""
        Converts the wigner representation in terms of covariance matrix and mean vector into the 
        Bargmann `A,B,C` triple for a density matrix (i.e. for `M` modes, `A` has shape `2M x 2M` 
        and `B` has shape `2M`).
        The order of the rows/columns of A and B corresponds to a density matrix with the usual 
        ordering of the indices.

        Note that here A and B are defined with inverted blocks with respect to the literature,
        otherwise the density matrix would have the left and the right indices swapped once we 
        convert to Fock.
        By inverted blocks we mean that if A is normally defined as
          `A = [[A_00, A_01], [A_10, A_11]]`,
        here we define it as 
        `A = [[A_11, A_10], [A_01, A_00]]`. 
        For `B` we have `B = [B_0, B_1] -> B = [B_1, B_0]`.

        Args:
            wigner_dm (WignerDM) : the Wigner DM representation of the state

        Returns:
            The BargmannDM representation of the input state

        """
        N = wigner_dm.data.cov.shape[-1] // 2

        A = math.matmul(
            self._cayley(self._pq_to_aadag(wigner_dm.data.cov), c=0.5), math.Xmat(N)
        )  # X on the right, so the index order will be rho_{left,right}

        Q, beta = self._wigner_to_husimi(wigner_dm.data.cov, wigner_dm.data.means)
        B = math.solve(Q, beta)  # no conjugate, so that the index order will be rho_{left,right}
        C = math.exp(-0.5 * math.sum(math.conj(beta) * B)) / math.sqrt(math.det(Q))

        return BargmannDM(A, B, C)


    def _wignerket_to_bargmannket(self, wigner_ket: WignerKet) -> BargmannKet:
        r"""
        Converts the Wigner representation in terms of covariance matrix and mean vector into the 
        Bargmann A,B,C triple for a Hilbert vector (i.e. for M modes, A has shape M x M and B has 
        shape M).

        Args:
            wigner_ket (WignerKet) : the input Wigner representation of the state

        Returns:
            The Bargmann representation of the input state
        """

        N = wigner_ket.data.cov.shape[-1] // 2
        bargmann_dm = self._wignerdm_to_bargmanndm(wigner_ket.data.cov, wigner_ket.data.means) 
        # NOTE: with A_rho and B_rho defined with inverted blocks, we now keep the first half 
        # rather than the second
        
        return BargmannKet(bargmann_dm.data.A[:N, :N], 
                           bargmann_dm.data.B[:N], 
                           math.sqrt(bargmann_dm.data.C))


    #DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
    def _wigner_to_bargmann_Choi(self, X, Y, d):
        r"""
        Converts the wigner representation in terms of covariance matrix and mean vector into 
        the Bargmann `A,B,C` triple for a channel (i.e. for M modes, A has shape 4M x 4M and B has 
        shape 4M).
        We have freedom to choose the order of the indices of the Choi matrix by rearranging the 
        `MxM` blocks of A and the M-subvectors of B.
        Here we choose the order `[out_l, in_l out_r, in_r]` (`in_l` and `in_r` to be contracted 
        with the left and right indices of the density matrix) so that after the contraction the 
        result has the right order `[out_l, out_r]`.

        Args:
            X ()    :
            Y ()    :
            d ()    :

        Returns:
            The Bargmann Abc triplet for a quantum channel
        """

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

        B = math.matvec(math.conj(R), math.concat([b, -math.matvec(XT, b)], axis=-1)
                        ) / math.sqrt(settings.HBAR, dtype=R.dtype
        )

        B = math.concat([B[2 * N :], B[: 2 * N]], axis=-1)  # yes: opposite order
        C = math.exp(-0.5 * math.sum(d*b) / settings.HBAR) / math.sqrt(math.det(xi), dtype=b.dtype)
        # now A and B have order [out_l, in_l out_r, in_r].

        return A, B, math.cast(C, "complex128")


    #DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
    def _wigner_to_bargmann_U(self, X, d):
        r"""
        Converts the wigner representation in terms of covariance matrix and mean vector into the 
        Bargmann `A,B,C` triple for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` 
        has shape `2M`).

        Args:
            X ()    :
            d ()    :

        Returns:
            The Bargmann Abc triplet for a unitary transformation
        """

        N = X.shape[-1] // 2
        A, B, C = self._wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
        # NOTE: with A_Choi and B_Choi defined with inverted blocks, we now keep the first half 
        # rather than the second

        return A[: 2 * N, : 2 * N], B[: 2 * N], math.sqrt(C)


    ########################################################################
    ###                     From Wigner to Fock                         ###
    ########################################################################

    def _wignerket_to_fockket(self,
        wignerket: WignerKet,
        max_prob: float = 1.0,
        max_photons: Optional[int] = None,
    ) -> FockKet:
        r"""
        Returns the Fock representation of a Gaussian state in ket form.

        Args:
            wignerket (WignerKet)       : the WignerKet object. 
            max_prob (float)            : the maximum probability of a the state (applies only if 
                                          the ket is returned)
            max_photons (Optional[int]) : the maximum number of photons in the state (applies only 
                                          if the ket is returned)

        Returns:
            FockKet: the fock representation of the ket.
        """

        A, B, C = self._wignerket_to_bargmannket(wignerket)

        if max_photons is None:
            max_photons = sum(wignerket.data.shape) - len(wignerket.data.shape)

        if max_prob < 1.0 or max_photons < sum(wignerket.data.shape) - len(wignerket.data.shape):
            return math.hermite_renormalized_binomial(
                A, B, C, shape=wignerket.data.shape, max_l2=max_prob, global_cutoff=max_photons + 1
            )
        
        else:
            return math.hermite_renormalized(A, B, C, shape=tuple(wignerket.data.shape))


    # TODO : fix function, signature, code and doc don't match: which is correct?
    def _wignerdm_to_fockdm(self, wignerdm: WignerDM) -> FockDM:
        r"""
        Returns the Fock representation of a Gaussian state in density matrix form.

        Args:
            wignerdm (WignerDM): the WignerDM object. 
            max_prob: the maximum probability of a the state (applies only if the ket is returned)
            max_photons: the maximum number of photons in the state (applies only if the ket is returned)

        Returns:
            Tensor: the fock representation
        """

        A, B, C = self._wignerdm_to_bargmanndm(wignerdm)

        return math.hermite_renormalized(A, B, C, shape=wignerdm.data.shape)



    #DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
    def _wigner_to_fock_U(self, X:Batch[Matrix], d:Batch[Vector], shape):
        r"""
        Returns the Fock representation of a Gaussian unitary transformation.
        The index order is out_l, in_l, where in_l is to be contracted with the indices of a ket,
        or with the left indices of a density matrix.

        Args:
            X (Batch[Matrix])   : the X matrix
            d (Batch[Vector])   : the d vector
            shape ()            : the shape of the tensor

        Returns:
            Tensor: the fock representation of the unitary transformation
        """

        A, B, C = self._wigner_to_bargmann_U(X, d)

        return math.hermite_renormalized(A, B, C, shape=tuple(shape))


    #DO NOT TOUCH FOR NOW, THIS IS FOR TRANSFORMATION.
    def _wigner_to_fock_Choi(self, X:Batch[Matrix], Y:Batch[Matrix], d:Batch[Vector], shape):
        r"""
        Returns the Fock representation of a Gaussian Choi matrix.
        The order of choi indices is 
        :math:`[\mathrm{out}_l, \mathrm{in}_l, \mathrm{out}_r, \mathrm{in}_r]`
        where 
        :math:`\mathrm{in}_l` 
        and 
        :math:`\mathrm{in}_r` 
        are to be contracted with the left and right indices of a density matrix.

        Arguments:
            X: the X matrix
            Y: the Y matrix
            d: the d vector
            shape: the shape of the tensor

        Returns:
            Tensor: the fock representation of the Choi matrix
        """

        A, B, C = self._wigner_to_bargmann_Choi(X, Y, d)

        return math.hermite_renormalized(A, B, C, shape=tuple(shape))


    ########################################################################
    ###                     From Bargmann to Fock                        ###
    ########################################################################
    ###!!! Just the math.hermite_renormalized


    ########################################################################
    ###                     From Fock to Wavefunction                    ###
    ########################################################################

    @tensor_int_cache # TODO : check whether this should be private or not
    def _oscillator_eigenstates(self, q: RealVector, cutoff: int) -> Tensor:
        r"""
        Harmonic oscillator eigenstate wavefunctions 
        `\psi_n(q) = <q|n>` for n = 0, 1, 2, ..., cutoff-1.

        Args:
            q (Vector)  : a vector containing the q points at which the function is evaluated 
                          (units of \sqrt{\hbar})
            cutoff (int): maximum number of photons

        Returns:
            Tensor: a tensor of shape ``(cutoff, len(q))``. The entry with index ``[n, j]`` 
            represents the eigenstate evaluated with number of photons ``n`` evaluated at 
            position ``q[j]``, i.e., `\psi_n(q_j) = <q_j|n>`.

        .. details::

            .. admonition:: Definition
                :class: defn

            The q-quadrature eigenstates are defined as

            .. math::

                \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
                    \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

            where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
        """

        def f_hermite_polys(xi): # TODO : un-nest this? if possible
            poly = math.hermite_renormalized(
                R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff)
            return math.cast(poly, "float64")
        
        omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
        x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

        # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
        prefactor = (omega_over_hbar/np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

        # Renormalized physicist hermite polys: Hn / sqrt(n!)
        R = np.array([[2 + 0j]])  # to get the physicist polys

        hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

        # (real) wavefunction
        psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
        return psi


    def _fockket_to_wavefunctionqket(self, 
                                     fock_ket:FockKet, 
                                     qs:Optional[Sequence[Sequence[float]]] = None
                                     ) -> WavefunctionQKet:
        r"""
        Returns the position wavefunction of the Fock ket state at a vector of positions.

        Args:
            fockket (FockKet): a Fock ket object.
            qs (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
                If ``None``, a set of positions is automatically generated.

        Returns:
            ComplexFunctionND: the wavefunction at the given positions wrapped in a
            :class:`~.ComplexFunctionND` object.
        """

        # TODO : the object has no cutoffs, where do we get them from?
        krausses = [math.transpose(self._oscillator_eigenstates(q, c)) for q, c in zip(qs, self.cutoffs)]

        ket = fock_ket.data.array

        for i, h_n in enumerate(krausses):
            ket = fock_ket.apply_kraus_to_ket(h_n, ket, [i])

        return ket  # now in q basis


    def _fockdm_to_wavefunctionqdm(self, 
                                   fock_dm: FockDM, 
                                   qs: Optional[Sequence[Sequence[float]]] = None
                                   ) -> WavaefunctionQDM:
        r"""
        Returns the position wavefunction of the Fock density matrix at a vector of positions.

        Args:
            fockdm (FockDM): a Fock density matrix object.
            qs (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
                If ``None``, a set of positions is automatically generated.

        Returns:
            ComplexFunctionND: the wavefunction at the given positions wrapped in a
            :class:`~.ComplexFunctionND` object.
        """
        # TODO : the object has no cutoffs, where do we get them from?
        krausses = [math.transpose(self._oscillator_eigenstates(q, c)) for q, c in zip(qs, self.cutoffs)]

        dm = fock_dm.data.array
        for i, h_n in enumerate(krausses):
            dm = fock_dm.apply_kraus_to_dm(h_n, dm, [i])
        return dm
