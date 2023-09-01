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
from typing import List, Optional, Tuple, Union, Sequence
from mrmustard import settings
from mrmustard.math import Math
from mrmustard.physics.fock import apply_kraus_to_ket, apply_kraus_to_dm
from mrmustard.math.caching import tensor_int_cache
from mrmustard.typing import Matrix, RealVector, Tensor, Vector
from mrmustard.lab.representations.representation import Representation
from mrmustard.lab.representations.bargmann_ket import BargmannKet
from mrmustard.lab.representations.bargmann_dm import BargmannDM
from mrmustard.lab.representations.fock_ket import FockKet
from mrmustard.lab.representations.fock_dm import FockDM
from mrmustard.lab.representations.wavefunction_ket import WaveFunctionKet
from mrmustard.lab.representations.wavefunction_dm import WaveFunctionDM
from mrmustard.lab.representations.wigner_ket import WignerKet
from mrmustard.lab.representations.wigner_dm import WignerDM
from mrmustard.lab import State, Rgate

math = Math()


class Converter:
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
            - Cyclic
            - Unweighted
            - Order : 8
            - Size : 6
            - Degree 1 for all nodes
        """
    def __init__(self) -> None:
        ### DEFINE NODES - REPRESENTATION NAMES

        # Ket part of the graph
        wigner_K = "WignerKet"
        bargmann_K = "BargmannKet"
        fock_K = "FockKet"
        wavefunction_K = "WaveFunctionKet"

        # DM component of the graph
        wigner_DM = "WignerDM"
        bargmann_DM = "BargmannDM"
        fock_DM = "FockDM"
        wavefunction_DM = "WaveFunctionDM"
       

        ### DEFINE EDGES - CONNECTIONS

        # Ket component of the graph
        w2b_K = (wigner_K, bargmann_K)
        b2f_K = (bargmann_K, fock_K)
        w2f_K = (wigner_K, fock_K)
        f2wf_K = (fock_K, wavefunction_K)

        # DM component of the graph
        w2b_DM = (wigner_DM, bargmann_DM)
        b2f_DM = (bargmann_DM, fock_DM)
        w2f_DM = (wigner_DM, fock_DM)
        f2wf_DM = (fock_DM, wavefunction_DM)

        ### DEFINE EDGE LABELS - FORMULAS

        # Ket component of the graph
        w2b_K = self._wignerket_to_bargmannket
        b2f_K = self._bargmannket_to_fockket
        w2f_K = self._wignerket_to_fockket
        f2wf_K = self._fockket_to_wavefunctionket

        # DM component of the graph
        w2b_DM = self._wignerdm_to_bargmanndm
        b2f_DM = self._bargmanndm_to_fockdm
        w2f_DM = self._wignerdm_to_fockdm
        f2wf_DM = self._fockdm_to_wavefunctiondm

        ### DEFINE GRAPH

        edges = [w2b_K, b2f_K, w2f_K, f2wf_K, w2b_DM, b2f_DM, w2f_DM, f2wf_DM]

        transition_formulas = {
            w2b_K: {"f": f_w2b_K},
            b2f_K: {"f": f_b2f_K},
            w2f_K: {"f": f_w2f_K},
            f2wf_K: {"f": f_f2wf_K},
            w2b_DM: {"f": f_w2b_DM},
            b2f_DM: {"f": f_b2f_DM},
            w2f_DM: {"f": f_w2f_DM},
            f2wf_DM: {"f": f_f2wf_DM},
        }

        self.g = nx.DiGraph()

        self.g.add_edges_from(edges)

        nx.set_edge_attributes(self.g, transition_formulas)

    def _find_target_node_name(self, source: str, destination: str) -> str:
        r""" " Given source and destination names, returns the name of the target node in the graph.

        Args:
            source:         the class name of the source Representation, containing the Ket/DM
                            suffix
            destination:    the name of the target Representation without the 'Ket'/'DM' suffix

        Raises:
            ValueError:     if the source name doesn't contain the desired 'Ket' or "DM' suffix

        Returns:
            The string of the target representation concatenated with either 'Ket' or 'DM'
            depending on the source.
        """

        if source.endswith("Ket"):
            return destination + "Ket"
        elif source.endswith("DM"):
            return destination + "DM"
        else:
            raise ValueError("Invalid input: source name must contain 'Ket' or 'DM'.")

    def convert(self, source: Representation, destination: str, **kwargs) -> Representation:
        r"""Converts from a source Representation to target Representation.

        .. code-block::
            # assuming we have some State object s
            c = Converter()
            new_s = c.convert(source=s, destination="Fock")

        Args:
            source:         the state which representation must be transformed
            destination:    the name of the target prepresentation, this name must NOT include
                            'Ket'/'DM'

        Raises:
            ValueError:     if the destination or the source names are not supported, aka either
                            not in the graph or the source contains neither 'Ket' nor 'DM'.

        Returns:
            The target representation
        """
        try:
            s_name = source.__class__.__name__
            d_name = self._find_target_node_name(source=s_name, destination=destination)
            f = self.g[s_name][d_name]["f"]

            if s_name == "WignerKet" and d_name == "FockKet":
                max_prob = kwargs.get("max_prob") if kwargs.get("max_prob") else 1.0
                max_photon = kwargs.get("max_photon") if kwargs.get("max_photon") else None
                cutoffs = kwargs.get("cutoffs") if kwargs.get("cutoffs") else None
                return f(source, max_prob=max_prob, max_photon=max_photon, cutoffs=cutoffs)
            elif s_name == "WignerDM" and d_name == "FockDM":
                cutoffs = kwargs.get("cutoffs") if kwargs.get("cutoffs") else None
                return f(source, cutoffs=cutoffs)
            elif d_name == "WaveFunctionQKet" or d_name == "WaveFunctionQDM":
                qs = kwargs.get("qs")
                return f(source, qs=qs)
            else:
                return f(source)

        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Either {s_name} or {destination} is not a valid representation name"
            ) from e

    def shortest_path(self, source: Representation, destination: Representation):
        raise NotImplementedError

    def add_edge(self):
        raise NotImplementedError

    def show(self) -> None:
        raise NotImplementedError

    ########################################################################
    ###                    From Wigner to Husimi                         ###
    ########################################################################
    def _pq_to_aadag(self, X: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        r"""
        Maps a matrix or vector from the q/p basis to the a/adagger basis

        Args:
            X (Union[Matrix, Vector]) : A matrix or vector in the Q/P basis

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

    def _wigner_to_husimi(self, cov: Matrix, means: Vector) -> Tuple[Matrix, Vector]:
        r"""
        Converts from a Wigner Representation (covariance and mean vector) to Husimi Representation.

        Args:
            cov (Matrix)     : covariance matrix of the state
            means (Vector)   : mean vector of the state

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

    def _cayley(self, X: Tensor, c: float) -> Tensor:
        r"""
        Returns the Cayley transformation of a matrix:
        :math:`cay(X) = (X - cI)(X + cI)^{-1}`

        Args:
            c (float): the parameter of the self._cayley transform
            X (Tensor): a matrix

        Returns:
            The cayley transformation of X tensor with respect to c
        """

        I = math.eye(X.shape[0], dtype=X.dtype)

        return math.solve(X + c * I, X - c * I)  # TODO : solve the argument nb issue

    def _wignerdm_to_bargmanndm(self, wigner_dm: WignerDM) -> BargmannDM:
        r"""
        Converts the wigner representation in terms of covariance matrix and mean vector into the
        Bargmann `A,B,C` triple for a density matrix (i.e. for `M` modes, `A` has shape `2M x 2M`
        and `B` has shape `2M`).
        The order of the rows/columns of A and B corresponds to a density matrix with the usual
        ordering of the indices.

        Note that here A and B are defined with respect to the literature.

        Args:
            wigner_dm (WignerDM) : the Wigner DM representation of the state

        Returns:
            The BargmannDM representation of the input state

        """
        N = wigner_dm.data.cov.shape[-1] // 2

        A = math.matmul(math.Xmat(N), self._cayley(self._pq_to_aadag(wigner_dm.data.cov), c=0.5))
        Q, beta = self._wigner_to_husimi(wigner_dm.data.cov, wigner_dm.data.means)
        b = math.solve(Q, beta)
        B = math.conj(b)
        C = math.exp(-0.5 * math.sum(math.conj(beta) * b)) / math.sqrt(math.det(Q))

        return BargmannDM(A=A, b=B, c=C)

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

        N = wigner_ket.data.symplectic.shape[-1] // 2
        cov = wigner_ket.cov
        means = wigner_ket.means
        wigner_dm = WignerDM(cov=cov, means=means)
        bargmann_dm = self._wignerdm_to_bargmanndm(wigner_dm=wigner_dm)

        return BargmannKet(
            bargmann_dm.data.A[N:, N:], bargmann_dm.data.b[N:], math.sqrt(bargmann_dm.data.c)
        )

    ########################################################################
    ###                     From Bargmann to Fock                         ###
    ########################################################################

    def _bargmannket_to_fockket(
        self,
        bargmannket: BargmannKet,
        max_prob: float = 1.0,
        max_photon: Optional[int] = None,
        cutoffs: List[int] = None,
    ) -> FockKet:
        r"""
        Returns the Fock representation of a Gaussian state in ket form.

        Args:
            bargmannket (BargmannKet)       : the BargmannKet object.
            cutoffs (List[int]) .       : the shape of the desired Fock tensor
            max_prob (float)            : the maximum probability of a the state (applies only if
                                          the ket is returned)
            max_photon (Optional[int]) : the maximum number of photons in the state (applies only
                                          if the ket is returned)

        Returns:
            FockKet: the fock representation of the ket.
        """
        A = bargmannket.data.A
        B = bargmannket.data.b
        C = bargmannket.data.c

        if cutoffs is None:
            cutoffs = np.repeat(
                settings.AUTOCUTOFF_MIN_CUTOFF, bargmannket.data.A.shape[-1] // 2
            )

        if max_photon is None:
            max_photon = sum(np.shape(cutoffs)) - len(np.shape(cutoffs))

        if max_prob < 1.0 or max_photon < sum(np.shape(cutoffs)) - len(np.shape(cutoffs)):
            return FockKet(
                array=math.hermite_renormalized_binomial(
                    A, B, C, shape=cutoffs.shape, max_l2=max_prob, global_cutoff=max_photon + 1
                )
            )

        else:
            return FockKet(array=math.hermite_renormalized(A, B, C, shape=tuple(cutoffs)))

    def _bargmanndm_to_fockdm(
        self,
        bargmanndm: BargmannDM,
        cutoffs: List[int] = None,
    ) -> FockDM:
        r"""
        Returns the Fock representation of a Gaussian state in density matrix form.

        Args:
            bargmanndm (BargmannDM): the BargmannDM object.
            cutoffs (List[int]): the shape of the desired Fock tensor

        Returns:
            Tensor: the fock representation
        """
        A = bargmanndm.data.A
        B = bargmanndm.data.b
        C = bargmanndm.data.c

        if cutoffs is None:
            cutoffs = np.repeat(settings.AUTOCUTOFF_MIN_CUTOFF, bargmanndm.data.A.shape[-1] // 2)

        return FockDM(array=math.hermite_renormalized(A, B, C, shape=tuple(cutoffs.shape)))
    

    ########################################################################
    ###                     From Wigner to Fock                         ###
    ########################################################################

    def _wignerket_to_fockket(
        self,
        wignerket: WignerKet,
        max_prob: float = 1.0,
        max_photon: Optional[int] = None,
        cutoffs: List[int] = None,
    ) -> FockKet:
        r"""
        Returns the Fock representation of a Gaussian state in ket form.

        Args:
            wignerket (WignerKet)       : the WignerKet object.
            cutoffs (List[int]) .       : the shape of the desired Fock tensor
            max_prob (float)            : the maximum probability of a the state (applies only if
                                          the ket is returned)
            max_photon (Optional[int]) : the maximum number of photons in the state (applies only
                                          if the ket is returned)

        Returns:
            FockKet: the fock representation of the ket.
        """
        bargmann_ket = self._wignerket_to_bargmannket(wignerket)
        return self._bargmannket_to_fockket(bargmannket=bargmann_ket, max_prob=max_prob, max_photon=max_photon, cutoffs=cutoffs)

    def _wignerdm_to_fockdm(
        self,
        wignerdm: WignerDM,
        cutoffs: List[int] = None,
    ) -> FockDM:
        r"""
        Returns the Fock representation of a Gaussian state in density matrix form.

        Args:
            wignerdm (WignerDM): the WignerDM object.
            cutoffs (List[int]): the shape of the desired Fock tensor

        Returns:
            Tensor: the fock representation
        """
        bargmann_dm = self._wignerdm_to_bargmanndm(wignerdm)
        return self._bargmanndm_to_fockdm(bargmanndm=bargmann_dm, cutoffs=cutoffs)

    ########################################################################
    ###                     From Fock to Wavefunction                    ###
    ########################################################################

    @tensor_int_cache
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
                    \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\hbar}} x)

            where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
        """

        def f_hermite_polys(xi):  # TODO : un-nest this? if possible
            poly = math.hermite_renormalized(
                R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff
            )
            return math.cast(poly, "float64")

        omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
        x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

        # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
        prefactor = (omega_over_hbar / np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

        # Renormalized physicist hermite polys: Hn / sqrt(n!)
        R = np.array([[2 + 0j]])  # to get the physicist polys

        hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

        # (real) wavefunction
        psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
        return psi

    def _fockket_to_wavefunctionket(
        self, fock_ket: FockKet, points: Optional[Sequence[Sequence[float]]] = None, quadrature_angle: float = 0.0
    ) -> WaveFunctionKet:
        r"""
        Returns the position wavefunction of the Fock ket state at a vector of positions.

        Args:
            fockket (FockKet): a Fock ket object.
            points (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
            quadrature_angle (float): the angle indicates the wavefunction along with axis. For example, angle=0.0 along q-axis, angle=np.pi/2 along p-axis.

        Returns:
            ComplexFunctionND: the wavefunction at the given positions wrapped in a
            :class:`~.ComplexFunctionND` object.
        """
        if not points:
            raise ValueError(
                "The number of points are necessary to generate the wavefunction state."
            )
        if not quadrature_angle:
            raise ValueError(
                "The axis of wavefunction along with needs to be given."
            )

        fock_ket_new = (State(ket=fock_ket.data.array) >> Rgate(quadrature_angle)).ket(cutoffs=fock_ket.data.array.shape)

        krausses = [
            math.transpose(self._oscillator_eigenstates(q, c))
            for q, [c] in zip(points, fock_ket_new.data.cutoffs)
        ]

        ket = fock_ket_new.data.array

        for i, h_n in enumerate(krausses):
            ket = apply_kraus_to_ket(h_n, ket, [i])

        return ket  # now in q basis

    def _fockdm_to_wavefunctiondm(
        self, fock_dm: FockDM, points: Optional[Sequence[Sequence[float]]] = None, quadrature_angle: float = 0.0
    ) -> WaveFunctionDM:
        r"""
        Returns the position wavefunction of the Fock density matrix at a vector of positions.

        Args:
            fockdm (FockDM): a Fock density matrix object.
            points (optional Sequence[Sequence[float]]): a sequence of positions for each mode.
            quadrature_angle (float): the angle indicates the wavefunction along with axis. For example, angle=0.0 along q-axis, angle=np.pi/2 along p-axis.

        Returns:
            ComplexFunctionND: the wavefunction at the given positions wrapped in a
            :class:`~.ComplexFunctionND` object.
        """
        if not points:
            raise AssertionError(
                "The number of points are necessary to generate the wavefunction state."
            )
        if not quadrature_angle:
            raise ValueError(
                "The axis of wavefunction along with needs to be given."
            )
        
        fock_dm_new = (State(dm=fock_dm.data.array) >> Rgate(quadrature_angle)).dm(cutoffs=fock_dm.data.array.shape)

        krausses = [
            math.transpose(self._oscillator_eigenstates(q, c))
            for q, [c] in zip(points, fock_dm_new.data.cutoffs)
        ]

        dm = fock_dm_new.data.array
        for i, h_n in enumerate(krausses):
            dm = apply_kraus_to_dm(h_n, dm, [i])
        return dm
