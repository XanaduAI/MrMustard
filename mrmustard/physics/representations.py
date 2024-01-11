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

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from mrmustard import math
from mrmustard.physics import bargmann
from mrmustard.physics.ansatze import Ansatz, PolyExpAnsatz
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Scalar
import numpy as np
from mrmustard import math


class Representation:
    r"""
    A base class for representations.
    """

    def from_ansatz(self, ansatz: Ansatz) -> Representation:
        r"""
        Returns a representation object from an ansatz object.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def __eq__(self, other: Representation) -> bool:
        r"""
        Whether this representation is equal to another.
        """
        return self.ansatz == other.ansatz

    def __add__(self, other: Representation) -> Representation:
        r"""
        Adds this representation to another.
        """
        return self.from_ansatz(self.ansatz + other.ansatz)

    def __sub__(self, other) -> Representation:
        r"""
        Subtracts another representation from this one.
        """
        return self.from_ansatz(self.ansatz - other.ansatz)

    def __mul__(self, other: Representation | Scalar) -> Representation:
        r"""
        Multiplies this representation by another or by a scalar.
        """
        try:
            return self.from_ansatz(self.ansatz * other.ansatz)
        except AttributeError:
            return self.from_ansatz(self.ansatz * other)

    def __rmul__(self, other: Representation | Scalar) -> Representation:
        r"""
        Multiplies this representation by another or by a scalar on the right.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Representation | Scalar) -> Representation:
        r"""
        Divides this representation by another or by a scalar.
        """
        try:
            return self.from_ansatz(self.ansatz / other.ansatz)
        except AttributeError:
            return self.from_ansatz(self.ansatz / other)

    def __rtruediv__(self, other: Representation | Scalar) -> Representation:
        r"""
        Divides this representation by another or by a scalar on the right.
        """
        return self.from_ansatz(other / self.ansatz)

    def __and__(self, other: Representation) -> Representation:
        r"""
        Takes the outer product of this representation with another.
        """
        return self.from_ansatz(self.ansatz & other.ansatz)


class Bargmann(Representation):
    r"""This class is the Fock-Bargmann representation of a broad class of quantum states,
    transformations, measurements, channels, etc.

    The ansatz available in this representation is a linear combination of
    exponentials of bilinear forms with a polynomial part:

    .. math::
        F(z) = \sum_i \textrm{poly}_i(z) \textrm{exp}(z^T A_i z / 2 + z^T b_i)

    This function allows for vector space operations on Bargmann objects including
    linear combinations, outer product, and inner product.
    The inner product is defined as the contraction of two Bargmann objects across
    marked indices. This can also be used to contract existing indices
    in a single Bargmann object, e.g. to implement the partial trace.

    Note that the operations that change the shape of the ansatz (outer product (``&``)
    and inner product (``@``)) do not automatically modify the ordering of the
    combined or leftover indices.

    Examples:
        .. code-block:: python

            A = math.astensor([[[1.0]]])  # 1x1x1
            b = math.astensor([[0.0]])    # 1x1
            c = math.astensor([0.9])      # 1
            psi1 = Bargmann(A, b, c)
            psi2 = Bargmann(A, b, c)
            psi3 = 1.3 * psi1 - 2.1 * psi2  # linear combination
            assert psi3.A.shape == (2, 1, 1)  # stacked along batch dimension
            psi4 = psi1[0] @ psi2[0]  # contract wires 0 on each (inner product)
            assert psi4.A.shape == (1,)  # A is 0x0 now (no wires left)
            psi5 = psi1 & psi2  # outer product (tensor product)
            rho = psi1.conj() & psi1   # outer product (this is now the density matrix)
            assert rho.A.shape == (1, 2, 2)  # we have two wires now
            assert np.allclose(rho.trace((0,), (1,)), np.abs(c)**2)


    Args:
        A: batch of quadratic coefficient :math:`A_i`
        b: batch of linear coefficients :math:`b_i`
        c: batch of arrays :math:`c_i` (default: [1.0])
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
        c: Batch[ComplexTensor] = [1.0],
    ):
        r"""Initializes the Bargmann representation. Args can be passed non-batched,
        they will be automatically broadcasted to the correct batch shape.

        Args:
            A: batch of quadratic coefficient :math:`A_i`
            b: batch of linear coefficients :math:`b_i`
            c: batch of arrays :math:`c_i` (default: [1.0])
        """
        self._contract_idxs: tuple[int, ...] = ()
        self.ansatz = PolyExpAnsatz(A, b, c)

    def __call__(self, z: ComplexTensor) -> ComplexTensor:
        r"""Evaluates the Bargmann function at the given array of points."""
        return self.ansatz(z)

    def from_ansatz(self, ansatz: PolyExpAnsatz) -> Bargmann:
        r"""Returns a Bargmann object from an ansatz object."""
        return self.__class__(ansatz.A, ansatz.b, ansatz.c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The batch of quadratic coefficient :math:`A_i`.
        """
        return self.ansatz.A

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The batch of linear coefficients :math:`b_i`
        """
        return self.ansatz.b

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The batch of arrays :math:`c_i`.
        """
        return self.ansatz.c

    def conj(self):
        r"""
        The conjugate of this Bargmann object.
        """
        new = self.__class__(math.conj(self.A), math.conj(self.b), math.conj(self.c))
        new._contract_idxs = self._contract_idxs
        return new

    def __getitem__(self, idx: int | tuple[int, ...]) -> Bargmann:
        r"""Returns a copy of self with the given indices marked for contraction."""
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.ansatz.dim:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz {self.ansatz.__class__.__qualname__} of dimension {self.ansatz.dim}."
                )
        new = self.__class__(self.A, self.b, self.c)
        new._contract_idxs = idx
        return new

    def __matmul__(self, other: Bargmann) -> Bargmann:
        r"""Implements the inner product of ansatze across the marked indices."""
        if self.ansatz.degree > 0 or other.ansatz.degree > 0:
            raise NotImplementedError(
                "Inner product of ansatze is only supported for ansatze with polynomial of degree 0."
            )
        Abc = []
        for A1, b1, c1 in zip(self.A, self.b, self.c):
            for A2, b2, c2 in zip(other.A, other.b, other.c):
                Abc.append(
                    bargmann.contract_two_Abc(
                        (A1, b1, c1),
                        (A2, b2, c2),
                        self._contract_idxs,
                        other._contract_idxs,
                    )
                )
        A, b, c = zip(*Abc)
        return self.__class__(math.astensor(A), math.astensor(b), math.astensor(c))

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> Bargmann:
        r"""Implements the partial trace over the given index pairs.

        Args:
            idx_z: indices to trace over
            idx_zconj: indices to trace over

        Returns:
            Bargmann: the ansatz with the given indices traced over
        """
        if self.ansatz.degree > 0:
            raise NotImplementedError(
                "Partial trace is only supported for ansatze with polynomial of degree ``0``."
            )
        if len(idx_z) != len(idx_zconj):
            msg = f"The number of indices to trace over must be the same for ``z`` and ``z*`` (got {len(idx_z)} and {len(idx_zconj)})."
            raise ValueError(msg)
        A, b, c = [], [], []
        for Abci in zip(self.A, self.b, self.c):
            Aij, bij, cij = bargmann.complex_gaussian_integral(Abci, idx_z, idx_zconj, measure=-1.0)
            A.append(Aij)
            b.append(bij)
            c.append(cij)
        return self.__class__(math.astensor(A), math.astensor(b), math.astensor(c))

    def reorder(self, order: tuple[int, ...] | list[int]) -> Bargmann:
        r"""Reorders the indices of the A matrix and b vector of an (A,b,c) triple.
        Returns a new Bargmann object."""
        A, b, c = bargmann.reorder_abc((self.A, self.b, self.c), order)
        return self.__class__(A, b, c)

    def plot(
        self,
        just_phase: bool = False,
        with_measure: bool = False,
        log_scale: bool = False,
        xlim=(-2 * np.pi, 2 * np.pi),
        ylim=(-2 * np.pi, 2 * np.pi),
        **kwargs,
    ):  # pragma: no cover
        r"""Plots the Bargmann function F(z) on the complex plane. Phase is represented by color,
        magnitude by brightness. The function can be multiplied by exp(-|z|^2) to represent
        the Bargmann function times the measure function (for integration).

        Args:
            just_phase (bool): whether to plot only the phase of the Bargmann function
            with_measure (bool): whether to plot the bargmann function times the measure function exp(-|z|^2)
            log_scale (bool): whether to plot the log of the Bargmann function
            xlim (tuple[float, float]): x limits of the plot
            ylim (tuple[float, float]): y limits of the plot

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: the figure and axes of the plot
        """
        # eval F(z) on a grid of complex numbers
        X, Y = np.mgrid[xlim[0] : xlim[1] : 400j, ylim[0] : ylim[1] : 400j]
        Z = (X + 1j * Y).T
        f_values = self(Z[..., None])
        if log_scale:
            f_values = np.log(np.abs(f_values)) * np.exp(1j * np.angle(f_values))
        if with_measure:
            f_values = f_values * np.exp(-np.abs(Z) ** 2)

        # Get phase and magnitude of F(z)
        phases = np.angle(f_values) / (2 * np.pi) % 1
        magnitudes = np.abs(f_values)
        magnitudes_scaled = magnitudes / np.max(magnitudes)

        # Convert to RGB
        hsv_values = np.zeros(f_values.shape + (3,))
        hsv_values[..., 0] = phases
        hsv_values[..., 1] = 1
        hsv_values[..., 2] = 1 if just_phase else magnitudes_scaled
        rgb_values = colors.hsv_to_rgb(hsv_values)

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(rgb_values, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        ax.set_xlabel("$Re(z)$")
        ax.set_ylabel("$Im(z)$")

        name = "F_{" + self.ansatz.name + "}(z)"
        name = f"\\arg({name})\\log|{name}|" if log_scale else name
        title = name + "e^{-|z|^2}" if with_measure else name
        title = f"\\arg({name})" if just_phase else title
        ax.set_title(f"${title}$")
        plt.show(block=False)
        return fig, ax
