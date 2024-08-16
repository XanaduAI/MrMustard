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


"""
This module contains the classes for the available representations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Union

import numpy as np
from numpy.typing import ArrayLike

from matplotlib import colors
import matplotlib.pyplot as plt

from IPython.display import display

from mrmustard import math, settings
from mrmustard.physics.gaussian_integrals import (
    contract_two_Abc_poly,
    reorder_abc,
    complex_gaussian_integral,
)
from mrmustard.physics.ansatze import Ansatz, PolyExpAnsatz, ArrayAnsatz
from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Scalar,
    Tensor,
)
from mrmustard import widgets

__all__ = ["Representation", "Bargmann", "Fock"]


class Representation(ABC):
    r"""
    A base class for representations.

    Representations can be initialized using the ``from_ansatz`` method, which automatically equips
    them with all the functionality required to perform mathematical operations, such as equality,
    multiplication, subtraction, etc.
    """

    @property
    @abstractmethod
    def ansatz(self) -> Ansatz:
        r"""
        The ansatz of the representation.
        """

    @property
    @abstractmethod
    def data(self) -> tuple | Tensor:
        r"""
        The data of the representation.
        For now, it's the triple for Bargmann and the array for Fock.
        """

    @property
    @abstractmethod
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the representation.
        For now it's ``c`` for Bargmann and the array for Fock.
        """

    @classmethod
    @abstractmethod
    def from_ansatz(cls, ansatz: Ansatz) -> Representation:  # pragma: no cover
        r"""
        Returns a representation from an ansatz.
        """

    @abstractmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Representation:
        r"""
        Returns a representation from a function and kwargs.
        """

    @abstractmethod
    def reorder(self, order: tuple[int, ...] | list[int]) -> Representation:
        r"""
        Reorders the representation indices.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, ArrayLike]:
        r"""Serialize a Representation."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Representation:
        r"""Deserialize a Representation."""

    def __eq__(self, other: Representation) -> bool:
        r"""
        Whether this representation is equal to another.
        """
        return self.ansatz == other.ansatz

    def __add__(self, other: Representation) -> Representation:
        r"""
        Adds this representation to another.
        """
        if self.__class__.__name__ != other.__class__.__name__:
            msg = f"Cannot add ``{self.__class__.__name__}`` representation to "
            msg += f"``{other.__class__.__name__}`` representation."
            raise ValueError(msg)
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

    def __getitem__(self, idx: int | tuple[int, ...]) -> Representation:
        r"""
        Stores the indices for contraction.
        """
        raise NotImplementedError


class Bargmann(Representation):
    r"""
    The Fock-Bargmann representation of a broad class of quantum states, transformations,
    measurements, channels, etc.

    The ansatz available in this representation is a linear combination of exponentials
    of bilinear forms with a polynomial part:

    .. math::
        F(z) = \sum_i \textrm{poly}_i(z) \textrm{exp}(z^T A_i z / 2 + z^T b_i)

    This function allows for vector space operations on Bargmann objects including
    linear combinations (``+``), outer product (``&``), and inner product (``@``).

    .. code-block ::

        >>> from mrmustard.physics.representations import Bargmann
        >>> from mrmustard.physics.triples import displacement_gate_Abc, vacuum_state_Abc

        >>> # bargmann representation of one-mode vacuum
        >>> rep_vac = Bargmann(*vacuum_state_Abc(1))

        >>> # bargmann representation of one-mode dgate with gamma=1+0j
        >>> rep_dgate = Bargmann(*displacement_gate_Abc(1))

    The inner product is defined as the contraction of two Bargmann objects across marked indices.
    Indices are marked using ``__getitem__``. Once the indices are marked for contraction, they are
    be used the next time the inner product (``@``) is called. For example:

    .. code-block ::

        >>> import numpy as np

        >>> # mark indices for contraction
        >>> idx_vac = [0]
        >>> idx_rep = [1]

        >>> # bargmann representation of coh = vacuum >> dgate
        >>> rep_coh = rep_vac[idx_vac] @ rep_dgate[idx_rep]
        >>> assert np.allclose(rep_coh.A, [[0,],])
        >>> assert np.allclose(rep_coh.b, [1,])
        >>> assert np.allclose(rep_coh.c, 0.6065306597126334)

    This can also be used to contract existing indices in a single Bargmann object, e.g.
    to implement the partial trace.

    .. code-block ::

        >>> trace = (rep_coh @ rep_coh.conj()).trace([0], [1])
        >>> assert np.allclose(trace.A, 0)
        >>> assert np.allclose(trace.b, 0)
        >>> assert trace.c == 1

    The ``A``, ``b``, and ``c`` parameters can be batched to represent superpositions.

    .. code-block ::

        >>> # bargmann representation of one-mode coherent state with gamma=1+0j
        >>> A_plus = [[0,],]
        >>> b_plus = [1,]
        >>> c_plus = 0.6065306597126334

        >>> # bargmann representation of one-mode coherent state with gamma=-1+0j
        >>> A_minus = [[0,],]
        >>> b_minus = [-1,]
        >>> c_minus = 0.6065306597126334

        >>> # bargmann representation of a superposition of coherent states
        >>> A = [A_plus, A_minus]
        >>> b = [b_plus, b_minus]
        >>> c = [c_plus, c_minus]
        >>> rep_coh_sup = Bargmann(A, b, c)

    Note that the operations that change the shape of the ansatz (outer product and inner
    product) do not automatically modify the ordering of the combined or leftover indices.
    However, the ``reordering`` method allows reordering the representation after the products
    have been carried out.

    Args:
        A: A batch of quadratic coefficient :math:`A_i`.
        b: A batch of linear coefficients :math:`b_i`.
        c: A batch of arrays :math:`c_i`.

    Note: The args can be passed non-batched, as they will be automatically broadcasted to the
    correct batch shape.
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
        c: Batch[ComplexTensor] = 1.0,
    ):
        self._contract_idxs: tuple[int, ...] = ()
        self._ansatz = PolyExpAnsatz(A=A, b=b, c=c)

    @property
    def ansatz(self) -> PolyExpAnsatz:
        r"""
        The ansatz of the representation.
        """
        return self._ansatz

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

    @property
    def data(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The data of the representation.
        """
        return self.triple

    @property
    def scalar(self) -> Batch[ComplexTensor]:
        r"""
        The scalar part of the representation.
        """
        return self.c

    @property
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The batch of triples :math:`(A_i, b_i, c_i)`.
        """
        return self.A, self.b, self.c

    @classmethod
    def from_ansatz(cls, ansatz: PolyExpAnsatz) -> Bargmann:  # pylint: disable=arguments-differ
        r"""
        Returns a Bargmann object from an ansatz object.
        """
        return cls(ansatz.A, ansatz.b, ansatz.c)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Bargmann:
        r"""
        Returns a Bargmann object from a generator function.
        """
        return cls.from_ansatz(PolyExpAnsatz.from_function(fn, **kwargs))

    def conj(self):
        r"""
        The conjugate of this Bargmann object.
        """
        new = self.__class__(math.conj(self.A), math.conj(self.b), math.conj(self.c))
        new._contract_idxs = self._contract_idxs  # pylint: disable=protected-access
        return new

    def plot(
        self,
        just_phase: bool = False,
        with_measure: bool = False,
        log_scale: bool = False,
        xlim=(-2 * np.pi, 2 * np.pi),
        ylim=(-2 * np.pi, 2 * np.pi),
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:  # pragma: no cover
        r"""
        Plots the Bargmann function :math:`F(z)` on the complex plane. Phase is represented by
        color, magnitude by brightness. The function can be multiplied by :math:`exp(-|z|^2)`
        to represent the Bargmann function times the measure function (for integration).

        Args:
            just_phase: Whether to plot only the phase of the Bargmann function.
            with_measure: Whether to plot the bargmann function times the measure function
                :math:`exp(-|z|^2)`.
            log_scale: Whether to plot the log of the Bargmann function.
            xlim: The `x` limits of the plot.
            ylim: The `y` limits of the plot.

        Returns:
            The figure and axes of the plot
        """
        # eval F(z) on a grid of complex numbers
        X, Y = np.mgrid[xlim[0] : xlim[1] : 400j, ylim[0] : ylim[1] : 400j]
        Z = (X + 1j * Y).T
        f_values = self(Z[..., None])
        if log_scale:
            f_values = np.log(np.abs(f_values)) * np.exp(1j * np.angle(f_values))
        if with_measure:
            f_values = f_values * np.exp(-(np.abs(Z) ** 2))

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

    def reorder(self, order: tuple[int, ...] | list[int]) -> Bargmann:
        r"""
        Reorders the indices of the ``A`` matrix and ``b`` vector of the ``(A, b, c)`` triple in
        this Bargmann object.

        .. code-block::

            >>> from mrmustard.physics.representations import Bargmann
            >>> from mrmustard.physics.triples import displacement_gate_Abc

            >>> rep_dgate1 = Bargmann(*displacement_gate_Abc([0.1, 0.2, 0.3]))
            >>> rep_dgate2 = Bargmann(*displacement_gate_Abc([0.2, 0.3, 0.1]))

            >>> assert rep_dgate1.reorder([1, 2, 0, 4, 5, 3]) == rep_dgate2

        Args:
            order: The new order.

        Returns:
            The reordered Bargmann object.
        """
        A, b, c = reorder_abc((self.A, self.b, self.c), order)
        return self.__class__(A, b, c)

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> Bargmann:
        r"""
        The partial trace over the given index pairs.

        Args:
            idx_z: The first part of the pairs of indices to trace over.
            idx_zconj: The second part.

        Returns:
            Bargmann: the ansatz with the given indices traced over
        """
        if self.ansatz.degree > 0:
            raise NotImplementedError(
                "Partial trace is only supported for ansatze with polynomial of degree ``0``."
            )
        A, b, c = [], [], []
        for Abc in zip(self.A, self.b, self.c):
            Aij, bij, cij = complex_gaussian_integral(Abc, idx_z, idx_zconj, measure=-1.0)
            A.append(Aij)
            b.append(bij)
            c.append(cij)
        return Bargmann(A, b, c)

    def __call__(self, z: ComplexTensor) -> ComplexTensor:
        r"""
        Evaluates the Bargmann function at the given array of points.

        Args:
            z: The array of points.

        Returns:
            The value of the Bargmann function at ``z``.
        """
        return self.ansatz(z)

    def __getitem__(self, idx: int | tuple[int, ...]) -> Bargmann:
        r"""
        A copy of self with the given indices marked for contraction.
        """
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.ansatz.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz {self.ansatz.__class__.__qualname__} of dimension {self.ansatz.num_vars}."
                )
        new = self.__class__(self.A, self.b, self.c)
        new._contract_idxs = idx
        return new

    def __matmul__(self, other: Bargmann) -> Bargmann:
        r"""
        Implements the inner product in Bargmann representation.

        ..code-block::

        >>> from mrmustard.physics.representations import Bargmann
        >>> from mrmustard.physics.triples import displacement_gate_Abc, vacuum_state_Abc
        >>> rep1 = Bargmann(*vacuum_state_Abc(1))
        >>> rep2 = Bargmann(*displacement_gate_Abc(1))
        >>> rep3 = rep1[0] @ rep2[1]
        >>> assert np.allclose(rep3.A, [[0,],])
        >>> assert np.allclose(rep3.b, [1,])

         Args:
             other: Another Bargmann representation.

         Returns:
            Bargmann: the resulting Bargmann representation.

        """
        if isinstance(other, Fock):
            raise NotImplementedError("Only matmul Bargmann with Bargmann")

        idx_s = self._contract_idxs
        idx_o = other._contract_idxs

        Abc = []
        if settings.UNSAFE_ZIP_BATCH:
            if self.ansatz.batch_size != other.ansatz.batch_size:
                raise ValueError(
                    f"Batch size of the two ansatze must match since the settings.UNSAFE_ZIP_BATCH is {settings.UNSAFE_ZIP_BATCH}."
                )
            for (A1, b1, c1), (A2, b2, c2) in zip(
                zip(self.A, self.b, self.c), zip(other.A, other.b, other.c)
            ):
                Abc.append(contract_two_Abc_poly((A1, b1, c1), (A2, b2, c2), idx_s, idx_o))
        else:
            for A1, b1, c1 in zip(self.A, self.b, self.c):
                for A2, b2, c2 in zip(other.A, other.b, other.c):
                    Abc.append(contract_two_Abc_poly((A1, b1, c1), (A2, b2, c2), idx_s, idx_o))

        A, b, c = zip(*Abc)
        return Bargmann(A, b, c)

    def to_dict(self) -> dict[str, ArrayLike]:
        """Serialize a Bargmann instance."""
        return {"A": self.A, "b": self.b, "c": self.c}

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Bargmann:
        """Deserialize a Bargmann instance."""
        return cls(**data)

    def _ipython_display_(self):
        display(widgets.bargmann(self))


class Fock(Representation):
    r"""
    The Fock representation of a broad class of quantum states, transformations, measurements,
    channels, etc.

    The ansatz available in this representation is ``ArrayAnsatz``.

    This function allows for vector space operations on Fock objects including
    linear combinations, outer product (``&``), and inner product (``@``).

    .. code-block::

        >>> from mrmustard.physics.representations import Fock

        >>> # initialize Fock objects
        >>> array1 = np.random.random((5,7,8))
        >>> array2 = np.random.random((5,7,8))
        >>> array3 = np.random.random((3,5,7,8)) # where 3 is the batch.
        >>> fock1 = Fock(array1)
        >>> fock2 = Fock(array2)
        >>> fock3 = Fock(array3, batched=True)

        >>> # linear combination can be done with the same batch dimension
        >>> fock4 = 1.3 * fock1 - fock2 * 2.1

        >>> # division by a scalar
        >>> fock5 = fock1 / 1.3

        >>> # inner product by contracting on marked indices
        >>> fock6 = fock1[2] @ fock3[2]

        >>> # outer product (tensor product)
        >>> fock7 = fock1 & fock3

        >>> # conjugation
        >>> fock8 = fock1.conj()

    Args:
        array: the (batched) array in Fock representation.
        batched: whether the array input has a batch dimension.

    Note: The args can be passed non-batched, as they will be automatically broadcasted to the
    correct batch shape.

    """

    def __init__(self, array: Batch[Tensor], batched=False):
        self._contract_idxs: tuple[int, ...] = ()
        self._original_bargmann_data = None
        self._ansatz = ArrayAnsatz(array=array, batched=batched)

    @property
    def ansatz(self) -> ArrayAnsatz:
        r"""
        The ansatz of the representation.
        """
        return self._ansatz

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array from the ansatz.
        """
        return self.ansatz.array

    @property
    def data(self) -> Batch[Tensor]:
        r"""
        The data of the representation.
        """
        return self.array

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the representation.
        I.e. the vacuum component of the Fock object, whatever it may be.
        Given that the first axis of the array is the batch axis, this is the first element of the array.
        """
        return self.array[(slice(None),) + (0,) * self.ansatz.num_vars]

    @property
    def triple(self) -> tuple:
        r"""
        The data of the original Bargmann representation if it exists
        """
        if self._original_bargmann_data is None:
            raise AttributeError(
                "This Fock object does not have an original Bargmann representation."
            )
        return self._original_bargmann_data

    @classmethod
    def from_ansatz(cls, ansatz: ArrayAnsatz) -> Fock:  # pylint: disable=arguments-differ
        r"""
        Returns a Fock object from an ansatz object.
        """
        return cls(ansatz.array, batched=True)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Fock:
        r"""
        Returns a Fock object from a generator function.
        """
        return cls.from_ansatz(ArrayAnsatz.from_function(fn, **kwargs))

    def conj(self):
        r"""
        The conjugate of this Fock object.
        """
        new = self.from_ansatz(self.ansatz.conj)
        new._contract_idxs = self._contract_idxs  # pylint: disable=protected-access
        return new

    def reduce(self, shape: Union[int, Iterable[int]]) -> Fock:
        r"""
        Returns a new ``Fock`` with a sliced array.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.representations import Fock

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = Fock(array1)

            >>> fock2 = fock1.reduce(3)
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce(2)
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == Fock(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == Fock(array4)

        Args:
            shape: The shape of the array of the returned ``Fock``.
        """
        return self.from_ansatz(self.ansatz.reduce(shape))

    def reorder(self, order: tuple[int, ...] | list[int]) -> Fock:
        r"""
        Reorders the indices of the array with the given order.

        Args:
            order: The order. Does not need to refer to the batch dimension.

        Returns:
            The reordered Fock.
        """
        return self.from_ansatz(
            ArrayAnsatz(math.transpose(self.array, [0] + [i + 1 for i in order]))
        )

    def sum_batch(self) -> Fock:
        r"""
        Sums over the batch dimension of the array. Turns an object with any batch size to a batch size of 1.

        Returns:
            The collapsed Fock object.
        """
        return self.from_ansatz(ArrayAnsatz(math.expand_dims(math.sum(self.array, axes=[0]), 0)))

    def trace(self, idxs1: tuple[int, ...], idxs2: tuple[int, ...]) -> Fock:
        r"""
        Implements the partial trace over the given index pairs.

        Args:
            idxs1: The first part of the pairs of indices to trace over.
            idxs2: The second part.

        Returns:
            The traced-over Fock object.
        """
        if len(idxs1) != len(idxs2) or not set(idxs1).isdisjoint(idxs2):
            raise ValueError("idxs must be of equal length and disjoint")
        order = (
            [0]
            + [i + 1 for i in range(len(self.array.shape) - 1) if i not in idxs1 + idxs2]
            + [i + 1 for i in idxs1]
            + [i + 1 for i in idxs2]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idxs2) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idxs1)] + (n, n))
        trace = math.trace(new_array)
        return self.from_ansatz(ArrayAnsatz([trace] if trace.shape == () else trace))

    def __getitem__(self, idx: int | tuple[int, ...]) -> Fock:
        r"""
        Returns a copy of self with the given indices marked for contraction.
        """
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= len(self.array.shape):
                raise IndexError(
                    f"Index {i} out of bounds for ansatz {self.ansatz.__class__.__qualname__} with {self.ansatz.num_vars} variables."
                )
        new = self.from_ansatz(self.ansatz)
        new._contract_idxs = idx
        return new

    def __matmul__(self, other: Fock) -> Fock:
        r"""
        Implements the inner product of fock arrays over the marked indices.

        .. code-block::
            >>> from mrmustard.physics.representations import Fock
            >>> f = Fock(np.random.random((3, 5, 10)))  # 10 is reduced to 8
            >>> g = Fock(np.random.random((2, 5, 8)))
            >>> h = f[1,2] @ g[1,2]
            >>> assert h.array.shape == (1,3,2)  # batch size is 1
            >>> f = Fock(np.random.random((3, 5, 10)), batched=True)
            >>> g = Fock(np.random.random((2, 5, 8)), batched=True)
            >>> h = f[0,1] @ g[0,1]
            >>> assert h.array.shape == (6,)  # batch size is 3 x 2 = 6

        Args:
            other: Another representation.

        Returns:
            A ``Fock``representation.
        """
        if isinstance(other, Bargmann):
            raise NotImplementedError("only matmul Fock with Fock")

        idx_s = list(self._contract_idxs)
        idx_o = list(other._contract_idxs)

        # the number of batches in self and other
        n_batches_s = self.array.shape[0]
        n_batches_o = other.array.shape[0]

        # the shapes each batch in self and other
        shape_s = self.array.shape[1:]
        shape_o = other.array.shape[1:]

        new_shape_s = list(shape_s)
        new_shape_o = list(shape_o)
        for s, o in zip(idx_s, idx_o):
            new_shape_s[s] = min(shape_s[s], shape_o[o])
            new_shape_o[o] = min(shape_s[s], shape_o[o])

        reduced_s = self.reduce(new_shape_s)[idx_s]
        reduced_o = other.reduce(new_shape_o)[idx_o]

        axes = [list(idx_s), list(idx_o)]
        batched_array = []
        for i in range(n_batches_s):
            for j in range(n_batches_o):
                batched_array.append(math.tensordot(reduced_s.array[i], reduced_o.array[j], axes))
        return self.from_ansatz(ArrayAnsatz(batched_array))

    def to_dict(self) -> dict[str, ArrayLike]:
        """Serialize a Fock instance."""
        return {"array": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Fock:
        """Deserialize a Fock instance."""
        return cls(data["array"], batched=True)

    def _ipython_display_(self):
        w = widgets.fock(self)
        if w is None:
            print(repr(self))
            return
        display(w)
