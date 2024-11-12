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
This module contains the PolyExp ansatz.
"""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from typing import Any, Callable
import itertools

import numpy as np
from numpy.typing import ArrayLike

from matplotlib import colors
import matplotlib.pyplot as plt

from IPython.display import display

from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Scalar,
    Vector,
)

from mrmustard.physics.gaussian_integrals import (
    reorder_abc,
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
)

from mrmustard import math, settings, widgets
from mrmustard.math.parameters import Variable

from mrmustard.utils.argsort import argsort_gen

from .base import Ansatz
from ..batches import Batch

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz function:

        :math:`F(z) = \sum_i [\sum_k c^{(i)}_k \partial_y^k \textrm{exp}((z,y)^T A_i (z,y) / 2 + (z,y)^T b_i)|_{y=0}]`

    with ``k`` being a multi-index. The matrices :math:`A_i` and vectors :math:`b_i` are
    parameters of the exponential terms in the ansatz, and :math:`z` is a vector of variables, and  and :math:`y` is a vector linked to the polynomial coefficients.
    The dimension of ``z + y`` must be equal to the dimension of ``A`` and ``b``.

        .. code-block::

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz


        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array([[1.0,2.0,3.0]])

        >>> F = PolyExpAnsatz(A, b, c)
        >>> z = np.array([[1.0],[2.0],[3.0]])

        >>> # calculate the value of the function at the three different ``z``, since z is batched.
        >>> val = F(z)

    Args:
        A: A batch of quadratic coefficient :math:`A_i`.
        b: A batch of linear coefficients :math:`b_i`.
        c: A batch of arrays :math:`c_i`.
    """

    def __init__(
        self,
        A: ComplexMatrix | Batch[ComplexMatrix],
        b: ComplexVector | Batch[ComplexVector],
        c: ComplexTensor | Batch[ComplexTensor],
        batch_label: str | None = None,
        name: str = "",
    ):
        super().__init__()
        batch_labels = [batch_label] if batch_label else None
        self._A = (
            Batch([A], batch_labels=batch_labels)
            if A is not None and not isinstance(A, Batch)
            else A
        )
        self._b = (
            Batch([b], batch_labels=batch_labels)
            if b is not None and not isinstance(b, Batch)
            else b
        )
        self._c = (
            Batch([c], batch_labels=batch_labels)
            if c is not None and not isinstance(c, Batch)
            else c
        )
        self._simplified = False
        self.name = name

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The batch of quadratic coefficient :math:`A_i`.
        """
        self._generate_ansatz()
        return self._A

    @A.setter
    def A(self, value: ComplexMatrix | Batch[ComplexMatrix]):
        self._A = value if isinstance(value, Batch) else Batch([value])

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The batch of linear coefficients :math:`b_i`
        """
        self._generate_ansatz()
        return self._b

    @b.setter
    def b(self, value: ComplexVector | Batch[ComplexVector]):
        self._b = value if isinstance(value, Batch) else Batch([value])

    @property
    def batch_size(self):
        return self.c.shape[0]

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The batch of arrays :math:`c_i`.
        """
        self._generate_ansatz()
        return self._c

    @c.setter
    def c(self, value: ComplexTensor | Batch[ComplexTensor]):
        self._c = value if isinstance(value, Batch) else Batch([value])

    @property
    def conj(self):
        ret = PolyExpAnsatz(math.conj(self.A), math.conj(self.b), math.conj(self.c))
        ret._contract_idxs = self._contract_idxs
        return ret

    @property
    def data(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        return self.triple

    @property
    def num_vars(self):
        return self.A.shape[-1] - self.polynomial_shape[0]

    @property
    def polynomial_shape(self) -> tuple[int, tuple]:
        r"""
        This method finds the dimensionality of the polynomial, i.e. how many wires
        have polynomials attached to them and what the degree(+1) of the polynomial is
        on each of the wires.
        """
        shape_poly = self.c.core_shape
        dim_poly = len(shape_poly)
        return dim_poly, shape_poly

    @property
    def scalar(self) -> Batch[ComplexTensor]:
        if self.polynomial_shape[0] > 0:
            return self([])
        else:
            return self.c

    @property
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        return self.A, self.b, self.c

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> PolyExpAnsatz:
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> PolyExpAnsatz:
        ansatz = cls(None, None, None)
        ansatz._fn = fn
        ansatz._kwargs = kwargs
        return ansatz

    def decompose_ansatz(self) -> PolyExpAnsatz:
        r"""
        This method decomposes a PolyExp ansatz. Given an ansatz of dimension:
        A=(batch,n+m,n+m), b=(batch,n+m), c = (batch,k_1,k_2,...,k_m),
        it can be rewritten as an ansatz of dimension
        A=(batch,2n,2n), b=(batch,2n), c = (batch,l_1,l_2,...,l_n), with l_i = sum_j k_j
        This decomposition is typically favourable if m>n, and will only run if that is the case.
        The naming convention is ``n = dim_alpha``  and ``m = dim_beta`` and ``(k_1,k_2,...,k_m) = shape_beta``
        """
        dim_beta, _ = self.polynomial_shape
        dim_alpha = self.A.shape[-1] - dim_beta
        batch_size = self.batch_size
        if dim_beta > dim_alpha:
            A_decomp = []
            b_decomp = []
            c_decomp = []
            for i in range(batch_size):
                A_decomp_i, b_decomp_i, c_decomp_i = self._decompose_ansatz_single(
                    self.A[i], self.b[i], self.c[i]
                )
                A_decomp.append(A_decomp_i)
                b_decomp.append(b_decomp_i)
                c_decomp.append(c_decomp_i)

            return PolyExpAnsatz(A_decomp, b_decomp, c_decomp)
        else:
            return PolyExpAnsatz(self.A, self.b, self.c)

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

        name = "F_{" + self.name + "}(z)"
        name = f"\\arg({name})\\log|{name}|" if log_scale else name
        title = name + "e^{-|z|^2}" if with_measure else name
        title = f"\\arg({name})" if just_phase else title
        ax.set_title(f"${title}$")
        plt.show(block=False)
        return fig, ax

    def reorder(self, order: tuple[int, ...] | list[int]) -> PolyExpAnsatz:
        A, b, c = reorder_abc(self.triple, order)
        return PolyExpAnsatz(A, b, c)

    def simplify(self) -> None:
        r"""
        Simplifies the ansatz by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the ansatz has already been simplified, so it is safe to call.
        """
        if self._simplified:
            return
        indices_to_check = set(range(self.batch_size))
        removed = []
        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.allclose(self.A[i], self.A[j]) and np.allclose(self.b[i], self.b[j]):
                    self.c = math.update_add_tensor(self.c, [[i]], [self.c[j]])
                    indices_to_check.remove(j)
                    removed.append(j)
        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.A = math.gather(self.A, to_keep, axis=0)
        self.b = math.gather(self.b, to_keep, axis=0)
        self.c = math.gather(self.c, to_keep, axis=0)
        self._simplified = True

    def simplify_v2(self) -> None:
        r"""
        A different implementation of ``simplify`` that orders the batch dimension first.
        """
        if self._simplified:
            return
        self._order_batch()
        to_keep = [d0 := 0]
        mat, vec = self.A[d0], self.b[d0]
        for d in range(1, self.batch_size):
            if np.allclose(mat, self.A[d]) and np.allclose(vec, self.b[d]):
                self.c = math.update_add_tensor(self.c, [[d0]], [self.c[d]])
            else:
                to_keep.append(d)
                d0 = d
                mat, vec = self.A[d0], self.b[d0]
        self.A = math.gather(self.A, to_keep, axis=0)
        self.b = math.gather(self.b, to_keep, axis=0)
        self.c = math.gather(self.c, to_keep, axis=0)
        self._simplified = True

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"A": self.A, "b": self.b, "c": self.c}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> PolyExpAnsatz:
        A, b, c = complex_gaussian_integral_1(self.triple, idx_z, idx_zconj, measure=-1.0)
        return PolyExpAnsatz(A, b, c)

    def _call_all(self, z: Batch[Vector]) -> PolyExpAnsatz:
        r"""
        Value of this ansatz at ``z``. If ``z`` is batched a value of the function at each of the batches are returned.
        If ``Abc`` is batched it is thought of as a linear combination, and thus the results are added linearly together.
        Note that the batch dimension of ``z`` and ``Abc`` can be different.

        Conventions in code comments:
            n: is the same as dim_alpha
            m: is the same as dim_beta

        Args:
            z: point in C^n where the function is evaluated

        Returns:
            The value of the function.
        """
        dim_beta, shape_beta = self.polynomial_shape
        dim_alpha = self.A.shape[-1] - dim_beta
        batch_size = self.batch_size

        z = math.atleast_2d(z)  # shape (b_arg, n)
        if z.shape[-1] != dim_alpha or z.shape[-1] != self.num_vars:
            raise ValueError(
                "The sum of the dimension of the argument and polynomial must be equal to the dimension of A and b."
            )
        zz = math.einsum("...a,...b->...ab", z, z)[..., None, :, :]  # shape (b_arg, 1, n, n))

        A_part = math.sum(
            self.A[..., :dim_alpha, :dim_alpha] * zz, axes=[-1, -2]
        )  # sum((b_arg,1,n,n) * (b_abc,n,n), [-1,-2]) ~ (b_arg,b_abc)
        b_part = math.sum(
            self.b[..., :dim_alpha] * z[..., None, :], axes=[-1]
        )  # sum((b_arg,1,n) * (b_abc,n), [-1]) ~ (b_arg,b_abc)

        exp_sum = math.exp(1 / 2 * A_part + b_part)  # (b_arg, b_abc)
        if dim_beta == 0:
            val = math.sum(exp_sum * self.c, axes=[-1])  # (b_arg)
        else:
            b_poly = math.astensor(
                math.einsum(
                    "ijk,hk",
                    math.cast(self.A[..., dim_alpha:, :dim_alpha], "complex128"),
                    math.cast(z, "complex128"),
                )
                + self.b[..., dim_alpha:]
            )  # (b_arg, b_abc, m)
            b_poly = math.moveaxis(b_poly, 0, 1)  # (b_abc, b_arg, m)
            A_poly = self.A[..., dim_alpha:, dim_alpha:]  # (b_abc, m)
            poly = math.astensor(
                [
                    math.hermite_renormalized_batch(A_poly[i], b_poly[i], complex(1), shape_beta)
                    for i in range(batch_size)
                ]
            )  # (b_abc,b_arg,poly)
            poly = math.moveaxis(poly, 0, 1)  # (b_arg,b_abc,poly)
            val = math.sum(
                exp_sum
                * math.sum(
                    poly * self.c,
                    axes=math.arange(2, 2 + dim_beta, dtype=math.int32).tolist(),
                ),
                axes=[-1],
            )  # (b_arg)
        return val

    def _call_none(self, z: Batch[Vector]) -> PolyExpAnsatz:
        r"""
        Returns a new ansatz that corresponds to currying (partially evaluate) the current one.
        For example, if ``self`` represents the function ``F(z1,z2)``, the call ``self._call_none([np.array([1.0, None]])``
        returns ``F(1.0, z2)`` as a new ansatz with a single variable.
        Note that the batch of the triple and argument in this method is handled parwise, unlike the regular call where the batch over the triple is a superposition.

        Args:
            z: slice in C^n where the function is evaluated, while unevaluated along other axes of the space.

        Returns:
            A new ansatz.
        """
        batch_abc = self.batch_size
        batch_arg = z.shape[0]
        if batch_abc != batch_arg and batch_abc != 1 and batch_arg != 1:
            raise ValueError(
                "Batch size of the ansatz and argument must match or one of the batch sizes must be 1."
            )
        Abc = []
        max_batch = max(batch_abc, batch_arg)
        for i in range(max_batch):
            abc_index = 0 if batch_abc == 1 else i
            arg_index = 0 if batch_arg == 1 else i
            Abc.append(
                self._call_none_single(
                    self.A[abc_index], self.b[abc_index], self.c[abc_index], z[arg_index]
                )
            )
        A, b, c = zip(*Abc)
        return PolyExpAnsatz(A=A, b=b, c=c)

    def _call_none_single(self, Ai, bi, ci, zi):
        r"""
        Helper function for the call_none method. Returns the new triple.

        Args:
            Ai: The matrix of the Bargmann function
            bi: The vector of the Bargmann function
            ci: The polynomial coefficients (or scalar)
            z: point in C^n where the function is evaluated

        Returns:
            The new Abc triple.
        """
        gamma = math.astensor(zi[zi != None], dtype=math.complex128)

        z_none = np.argwhere(zi == None).reshape(-1)
        z_not_none = np.argwhere(zi != None).reshape(-1)
        beta_indices = np.arange(len(zi), Ai.shape[-1])
        new_indices = np.concatenate([z_none, beta_indices], axis=0)

        # new A
        new_A = math.gather(math.gather(Ai, new_indices, axis=0), new_indices, axis=1)

        # new b
        b_alpha = math.einsum(
            "ij,j",
            math.gather(math.gather(Ai, z_none, axis=0), z_not_none, axis=1),
            gamma,
        )
        b_beta = math.einsum(
            "ij,j",
            math.gather(math.gather(Ai, beta_indices, axis=0), z_not_none, axis=1),
            gamma,
        )
        new_b = math.gather(bi, new_indices, axis=0) + math.concat((b_alpha, b_beta), axis=-1)

        # new c
        A_part = math.einsum(
            "i,j,ij",
            gamma,
            gamma,
            math.gather(math.gather(Ai, z_not_none, axis=0), z_not_none, axis=1),
        )
        b_part = math.einsum("j,j", math.gather(bi, z_not_none, axis=0), gamma)
        exp_sum = math.exp(1 / 2 * A_part + b_part)
        new_c = ci * exp_sum
        return new_A, new_b, new_c

    def _decompose_ansatz_single(self, Ai, bi, ci):
        dim_beta, shape_beta = self.polynomial_shape
        dim_alpha = self.A.shape[-1] - dim_beta
        A_bar = math.block(
            [
                [
                    math.zeros((dim_alpha, dim_alpha), dtype=Ai.dtype),
                    Ai[:dim_alpha, dim_alpha:],
                ],
                [
                    Ai[dim_alpha:, :dim_alpha],
                    Ai[dim_alpha:, dim_alpha:],
                ],
            ]
        )
        b_bar = math.concat((math.zeros((dim_alpha), dtype=bi.dtype), bi[dim_alpha:]), axis=0)
        poly_bar = math.hermite_renormalized(
            A_bar,
            b_bar,
            complex(1),
            (math.sum(shape_beta),) * dim_alpha + shape_beta,
        )
        c_decomp = math.sum(
            poly_bar * ci,
            axes=math.arange(
                len(poly_bar.shape) - dim_beta, len(poly_bar.shape), dtype=math.int32
            ).tolist(),
        )
        A_decomp = math.block(
            [
                [
                    Ai[:dim_alpha, :dim_alpha],
                    math.eye(dim_alpha, dtype=Ai.dtype),
                ],
                [
                    math.eye((dim_alpha), dtype=Ai.dtype),
                    math.zeros((dim_alpha, dim_alpha), dtype=Ai.dtype),
                ],
            ]
        )
        b_decomp = math.concat((bi[:dim_alpha], math.zeros((dim_alpha), dtype=bi.dtype)), axis=0)
        return A_decomp, b_decomp, c_decomp

    def _generate_ansatz(self):
        r"""
        This method computes and sets the (A, b, c) given a function
        and some kwargs.
        """
        names = list(self._kwargs.keys())
        vars = list(self._kwargs.values())

        params = {}
        param_types = []
        for name, param in zip(names, vars):
            try:
                params[name] = param.value
                param_types.append(type(param))
            except AttributeError:
                params[name] = param

        if self._c is None or Variable in param_types:
            A, b, c = self._fn(**params)
            self.A = A
            self.b = b
            self.c = c

    def _ipython_display_(self):
        display(widgets.bargmann(self))

    def _order_batch(self):
        r"""
        This method orders the batch dimension by the lexicographical order of the
        flattened arrays (A, b, c). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two PolyExp ansatz.
        """
        generators = [
            itertools.chain(
                math.asnumpy(self.b[i]).flat,
                math.asnumpy(self.A[i]).flat,
                math.asnumpy(self.c[i]).flat,
            )
            for i in range(self.batch_size)
        ]
        sorted_indices = argsort_gen(generators)
        self.A = math.gather(self.A, sorted_indices, axis=0)
        self.b = math.gather(self.b, sorted_indices, axis=0)
        self.c = math.gather(self.c, sorted_indices, axis=0)

    def __add__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Adds two PolyExp ansatz together. This means concatenating them in the batch dimension.
        In the case where c is a polynomial of different shapes it will add padding zeros to make
        the shapes fit. Example: If the shape of c1 is (1,3,4,5) and the shape of c2 is (1,5,4,3) then the
        shape of the combined object will be (2,5,4,5). It also pads A and b, to account for an eventual
        different number of polynomial wires.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.")
        (n1, _) = self.A.core_shape
        (n2, _) = other.A.core_shape
        self_num_poly, _ = self.polynomial_shape
        other_num_poly, _ = other.polynomial_shape
        if self_num_poly - other_num_poly != n1 - n2:
            raise ValueError(
                f"Inconsistent polynomial shapes: the A matrices have shape {(n1,n1)} and {(n2,n2)} (difference of {n1-n2}), "
                f"but the polynomials have {self_num_poly} and {other_num_poly} variables (difference of {self_num_poly-other_num_poly})."
            )

        new_A = self.A.concat(other.A)
        new_b = self.b.concat(other.b)
        new_c = self.c.concat(other.c)

        return PolyExpAnsatz(new_A, new_b, new_c)

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Tensor product of this PolyExpAnsatz with another.
        Equivalent to :math:`F(a) * G(b)` (with different arguments, that is).
        As it distributes over addition on both self and other,
        the batch size of the result is the product of the batch
        size of this ansatz and the other one.

        Args:
            other: Another PolyExpAnsatz.

        Returns:
            The tensor product of this PolyExpAnsatz and other.
        """

        # TODO: find a general solution such that this can be done batched
        def andA(A1, A2, dim_alpha1, dim_alpha2, dim_beta1, dim_beta2):
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha1, :dim_alpha1],
                        math.zeros((dim_alpha1, dim_alpha2), dtype=math.complex128),
                        A1[:dim_alpha1, dim_alpha1:],
                        math.zeros((dim_alpha1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        math.zeros((dim_alpha2, dim_alpha1), dtype=math.complex128),
                        A2[:dim_alpha2:, :dim_alpha2],
                        math.zeros((dim_alpha2, dim_beta1), dtype=math.complex128),
                        A2[:dim_alpha2, dim_alpha2:],
                    ],
                    [
                        A1[dim_alpha1:, :dim_alpha1],
                        math.zeros((dim_beta1, dim_alpha2), dtype=math.complex128),
                        A1[dim_alpha1:, dim_alpha1:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        math.zeros((dim_beta2, dim_alpha1), dtype=math.complex128),
                        A2[dim_alpha2:, :dim_alpha2],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha2:, dim_alpha2:],
                    ],
                ]
            )
            return A3

        def andb(b1, b2, dim_alpha1, dim_alpha2):
            b3 = math.reshape(
                math.block(
                    [
                        [
                            b1[:dim_alpha1],
                            b2[:dim_alpha2],
                            b1[dim_alpha1:],
                            b2[dim_alpha2:],
                        ]
                    ]
                ),
                -1,
            )
            return b3

        def andc(c1, c2):
            c3 = math.outer(c1, c2)
            return c3

        dim_beta1, _ = self.polynomial_shape
        dim_beta2, _ = other.polynomial_shape

        dim_alpha1 = self.A.shape[-1] - dim_beta1
        dim_alpha2 = other.A.shape[-1] - dim_beta2

        As = [
            andA(
                math.cast(A1, "complex128"),
                math.cast(A2, "complex128"),
                dim_alpha1,
                dim_alpha2,
                dim_beta1,
                dim_beta2,
            )
            for A1, A2 in itertools.product(self.A._items, other.A._items)
        ]
        bs = [
            andb(b1, b2, dim_alpha1, dim_alpha2)
            for b1, b2 in itertools.product(self.b._items, other.b._items)
        ]
        cs = [andc(c1, c2) for c1, c2 in itertools.product(self.c._items, other.c._items)]

        A_batch = Batch(As, *self.A._new_batch(other.A, "prod"))
        b_batch = Batch(bs, *self.b._new_batch(other.b, "prod"))
        c_batch = Batch(cs, *self.c._new_batch(other.c, "prod"))

        return PolyExpAnsatz(A_batch, b_batch, c_batch)

    def __call__(self, z: Batch[Vector]) -> Scalar | PolyExpAnsatz:
        r"""
        Returns either the value of the ansatz or a new ansatz depending on the argument.
        If the argument contains None, returns a new ansatz.
        If the argument only contains numbers, returns the value of the ansatz at that argument.
        Note that the batch dimensions are handled differently in the two cases. See subfunctions for further information.

        Args:
            z: point in C^n where the function is evaluated.

        Returns:
            The value of the function if ``z`` has no ``None``, else it returns a new ansatz.
        """
        if (np.array(z) == None).any():
            return self._call_none(z)
        else:
            return self._call_all(z)

    def __eq__(self, other: PolyExpAnsatz) -> bool:
        if not isinstance(other, PolyExpAnsatz):
            return False
        self.simplify()
        other.simplify()
        return self.A == other.A and self.b == other.b and self.c == other.c

    def __getitem__(self, idx: int | tuple[int, ...]) -> PolyExpAnsatz:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz of dimension {self.num_vars}."
                )
        ret = PolyExpAnsatz(self.A, self.b, self.c)
        ret._contract_idxs = idx
        return ret

    def __matmul__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Implements the inner product between PolyExpAnsatz.

        ..code-block::

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz
        >>> from mrmustard.physics.triples import displacement_gate_Abc, vacuum_state_Abc
        >>> rep1 = PolyExpAnsatz(*vacuum_state_Abc(1))
        >>> rep2 = PolyExpAnsatz(*displacement_gate_Abc(1))
        >>> rep3 = rep1[0] @ rep2[1]
        >>> assert np.allclose(rep3.A, [[0,],])
        >>> assert np.allclose(rep3.b, [1,])

         Args:
             other: Another PolyExpAnsatz .

         Returns:
            Bargmann: the resulting PolyExpAnsatz.

        """
        if not isinstance(other, PolyExpAnsatz):
            raise NotImplementedError("Only matmul PolyExpAnsatz with PolyExpAnsatz")

        idx_s = self._contract_idxs
        idx_o = other._contract_idxs

        if settings.UNSAFE_ZIP_BATCH:
            if self.batch_size != other.batch_size:
                raise ValueError(
                    f"Batch size of the two representations must match since the settings.UNSAFE_ZIP_BATCH is {settings.UNSAFE_ZIP_BATCH}."
                )
            A, b, c = complex_gaussian_integral_2(
                self.triple, other.triple, idx_s, idx_o, mode="zip"
            )
        else:
            A, b, c = complex_gaussian_integral_2(
                self.triple, other.triple, idx_s, idx_o, mode="kron"
            )

        return PolyExpAnsatz(A, b, c)

    def __mul__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        def mul_A(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def mul_b(b1, b2, dim_alpha):
            b3 = math.reshape(
                math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]),
                -1,
            )
            return b3

        def mul_c(c1, c2):
            c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
            return c3

        if isinstance(other, PolyExpAnsatz):
            dim_beta1, _ = self.polynomial_shape
            dim_beta2, _ = other.polynomial_shape

            dim_alpha1 = self.A.shape[-1] - dim_beta1
            dim_alpha2 = other.A.shape[-1] - dim_beta2
            if dim_alpha1 != dim_alpha2:
                raise TypeError("The dimensionality of the two ansatze must be the same.")
            dim_alpha = dim_alpha1

            new_a = [
                mul_A(
                    math.cast(A1, "complex128"),
                    math.cast(A2, "complex128"),
                    dim_alpha,
                    dim_beta1,
                    dim_beta2,
                )
                for A1, A2 in itertools.product(self.A, other.A)
            ]
            new_b = [mul_b(b1, b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
            new_c = [mul_c(c1, c2) for c1, c2 in itertools.product(self.c, other.c)]

            return PolyExpAnsatz(A=new_a, b=new_b, c=new_c)
        else:
            try:
                return PolyExpAnsatz(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __neg__(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, -self.c)

    def __truediv__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        def div_A(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def div_b(b1, b2, dim_alpha):
            b3 = math.reshape(
                math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]),
                -1,
            )
            return b3

        def div_c(c1, c2):
            c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
            return c3

        if isinstance(other, PolyExpAnsatz):
            dim_beta1, _ = self.polynomial_shape
            dim_beta2, _ = other.polynomial_shape
            if dim_beta1 == 0 and dim_beta2 == 0:
                dim_alpha1 = self.A.shape[-1] - dim_beta1
                dim_alpha2 = other.A.shape[-1] - dim_beta2
                if dim_alpha1 != dim_alpha2:
                    raise TypeError("The dimensionality of the two ansatze must be the same.")
                dim_alpha = dim_alpha1

                new_a = [
                    div_A(
                        math.cast(A1, "complex128"),
                        -math.cast(A2, "complex128"),
                        dim_alpha,
                        dim_beta1,
                        dim_beta2,
                    )
                    for A1, A2 in itertools.product(self.A, other.A)
                ]
                new_b = [div_b(b1, -b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
                new_c = [div_c(c1, 1 / c2) for c1, c2 in itertools.product(self.c, other.c)]

                return PolyExpAnsatz(A=new_a, b=new_b, c=new_c)
            else:
                raise NotImplementedError("Only implemented if both c are scalars")
        else:
            try:
                return PolyExpAnsatz(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e
