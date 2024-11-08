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
    Batch,
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

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz function:

        :math:`F_j(z) = \sum_i [\sum_k c^{(i)}_{jk} \partial_y^k \textrm{exp}((z,y)^T A^{(i)} (z,y) / 2 + (z,y)^T b^{(i)})|_{y=0}]`

    with ``j`` and ``k`` are multi-indices. The ``j`` index represents the output shape of the array of polynomials of derivatives,
    while the ``k`` index is contracted with the vectors of derivatives to form the polynomial of derivatives.
    The matrices :math:`A_i` and vectors :math:`b_i` are the parameters of the exponential terms in the ansatz,
    with :math:`z` and :math:`y` vectors of continuous complex variables.
    :math:`c^{(i)}_{jk}` are the coefficients of the polynomial of derivatives and :math:`y` is the vector of continuous
    complex variables that are derived by the polynomial of derivatives.

        .. code-block::

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz


        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array([[1.0, 2.0, 3.0]])

        >>> F = PolyExpAnsatz(A, b, c, num_derived_vars=1)
        >>> z = np.array([[1.0],[2.0],[3.0]])

        >>> # calculate the value of the function at the three different ``z``, since z is batched.
        >>> val = F(z)

    Args:
        A: A batch of quadratic coefficient :math:`A_i`.
        b: A batch of linear coefficients :math:`b_i`.
        c: A batch of arrays :math:`c_i`.
        num_derived_vars: The number of continuous variables :math:`y` that are derived by the polynomial of derivatives.
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix] | None,
        b: Batch[ComplexVector] | None,
        c: Batch[ComplexTensor] | None = math.ones([], dtype=math.complex128),
        num_derived_vars: int = 0,
        name: str = "",
    ):
        super().__init__()
        self._A = A
        self._b = b
        self._c = c
        self.num_derived_vars = num_derived_vars
        self._simplified = False
        self.name = name
        self._fn = None
        self._fn_kwargs = {}

    def _generate_ansatz(self):
        r"""
        This method computes and sets the (A, b, c) triple given a function and some kwargs.
        """
        if (
            self._A is None
            or self._b is None
            or self._c is None
            or Variable in {type(param) for param in self._fn_kwargs.values()}
        ):
            params = {}
            for name, param in self._fn_kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param

            data = self._fn(**params)
            if len(data) == 4:
                A, b, c, num_derived_vars = data
            else:
                A, b, c = data
                c = math.astensor(c)
                num_derived_vars = 0
            self._A = A
            self._b = b
            self._c = c
            self.num_derived_vars = num_derived_vars

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The batch of quadratic coefficient :math:`A_i`.
        """
        self._generate_ansatz()
        return math.atleast_3d(self._A)

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The batch of linear coefficients :math:`b_i`
        """
        self._generate_ansatz()
        return math.atleast_2d(self._b)

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The batch of polynomial coefficients :math:`c_i`.
        """
        self._generate_ansatz()
        return math.atleast_1d(self._c)

    @property
    def batch_size(self) -> int:
        return self.c.shape[0]

    @property
    def conj(self):
        ret = PolyExpAnsatz(
            math.conj(self.A), math.conj(self.b), math.conj(self.c), self.num_derived_vars
        )
        ret._contract_idxs = self._contract_idxs
        return ret

    @property
    def data(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor], int]:
        """Returns the triple and the number of derived variables necessary to reinstantiate the ansatz."""
        return self.triple, self.num_derived_vars

    @property
    def num_CV_vars(self) -> int:
        r"""
        The number of continuous variables that remain after the polynomial of derivatives is applied.
        This is the number of continuous variables of the Ansatz function itself.
        """
        return self.A.shape[-1] - self.num_derived_vars

    @property
    def num_DV_vars(self) -> int:
        r"""
        The number of discrete variables after the polynomial of derivatives is applied.
        This is the number of axes of the array of values that we get when we evaluate the ansatz at a point.
        """
        return len(self.shape_DV_vars)

    @property
    def num_vars(self):
        r"""
        The total number of variables of this ansatz before the polynomial of derivatives is applied.
        """
        return self.num_CV_vars + self.num_derived_vars

    @property
    def shape_DV_vars(self) -> tuple[int, ...]:
        r"""
        The shape of the discrete variables.
        This is the shape of the array of values that we get when we evaluate the ansatz at a point.
        """
        return self.c.shape[self.num_derived_vars + 1 :]

    @property
    def shape_derived_vars(self) -> tuple[int, ...]:
        r"""
        The shape of the derived variables (i.e. the polynomial of derivatives).
        """
        return self.c.shape[1 : self.num_derived_vars + 1]

    @property
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        """Returns the triple of parameters of the exponential part of the ansatz."""
        return self.A, self.b, self.c

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> PolyExpAnsatz:
        """Creates an ansatz from a dictionary. For deserialization purposes."""
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> PolyExpAnsatz:
        """Creates an ansatz given a function and its kwargs. This ansatz is lazily instantiated, i.e.
        the function is not called until the A,b,c attributes are accessed (even internally)."""
        ansatz = cls(None, None, None, None)
        ansatz._fn = fn
        ansatz._fn_kwargs = kwargs
        return ansatz

    def decompose_ansatz(self) -> PolyExpAnsatz:
        r"""
        This method decomposes a PolyExp ansatz to make it more efficient to evaluate.
        An ansatz with ``n`` CV variables and ``m`` derived variables has parameters with the following shapes:
        ``A=(batch;n+m,n+m)``, ``b=(batch;n+m)``, ``c = (batch;k_1,k_2,...,k_m;j_1,...,j_d)``, where ``d`` is the number of
        discrete variables, i.e. axes of the array of values that we get when we evaluate the ansatz at a point.
        This can be rewritten as an ansatz of dimension ``A=(batch;2n,2n)``, ``b=(batch;2n)``,
        ``c = (batch;l_1,l_2,...,l_n;j_1,...,j_d)``, with ``l_i = sum_j k_j``.
        This means that the number of continuous variables remains ``n``, the number of derived variables decreases from
        ``m`` to ``n``, and the number of discrete variables remains ``d``. The price we pay is that the order of the
        derivatives is larger (the order of each derivative is the sum of all the orders of the initial derivatives).
        This decomposition is typically favourable if ``m > n`` and the sum of the elements in ``c.shape[1:]`` is not too large.
        This method will actually decompose the ansatz only if ``m > n`` and return the original ansatz otherwise.
        """
        if self.num_derived_vars < self.num_CV_vars:
            return self
        A_dec = []
        b_dec = []
        c_dec = []
        for Ai, bi, ci in zip(self.A, self.b, self.c):
            A_dec_i, b_dec_i, c_dec_i = self._decompose_single(Ai, bi, ci)
            A_dec.append(A_dec_i)
            b_dec.append(b_dec_i)
            c_dec.append(c_dec_i)
        return PolyExpAnsatz(A_dec, b_dec, c_dec, self.num_CV_vars)

    def _decompose_single(self, Ai, bi, ci):
        r"""
        Decomposes a single batch element of the ansatz.
        """
        n = self.num_CV_vars
        m = self.num_derived_vars
        A_core = math.block(
            [[math.zeros((n, n), dtype=Ai.dtype), Ai[:n, m:]], [Ai[m:, :n], Ai[m:, m:]]]
        )
        b_core = math.concat((math.zeros((n,), dtype=bi.dtype), bi[m:]), axis=-1)
        poly_shape = (math.sum(self.shape_derived_vars),) * n + self.shape_derived_vars
        poly_core = math.hermite_renormalized(A_core, b_core, complex(1), poly_shape)
        c_prime = math.sum(poly_core * ci, axes=[i for i in range(n, n + m)])
        block = Ai[:n, :n]
        A_decomp = math.block(
            [[block, math.eye_like(block)], [math.eye_like(block), math.zeros_like(block)]]
        )
        b_decomp = math.concat((bi[:n], math.zeros((n,), dtype=bi.dtype)), axis=-1)
        return A_decomp, b_decomp, c_prime

    def plot(
        self,
        just_phase: bool = False,
        with_measure: bool = False,
        log_scale: bool = False,
        xlim=(-2 * np.pi, 2 * np.pi),
        ylim=(-2 * np.pi, 2 * np.pi),
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:  # pragma: no cover
        r"""
        Plots the ansatz as a Bargmann function :math:`F(z)` on the complex plane.
        Phase is represented by color, magnitude by brightness.
        The function can be multiplied by :math:`exp(-|z|^2)` to represent the Bargmann
        function times the measure function (for integration).

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
        if len(order) != self.num_vars:
            raise ValueError(
                f"The order must have the same length as the number of CV+DV variables, {self.num_vars}."
            )
        A, b, c = reorder_abc(self.triple, order)
        return PolyExpAnsatz(A, b, c, self.num_derived_vars)

    def simplify(self, sort_batch: bool = False) -> None:
        r"""
        Simplifies an ansatz by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the ansatz has already been simplified, so it is always safe to call.

        Args:
            sort_batch: If True, orders the batch dimension first before simplifying, which can
                be more efficient for large batch sizes. If False, uses the original implementation
                that checks pairs of terms.
        """
        if self._simplified:
            return

        to_keep = (
            self._find_unique_terms_sorted() if sort_batch else self._find_unique_terms_pairwise()
        )
        self.A = math.gather(self.A, to_keep, axis=0)
        self.b = math.gather(self.b, to_keep, axis=0)
        self.c = math.gather(self.c, to_keep, axis=0)
        self._simplified = True

    def _find_unique_terms_sorted(self) -> list[int]:
        """Find unique terms by first sorting the batch dimension.
        Needed in ``simplify``.

        Returns:
            List of indices to keep after simplification.
        """
        self._order_batch()
        to_keep = [d0 := 0]
        mat, vec = self.A[d0], self.b[d0]

        for d in range(1, self.batch_size):
            if not (np.array_equal(mat, self.A[d]) and np.array_equal(vec, self.b[d])):
                to_keep.append(d)
                d0 = d
                mat, vec = self.A[d0], self.b[d0]
            else:
                self.c = math.update_add_tensor(self.c, [[d0]], [self.c[d]])

        return to_keep

    def _find_unique_terms_pairwise(self) -> list[int]:
        """Find unique terms by checking all pairs.
        Needed in ``simplify``.

        Returns:
            List of indices to keep after simplification.
        """
        indices_to_check = set(range(self.batch_size))
        removed = []

        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.array_equal(self.A[i], self.A[j]) and np.array_equal(self.b[i], self.b[j]):
                    self.c = math.update_add_tensor(self.c, [[i]], [self.c[j]])
                    indices_to_check.remove(j)
                    removed.append(j)

        return [i for i in range(self.batch_size) if i not in removed]

    def to_dict(self) -> dict[str, ArrayLike]:
        """Returns a dictionary representation of the ansatz. For serialization purposes."""
        return {"A": self.A, "b": self.b, "c": self.c}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> PolyExpAnsatz:
        A, b, c = complex_gaussian_integral_1(self.triple, idx_z, idx_zconj, measure=-1.0)
        return PolyExpAnsatz(A, b, c, self.num_derived_vars)

    def _eval_at_point(self: PolyExpAnsatz, z: Batch[Vector]) -> Batch[ComplexTensor]:
        """Evaluates the ansatz at a point ``z``.
        The points can have an arbitrary number of batch dimensions, which are preserved in the output.

        Args:
            z: Point(s) in C^n where the function is evaluated.

        Returns:
            The value of the function at the point(s).
        """
        z = math.atleast_2d(z)
        z_batch_shape = z.shape[:-1]
        z = math.reshape(z, (-1, z.shape[-1]))  # shape (k, num_CV_vars)
        if z.shape[-1] != self.num_CV_vars:
            raise ValueError(
                f"The last dimension of `z` must equal {self.num_CV_vars}, got {z.shape[-1]}."
            )

        A_part = math.einsum(
            "ka,kb,iab->ik", z, z, self.A[..., : self.num_CV_vars, : self.num_CV_vars]
        )
        b_part = math.einsum("ka,ia->ik", z, self.b[..., : self.num_CV_vars])
        exp_sum = math.exp(1 / 2 * A_part + b_part)  # shape (batch_size, k)

        if self.num_derived_vars == 0:
            return math.einsum("nk,n...->k...", exp_sum, self.c)

        poly = self._compute_polynomial_part(z)  # shape (batch_size, k, *derived_shape)
        return math.reshape(
            self._combine_exp_and_poly(exp_sum, poly), z_batch_shape + self.shape_DV_vars
        )

    def _compute_polynomial_part(self, z: Batch[Vector]) -> Batch[Scalar]:
        """Computes the polynomial part of the ansatz evaluation.
        Needed in ``_eval_at_point``.

        Args:
            z: Input vector of shape ``(k, num_CV_vars)``

        Returns:
            Polynomial term of shape ``
        """
        b_poly = (
            math.einsum("iab,ka->ikb", self.A[..., : self.num_CV_vars, self.num_CV_vars :], z)
            + self.b[..., None, self.num_CV_vars :]
        )
        A_poly = self.A[
            ..., self.num_CV_vars :, self.num_CV_vars :
        ]  # shape (batch_size,derived_vars,derived_vars)
        result = []
        for i in range(self.batch_size):
            result.append(
                math.hermite_renormalized_batch(
                    A_poly[i], b_poly[i], complex(1), self.shape_derived_vars
                )
            )
        return math.astensor(result)

    def _combine_exp_and_poly(
        self, exp_sum: Batch[ComplexTensor], poly: Batch[ComplexMatrix]
    ) -> Batch[ComplexTensor]:
        """Combines exponential and polynomial parts using einsum.
        Needed in ``_eval_at_point``.

        Args:
            exp_sum: Exponential sum term
            poly: Polynomial term

        Returns:
            Final evaluation
        """
        c = math.reshape(
            self.c, (self.batch_size, np.prod(self.shape_derived_vars), np.prod(self.shape_DV_vars))
        )
        k = exp_sum.shape[1]  # batch_size of z
        poly = math.reshape(poly, (self.batch_size, k, np.prod(self.shape_derived_vars)))
        result = np.einsum("nk,ndD,nkd->nD", exp_sum, c, poly, optimize=True)
        return math.reshape(result, (self.batch_size, *self.shape_DV_vars))

    def _partial_eval(self, z: Batch[Vector], indices: tuple[int, ...]) -> PolyExpAnsatz:
        r"""
        Returns a new ansatz that corresponds to currying (partially evaluate) the current one.
        For example, if ``self`` represents the function ``F(z1,z2)``, the call ``self._partial_eval([np.array([1.0]), None)``
        returns ``F(1.0, z2)`` as a new ansatz with a single variable.

        Args:
            z: slice in C^n where the function is evaluated, while unevaluated along other axes of the space.

        Returns:
            A new ansatz.
        """
        if len(indices) == self.num_CV_vars:
            return self._eval_at_point(z)
        batch_z = z.shape[0]
        if self.batch_size != batch_z and self.batch_size != 1 and batch_z != 1:
            raise ValueError(
                "Batch size of the ansatz and argument must match or one of the batch sizes must be 1."
            )
        Abc = []
        max_batch = max(self.batch_size, batch_z)
        for i in range(max_batch):
            abc_index = 0 if self.batch_size == 1 else i
            arg_index = 0 if batch_z == 1 else i
            Abc.append(
                self._partial_eval_single(
                    self.A[abc_index], self.b[abc_index], self.c[abc_index], z[arg_index]
                )
            )
        A, b, c = zip(*Abc)
        return PolyExpAnsatz(A=A, b=b, c=c)

    def _partial_eval_single(self, Ai, bi, ci, zi):
        r"""
        Helper function for the _partial_eval method. Returns the new triple.

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

    def _equal_no_array(self, other: PolyExpAnsatz) -> bool:
        self.simplify()
        other.simplify()
        return np.allclose(self.b, other.b, atol=1e-10) and np.allclose(self.A, other.A, atol=1e-10)

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
        self._A = math.gather(self._A, sorted_indices, axis=0)
        self._b = math.gather(self._b, sorted_indices, axis=0)
        self._c = math.gather(self._c, sorted_indices, axis=0)

    def __add__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Adds two PolyExp ansatz together. This means concatenating them in the batch dimension.
        In the case where poly on self and other are of different shapes it will add padding zeros to make
        the shapes fit. Example: If the shape of poly1 is (1,3,4,5) and the shape of poly2 is (1,5,4,3) then the
        shape of the combined object will be (2,5,4,5). It also pads A and b, to account for an eventual
        different number of polynomial wires.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.")
        (_, n1, _) = self.A.shape
        (_, n2, _) = other.A.shape
        self_num_poly = self.num_DV_vars
        other_num_poly = other.num_DV_vars
        if self_num_poly - other_num_poly != n1 - n2:
            raise ValueError(
                f"Inconsistent polynomial shapes: the A matrices have shape {(n1,n1)} and {(n2,n2)} (difference of {n1-n2}), "
                f"but the polynomials have {self_num_poly} and {other_num_poly} variables (difference of {self_num_poly-other_num_poly})."
            )

        def pad_and_expand(mat, vec, array, target_size):
            pad_size = target_size - mat.shape[-1]
            padded_mat = math.pad(mat, ((0, 0), (0, pad_size), (0, pad_size)))
            padded_vec = math.pad(vec, ((0, 0), (0, pad_size)))
            padding_array = math.ones((1,) * pad_size, dtype=array.dtype)
            expanded_array = math.outer(array, padding_array)
            return padded_mat, padded_vec, expanded_array

        def combine_arrays(array1, array2):
            shape1 = array1.shape[1:]
            shape2 = array2.shape[1:]
            max_shape = tuple(map(max, zip(shape1, shape2)))
            pad_widths1 = [(0, 0)] + [(0, t - s) for s, t in zip(shape1, max_shape)]
            pad_widths2 = [(0, 0)] + [(0, t - s) for s, t in zip(shape2, max_shape)]
            padded_array1 = math.pad(array1, pad_widths1, "constant")
            padded_array2 = math.pad(array2, pad_widths2, "constant")
            return math.concat([padded_array1, padded_array2], axis=0)

        if n1 <= n2:
            mat1, vec1, array1 = pad_and_expand(self.A, self.b, self.c, n2)
            combined_matrices = math.concat([mat1, other.A], axis=0)
            combined_vectors = math.concat([vec1, other.b], axis=0)
            combined_arrays = combine_arrays(array1, other.c)
        else:
            mat2, vec2, array2 = pad_and_expand(other.A, other.b, other.c, n1)
            combined_matrices = math.concat([self.A, mat2], axis=0)
            combined_vectors = math.concat([self.b, vec2], axis=0)
            combined_arrays = combine_arrays(self.c, array2)
        # note output is not simplified
        return PolyExpAnsatz(combined_matrices, combined_vectors, combined_arrays)

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Tensor product of this PolyExpAnsatz with another.
        Equivalent to :math:`F(a) * G(b)` (with different arguments).
        As it distributes over addition on both self and other,
        the batch size of the result is the product of the batch
        size of this ansatz and the other one.

        Args:
            other: Another PolyExpAnsatz.

        Returns:
            The tensor product of this PolyExpAnsatz and other.
        """

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
            c3 = math.reshape(
                math.outer(c1, c2), (c1.shape + c2.shape)
            )  # TODO: reorder so that DV vars are at the end
            return c3

        dim_beta1 = self.num_DV_vars
        dim_beta2 = other.num_DV_vars

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
            for A1, A2 in itertools.product(self.A, other.A)
        ]
        bs = [andb(b1, b2, dim_alpha1, dim_alpha2) for b1, b2 in itertools.product(self.b, other.b)]
        cs = [andc(c1, c2) for c1, c2 in itertools.product(self.c, other.c)]
        return PolyExpAnsatz(As, bs, cs)

    def __call__(self, *z: Vector | None, mode: str = "zip") -> Scalar | PolyExpAnsatz:
        r"""
        Returns either the value of the ansatz or a new ansatz depending on the arguments.
        If an argument is None, the corresponding variable is not evaluated, and the method
        returns a new ansatz with the remaining variables unevaluated.
        For example, if the ansatz is a function of 3 variables and we want to evaluate it at a point in C^2,
        we would get a new ansatz with one variable unevaluated: F(z1, z2, None) or F(z1, None, z3), or F(None, z2, z3).
        The ``mode`` argument can be used to specify how the vectors of arguments are broadcast together.
        The default is "zip", which is to broadcast the vectors pairwise.
        The alternative is "kron", which is to broadcast the vectors Kronecker-style.
        For example, ``F(z1, z2, mode="zip")`` returns the array of values ``[F(z1[0], z2[0]), F(z1[1], z2[1]), ...]``.
        ``F(z1, z2, mode="kron")`` returns the Kronecker product of the vectors, i.e.
        ``[[F(z1[0], z2[0]), F(z1[0], z2[1]), ...], [F(z1[1], z2[0]), F(z1[1], z2[1]), ...], ...]``.

        Args:
            z: points in C where the function is (partially) evaluated or None if the variable is not evaluated.
            mode: "zip" or "kron"

        Returns:
            The value of the function or a new ansatz.
        """
        only_z = [z_i for z_i in z if z_i is not None]
        indices = [i for i, z_i in enumerate(z) if z_i is not None]
        if mode == "zip":
            z_batch = only_z[0].shape[0]
            if any(z_i.shape[0] != z_batch for z_i in only_z):
                raise ValueError(
                    f"In mode 'zip' the batch size of the z vectors must match. Got {[z_i.shape[0] for z_i in only_z]}."
                )
            return self._partial_eval(math.concat(only_z, axis=-1), indices)
        elif mode == "kron":
            return self._partial_eval(math.kron(*only_z), indices)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'zip' or 'kron'.")

    def __eq__(self, other: PolyExpAnsatz) -> bool:
        return self._equal_no_array(other) and np.allclose(self.c, other.c, atol=1e-10)

    def __getitem__(self, idx: int | tuple[int, ...]) -> PolyExpAnsatz:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz of dimension {self.num_vars}."
                )
        ret = PolyExpAnsatz(self.A, self.b, self.c, self.num_CV_vars)
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
            dim_beta1 = self.num_DV_vars
            dim_beta2 = other.num_DV_vars

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
            dim_beta1 = self.num_DV_vars
            dim_beta2 = other.num_DV_vars
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
