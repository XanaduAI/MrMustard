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

# pylint: disable=too-many-instance-attributes,too-many-positional-arguments

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence
import itertools

import numpy as np
from numpy.typing import ArrayLike

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
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
    join_Abc,
)

from mrmustard.math.lattice.strategies import vanilla_fully_batched

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable

from mrmustard.utils.argsort import argsort_gen

from .base import Ansatz

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""
    This class represents the ansatz function:

        :math:`F^{(i)}(z) = \sum_k c^{(i)}_{k} \partial_y^k \textrm{exp}(\frac{1}{2}(z,y)^T A^{(i)} (z,y) + (z,y)^T b^{(i)})|_{y=0}`

    with ``k`` a multi-index. The ``i`` index is a batch index that can be used for linear
    superposition or batching purposes. The ``c^{(i)}_k`` tensor is contracted with the vectors of
    derivatives :math:`\partial_y^k` to form the polynomials of derivatives.
    The matrices :math:`A^{(i)}` and vectors :math:`b^{(i)}` are the parameters of the exponential
    terms in the ansatz, with :math:`z` and :math:`y` vectors of continuous complex variables.
    They have shape ``(L, n+m, n+m)`` and ``(L, n+m)``, respectively for ``n`` continuous variables
    and ``m`` derived variables (i.e. :math:`z\in\mathbb{C}^{n}` and :math:`y\in\mathbb{C}^{m}`).
    The tensors :math:`c^{(i)}_{k}` contain the coefficients of the polynomial of derivatives and
    have shape ``(L, *derived)``, where ``*derived`` is the shape of the derived variables, which
    implies ``len(c.shape[1:]) = m``.

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
        A: A batch of quadratic coefficient :math:`A^{(i)}`.
        b: A batch of linear coefficients :math:`b^{(i)}`.
        c: A batch of arrays :math:`c^{(i)}`.
        num_derived_vars: The number of variables :math:`y` that are derived by the polynomial of derivatives.
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix] | None,
        b: Batch[ComplexVector] | None,
        c: Batch[ComplexTensor] | None = np.ones([], dtype=np.complex128),
        num_derived_vars: int = 0,  # i.e. size of y
        name: str = "",
    ):
        super().__init__()
        self._A = math.atleast_3d(math.astensor(A)) if A is not None else None
        self._b = math.atleast_2d(math.astensor(b)) if b is not None else None
        self._c = math.atleast_nd(math.astensor(c), num_derived_vars + 1) if c is not None else None
        self.num_derived_vars = num_derived_vars
        self.name = name
        self._simplified = False
        self._fn = None
        self._fn_kwargs = {}
        self._batch_size = self._A.shape[0] if A is not None else None

    def _should_regenerate(self):
        return (
            self._A is None
            or self._b is None
            or self._c is None
            or Variable in {type(param) for param in self._fn_kwargs.values()}
        )

    def _generate_ansatz(self):
        r"""
        This method computes and sets the (A, b, c) triple given a function and its kwargs.
        """
        if self._should_regenerate():
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
            self._A = math.atleast_3d(A)
            self._b = math.atleast_2d(b)
            self._c = math.atleast_nd(c, num_derived_vars + 1)
            self.num_derived_vars = num_derived_vars
            self._batch_size = self._A.shape[0]

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The batch of quadratic coefficient :math:`A^{(i)}`.
        """
        self._generate_ansatz()
        return self._A

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The batch of linear coefficients :math:`b^{(i)}`.
        """
        self._generate_ansatz()
        return self._b

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The batch of polynomial coefficients :math:`c^{(i)}_{jk}`.
        """
        self._generate_ansatz()
        return self._c

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
        """
        return self.c

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            return 1
        return self._batch_size

    @property
    def conj(self):
        return PolyExpAnsatz(
            math.conj(self.A), math.conj(self.b), math.conj(self.c), self.num_derived_vars
        )

    @property
    def data(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor], int]:
        r"""Returns the triple and the number of derived variables necessary to reinstantiate the ansatz."""
        return self.triple, self.num_derived_vars

    @property
    def num_CV_vars(self) -> int:
        r"""
        The number of continuous variables that remain after the polynomial of derivatives is applied.
        This is the number of continuous variables of the Ansatz function itself, i.e. the size of ``z``
        in :math:`F_j^{(i)}(z)`.
        """
        return self.A.shape[-1] - self.num_derived_vars

    @property
    def num_vars(self):
        r"""
        The total number of continuous variables of this ansatz before the polynomial of derivatives is applied.
        """
        return self.num_CV_vars + self.num_derived_vars

    @property
    def shape_derived_vars(self) -> tuple[int, ...]:
        r"""
        The shape of the derived variables (i.e. the polynomial of derivatives).
        Encoded in ``c`` as the axes between the batch size (first axis) and the discrete variables.
        """
        return self.c.shape[-self.num_derived_vars :] if self.num_derived_vars > 0 else ()

    @property
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""Returns the triple of parameters of the exponential part of the ansatz."""
        return self.A, self.b, self.c

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> PolyExpAnsatz:
        r"""Creates an ansatz from a dictionary. For deserialization purposes."""
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> PolyExpAnsatz:
        r"""Creates an ansatz given a function and its kwargs. This ansatz is lazily instantiated, i.e.
        the function is not called until the A,b,c attributes are accessed (even internally)."""
        ansatz = cls(None, None, None, None)
        ansatz._fn = fn
        ansatz._fn_kwargs = kwargs
        return ansatz

    def contract(
        self,
        other: PolyExpAnsatz,
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
        mode: Literal["zip", "kron"] = "kron",
    ) -> PolyExpAnsatz:
        r"""
        Contracts two ansatze across the specified indices.
        Args:
            other: The ansatz to contract with.
            idx1: The indices of the first ansatz to contract.
            idx2: The indices of the second ansatz to contract.
            mode: The mode of contraction. "zip" contracts the batch dimensions, "kron" contracts the CV dimensions.

        Returns:
            The contracted ansatz.
        """
        idx1 = (idx1,) if isinstance(idx1, int) else idx1
        idx2 = (idx2,) if isinstance(idx2, int) else idx2
        for i, j in zip(idx1, idx2):
            if i and i >= self.num_CV_vars:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz with {self.num_CV_vars} CV variables."
                )
            if j and j >= other.num_CV_vars:
                raise IndexError(
                    f"Index {j} out of bounds for ansatz with {other.num_CV_vars} CV variables."
                )

        if mode == "zip":
            if self.batch_size != other.batch_size:
                raise ValueError(
                    f"For mode='zip' the batch size of the two representations must match, got {self.batch_size} and {other.batch_size}."
                )
        A, b, c = complex_gaussian_integral_2(self.triple, other.triple, idx1, idx2, mode=mode)
        return PolyExpAnsatz(A, b, c, self.num_derived_vars + other.num_derived_vars)

    def decompose_ansatz(self) -> PolyExpAnsatz:
        r"""
        This method decomposes a PolyExp ansatz to make it more efficient to evaluate.
        An ansatz with ``n`` CV variables and ``m`` derived variables has parameters with the following shapes:
        ``A=(batch;n+m,n+m)``, ``b=(batch;n+m)``, ``c = (batch;k_1,k_2,...,k_m;j_1,...,j_d)``, where ``d`` is the number of
        discrete variables, i.e. axes of the array of values that we get when we evaluate the ansatz at a point.
        This can be rewritten as an ansatz of dimension ``A=(*batch;2n,2n)``, ``b=(*batch;2n)``,
        ``c = (*batch;l_1,l_2,...,l_n;j_1,...,j_d)``, with ``l_i = sum_j k_j``.
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
            [[math.zeros((n, n), dtype=Ai.dtype), Ai[:n, n:]], [Ai[n:, :n], Ai[n:, n:]]]
        )
        b_core = math.concat((math.zeros((n,), dtype=bi.dtype), bi[n:]), axis=-1)
        poly_shape = (math.sum(self.shape_derived_vars),) * n + self.shape_derived_vars
        poly_core = math.hermite_renormalized(A_core, b_core, complex(1), poly_shape)
        c_prime = math.sum(poly_core, axis=[i for i in range(n, n + m)]) * ci
        block = Ai[:n, :n]
        A_decomp = math.block(
            [[block, math.eye_like(block)], [math.eye_like(block), math.zeros_like(block)]]
        )
        b_decomp = math.concat((bi[:n], math.zeros((n,), dtype=bi.dtype)), axis=-1)
        return A_decomp, b_decomp, c_prime

    def reorder(self, order_CV: Sequence[int]):
        r"""
        Reorders the CV indices of an (A,b,c) triple.
        The length of ``order_CV`` must be the number of CV variables.
        """
        if len(order_CV) != self.num_CV_vars:
            raise ValueError(f"order_CV must have length {self.num_CV_vars}, got {len(order_CV)}")
        # Add derived variable indices after CV indices
        order = list(order_CV) + list(
            range(self.num_CV_vars, self.num_CV_vars + self.num_derived_vars)
        )
        A = math.gather(math.gather(self.A, order, axis=-1), order, axis=-2)
        b = math.gather(self.b, order, axis=-1)
        return self.__class__(A, b, self.c, self.num_derived_vars)

    def simplify(self) -> None:
        r"""
        Simplifies an ansatz by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the ansatz has already been simplified, so it is always safe to call.
        """
        if self._simplified:
            return

        to_keep = self._find_unique_terms_sorted()
        self._A = math.gather(self._A, to_keep, axis=0)
        self._b = math.gather(self._b, to_keep, axis=0)
        self._c = math.gather(self._c, to_keep, axis=0)  # already added
        self._simplified = True

    def _find_unique_terms_sorted(self) -> list[int]:
        r"""
        Finds unique terms by first sorting the batch dimension and adds the corresponding c values.
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
                self._c = math.update_add_tensor(self.c, [[d0]], [self.c[d]])
        return to_keep

    def to_dict(self) -> dict[str, ArrayLike]:
        r"""Returns a dictionary representation of the ansatz. For serialization purposes."""
        return {"A": self.A, "b": self.b, "c": self.c, "num_derived_vars": self.num_derived_vars}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]):
        r"""
        Computes the trace of the ansatz across the specified index pairs.
        The index pairs must belong to the CV variables.
        """
        if len(idx_z) != len(idx_zconj):
            raise ValueError("idx_z and idx_zconj must have the same length.")
        if any(i >= self.num_CV_vars for i in idx_z) or any(
            i >= self.num_CV_vars for i in idx_zconj
        ):
            raise ValueError(
                f"All indices must be between 0 and {self.num_CV_vars-1}. Got {idx_z} and {idx_zconj}."
            )
        A, b, c = complex_gaussian_integral_1(self.triple, idx_z, idx_zconj, measure=-1.0)
        return self.__class__(A, b, c, self.num_derived_vars)

    def eval(self: PolyExpAnsatz, z: Batch[Vector]) -> Batch[ComplexTensor]:
        r"""
        Evaluates the ansatz at a batch of points ``z`` in C^(*b, n), where ``b`` is the batch shape
        and ``n`` is the number of CV variables.
        Note that since the ansatz may itself have a batch size `L`, the output will have shape
        ``(L, *b)``. If the batch size of size `L` is intended as a linear superposition of terms,
        then the batch of values of the ansatz should be the sum of the returned array along axis 0.

        Args:
            z: Point(s) in C^(*b, n) where the function is evaluated, ``b`` stands for batch shape.

        Returns:
            The value of the function at the point(s) with the same batch dimensions as ``z``.
            The output has shape (L, *b) where L is the batch size of the ansatz.
        """
        z = math.atleast_2d(z)
        z_batch_shape, z_dim = z.shape[:-1], z.shape[-1]
        if z_dim != self.num_CV_vars:
            raise ValueError(
                f"The last dimension of `z` must equal the number of CV variables {self.num_CV_vars}, got {z_dim}."
            )
        z = math.reshape(z, (-1, z_dim))  # shape (k, num_CV_vars)

        exp_sum = self._compute_exp_part(z)  # shape (batch_size, k)
        if self.num_derived_vars == 0:  # purely gaussian
            ret = math.einsum("ik,i...->ik...", exp_sum, self.c)
        else:
            poly = self._compute_polynomial_part(z)  # shape (batch_size, k, *derived_shape)
            ret = self._combine_exp_and_poly(exp_sum, poly)
        return math.reshape(ret, (self.batch_size,) + z_batch_shape)

    def _compute_exp_part(self, z: Batch[Vector]) -> Batch[Scalar]:
        r"""Computes the exponential part of the ansatz evaluation. Needed in ``_eval``."""
        n = self.num_CV_vars
        A_part = math.einsum("ka,kb,iab->ik", z, z, self.A[:, :n, :n])
        b_part = math.einsum("ka,ia->ik", z, self.b[:, :n])
        return math.exp(1 / 2 * A_part + b_part)  # shape (batch_size, k)

    def _compute_polynomial_part(self, z: Batch[Vector]) -> Batch[Scalar]:
        r"""Computes the polynomial part of the ansatz evaluation. Needed in ``_eval``."""
        n = self.num_CV_vars
        b_poly = math.einsum("iab,ka->ikb", self.A[:, :n, n:], z) + self.b[:, None, n:]
        A_poly = self.A[:, n:, n:]  # shape (batch_size,derived_vars,derived_vars)
        result = []
        for Ai, bi in zip(A_poly, b_poly):
            result.append(
                math.hermite_renormalized_batch(Ai, bi, complex(1), self.shape_derived_vars)
            )
        return math.astensor(result)

    def _combine_exp_and_poly(
        self, exp_sum: Batch[ComplexTensor], poly: Batch[ComplexTensor]
    ) -> Batch[ComplexTensor]:
        r"""Combines exponential and polynomial parts using einsum. Needed in ``_eval``."""
        d = np.prod(self.shape_derived_vars)
        c = math.reshape(self.c, (self.batch_size, d))
        poly = math.reshape(poly, (self.batch_size, -1, d))
        # return math.einsum("ik,idD,ikd->ikD", exp_sum, c, poly, optimize=True)
        return math.einsum("ik,il,inl->in", exp_sum, c, poly)

    def partial_eval(self, z: Vector, indices: tuple[int, ...]) -> PolyExpAnsatz:
        r"""
        Returns a new ansatz that corresponds to currying (partially evaluate) the current one.
        For example, if ``self`` represents the function ``F(z0,z1,z2)``, the call
        ``self.partial_eval(np.array([2.0,3.0]), (0,2))`` returns
        ``G(z1) = F(2.0, z1, 3.0)`` as a new ansatz of a single variable.
        The vector ``z`` must have shape (b,r), where ``r`` is the number of indices in ``indices``.
        The batch dimension of the resulting ansatz will be the product of batch dimension of ``z``
        and the batch dimension of the ansatz.

        Args:
            z: vector in ``C^(b, r)`` where the function is evaluated.
            indices: indices of the variables of the ansatz to be evaluated.

        Returns:
            A new ansatz.
        """
        if len(indices) == self.num_CV_vars:
            raise ValueError(
                "Cannot curry a function of the same number of variables as the ansatz. "
                "Use the _eval or __call__ method instead."
            )

        # evaluated, remaining and derived indices
        e = indices
        r = [i for i in range(self.num_CV_vars) if i not in indices]
        d = list(range(self.num_CV_vars, self.num_vars))

        # new A of shape (batch_size * b, r+d, r+d)
        new_A = math.gather(math.gather(self.A, r + d, axis=-1), r + d, axis=-2)
        new_A = math.tile(new_A, (z.shape[0], 1, 1))

        # new b of shape (batch_size * b, r+d)
        A_er = math.gather(math.gather(self.A, e, axis=-1), r, axis=-2)  # shape (batch_size, e, r)
        b_r = math.einsum("ier,be->ibr", A_er, z)  # shape (batch_size, b, r)

        if len(d) > 0:
            A_ed = math.gather(
                math.gather(self.A, e, axis=-1), d, axis=-2
            )  # shape (batch_size, e, d)
            b_d = math.einsum("ied,be->ibd", A_ed, z)  # shape (batch_size, b, d)
            new_b = math.gather(self.b, r + d, axis=-1)[:, None, :] + math.concat(
                (b_r, b_d), axis=-1
            )
        else:
            new_b = math.gather(self.b, r, axis=-1)[:, None, :] + b_r

        new_b = math.reshape(new_b, (self.batch_size * z.shape[0], -1))

        # new c of shape (batch_size * b,)
        A_ee = math.gather(math.gather(self.A, e, axis=-1), e, axis=-2)  # shape (batch_size, e, e)
        A_part = math.einsum("be,bf,ief->ib", z, z, A_ee)  # shape (batch_size, b)
        b_part = math.einsum(
            "be,ie->ib", z, math.gather(self.b, e, axis=-1)
        )  # shape (batch_size, b)
        exp_sum = math.exp(1 / 2 * A_part + b_part)  # shape (batch_size, b)
        new_c = math.einsum("ib,i...->ib...", exp_sum, self.c)
        c_shape = (
            (self.batch_size * z.shape[0], self.num_derived_vars)
            if self.num_derived_vars > 0
            else (self.batch_size * z.shape[0],)
        )
        new_c = math.reshape(new_c, c_shape)

        return PolyExpAnsatz(
            new_A,
            new_b,
            new_c,
            self.num_derived_vars,
        )

    def _equal_no_array(self, other: PolyExpAnsatz) -> bool:
        self.simplify()
        other.simplify()
        return np.allclose(self.b, other.b, atol=1e-10) and np.allclose(self.A, other.A, atol=1e-9)

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
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
        In the case where ``c`` on self and other have different shapes it will add padding zeros to make
        the shapes fit. Example: If the shape of ``c1`` is (1,3,4,5) and the shape of ``c2`` is (10,5,4,3) then the
        shape of the combined object will be (11,5,4,5).
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot add PolyExpAnsatz and {other.__class__}.")
        if self.num_CV_vars != other.num_CV_vars:
            raise ValueError(
                f"The number of CV variables must match. Got {self.num_CV_vars} and {other.num_CV_vars}."
            )

        def combine_arrays(array1, array2):
            """Combine arrays, padding with zeros if shapes differ in non-derived dimensions."""
            shape1 = array1.shape[1:]
            shape2 = array2.shape[1:]
            max_shape = tuple(map(max, zip(shape1, shape2)))
            pad_widths1 = [(0, 0)] + [(0, t - s) for t, s in zip(max_shape, shape1)]
            pad_widths2 = [(0, 0)] + [(0, t - s) for t, s in zip(max_shape, shape2)]
            padded_array1 = math.pad(array1, pad_widths1, "constant")
            padded_array2 = math.pad(array2, pad_widths2, "constant")
            return math.concat([padded_array1, padded_array2], axis=0)

        n_derived_vars = max(self.num_derived_vars, other.num_derived_vars)
        self_pad = max(0, other.num_derived_vars - self.num_derived_vars)
        other_pad = max(0, self.num_derived_vars - other.num_derived_vars)

        self_A = math.pad(self.A, [(0, 0), (0, self_pad), (0, self_pad)])
        other_A = math.pad(other.A, [(0, 0), (0, other_pad), (0, other_pad)])
        self_b = math.pad(self.b, [(0, 0), (0, self_pad)])
        other_b = math.pad(other.b, [(0, 0), (0, other_pad)])

        combined_matrices = math.concat([self_A, other_A], axis=0)
        combined_vectors = math.concat([self_b, other_b], axis=0)
        combined_arrays = combine_arrays(
            math.atleast_nd(self.c, n_derived_vars + 1),
            math.atleast_nd(other.c, n_derived_vars + 1),
        )

        return PolyExpAnsatz(
            combined_matrices,
            combined_vectors,
            combined_arrays,
            n_derived_vars,
        )

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Tensor product of this PolyExpAnsatz with another. Equivalent to :math:`H(a,b) = F(a) * G(b)`.
        As it distributes over addition on both self and other, the batch size of the result is the
        product of the batch size of this ansatz and the other one.

        Args:
            other: Another PolyExpAnsatz.

        Returns:
            The tensor product of this PolyExpAnsatz and other.
        """
        As, bs, cs = join_Abc(self.triple, other.triple, mode="kron")
        return PolyExpAnsatz(As, bs, cs)

    def __call__(
        self, *z: Vector | None, mode: Literal["zip", "kron"] = "kron"
    ) -> Scalar | PolyExpAnsatz:
        r"""
        Returns either the value of the ansatz or a new ansatz depending on the arguments.
        If an argument is None, the corresponding variable is not evaluated, and the method
        returns a new ansatz with the remaining variables unevaluated.
        For example, if the ansatz is a function of 3 variables F(z1, z2, z3) and we want to
        evaluate it at a point in C^2, we would get a new ansatz with one variable unevaluated:
        F(z1, z2, None) or F(z1, None, z3), or F(None, z2, z3). The ``mode`` argument can be used
        to specify how the vectors of arguments are broadcast together. The default is "zip", which
        is to broadcast the vectors pairwise. The alternative is "kron", which is to broadcast the
        the vectors Kronecker-style. For example, ``F(z1, z2, mode="zip")`` returns the array of values
        ``[F(z1[0], z2[0]), F(z1[1], z2[1]), …]``. On the other hand, ``F(z1, z2, mode="kron")``
        returns the Kronecker product of the vectors, i.e. ``[[F(z1[0], z2[0]), F(z1[0], z2[1]), …],
        [F(z1[1], z2[0]), F(z1[1], z2[1]), …], …]``. The 'kron' style is useful if we want to
        pass the points along each axis independently from each other. In `zip` mode the batch
        dimensions of the z vectors must match, while in `kron` mode they can differ, and the result
        will have a batch dimension equal to the product of the batch dimensions the ansatz, followed
        by the reshaped batch dimensions of the z vectors.

        TODO: make the kron version more efficient by avoiding the meshgrid.

        Args:
            z: points in C where the function is (partially) evaluated or None if the variable is
            not evaluated.
            mode: "zip" or "kron"

        Returns:
            The value of the function or a new ansatz.
        """
        evaluated_indices = [i for i, zi in enumerate(z) if zi is not None]
        if len(z) > self.num_CV_vars:
            raise ValueError(
                f"The ansatz was called with {len(z)} variables, "
                f"but it only has {self.num_CV_vars} CV variables."
            )

        # Full evaluation: all continuous variables have been provided.
        if len(evaluated_indices) == self.num_CV_vars:
            if mode == "zip":
                only_z = [math.atleast_2d(zi) for zi in z]
                batch_sizes = [zi.shape[0] for zi in only_z]
                if not all(bs == batch_sizes[0] for bs in batch_sizes):
                    raise ValueError(
                        f"In mode 'zip' all z vectors must have the same batch size, got {batch_sizes}."
                    )
                # Concatenate along the last axis to form an array of shape (batch, n)
                z_input = math.concat(only_z, axis=-1)
                return self.eval(z_input)
            elif mode == "kron":
                only_z = [math.atleast_1d(zi) for zi in z]
                if any(zi.ndim > 1 for zi in only_z):
                    raise ValueError(
                        "In `kron` mode the z vectors must not have a batch dimension."
                    )
                # Create a meshgrid from the provided arrays; they may have different batch sizes.
                grid = np.meshgrid(*only_z, indexing="ij")
                z_combined = math.astensor(np.stack(grid, axis=-1))  # shape (b0, b1, …, b_{n}, n)
                grid_shape = z_combined.shape[:-1]  # (b0, b1, …, b_n)
                z_flat = math.reshape(z_combined, (-1, self.num_CV_vars))  # shape (prod(b_i), n)
                result_flat = self.eval(
                    z_flat
                )  # returns an array of shape (batch_size, prod(b_i)) but squeezed because of no batch
                return math.squeeze(
                    math.reshape(result_flat, (self.batch_size,) + grid_shape)
                )  # (batch_size, b0, b1, …, b_n)
        else:
            # Partial evaluation: some CV variables are not provided.
            # In partial evaluation, the provided z's must not have a batch dimension.
            only_z = [math.transpose(math.atleast_2d(zi)) for zi in z if zi is not None]
            # For partial evaluation (i.e. currying) it's always zip
            z_input = math.concat(only_z, axis=1)
            return self.partial_eval(z_input, evaluated_indices)

    def __eq__(self, other: PolyExpAnsatz) -> bool:
        if not isinstance(other, PolyExpAnsatz):
            return False
        return self._equal_no_array(other) and np.allclose(self.c, other.c, atol=1e-10)

    def __mul__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot multiply PolyExpAnsatz and {other.__class__}.") from e

        if self.num_CV_vars != other.num_CV_vars:
            raise TypeError(
                "The number of CV variables of the two ansatze must be the same. "
                f"Got {self.num_CV_vars} and {other.num_CV_vars}."
            )
        # outer product along batch via tile and repeat to get all pairs
        A1 = math.tile(self.A, (other.A.shape[0], 1, 1))
        b1 = math.tile(self.b, (other.b.shape[0], 1))
        A2 = math.repeat(other.A, self.A.shape[0], axis=0)
        b2 = math.repeat(other.b, self.b.shape[0], axis=0)

        batch_size = self.batch_size * other.batch_size
        n = self.num_CV_vars  # alpha
        m1 = self.num_derived_vars  # beta1
        m2 = other.num_derived_vars  # beta2
        newA = math.zeros((batch_size, n + m1 + m2, n + m1 + m2), dtype=math.complex128)
        newb = math.zeros((batch_size, n + m1 + m2), dtype=math.complex128)

        newA[:, :n, :n] = A1[:, :n, :n] + A2[:, :n, :n]
        newb[:, :n] = b1[:, :n] + b2[:, :n]

        if m1 > 0:
            newA[:, :n, n:m1] = A1[:, :n, -m1:]
            newA[:, n:m1, :n] = A1[:, -m1:, :n]
            newA[:, n:m1, n:m1] = A1[:, -m1:, -m1:]
            newb[:, n:m1] = b1[:, -m1:]

        if m2 > 0:
            newA[:, :n, -m2:] = A2[:, :n, -m2:]
            newA[:, -m2:, :n] = A2[:, -m2:, :n]
            newA[:, -m2:, -m2:] = A2[:, -m2:, -m2:]
            newb[:, -m2:] = b2[:, -m2:]

        self_c = math.reshape(
            self.c,
            (self.batch_size, int(math.prod(self.shape_derived_vars))),
        )
        other_c = math.reshape(
            other.c,
            (other.batch_size, int(math.prod(other.shape_derived_vars))),
        )
        newc = math.einsum("ij,kl->ijl", self_c, other_c)
        newc = math.reshape(
            newc,
            (batch_size,) + self.shape_derived_vars + other.shape_derived_vars,
        )

        return PolyExpAnsatz(A=newA, b=newb, c=newc, num_derived_vars=m1 + m2)

    def __neg__(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, -self.c, self.num_derived_vars)

    def __truediv__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot multiply PolyExpAnsatz and {other.__class__}.") from e

        if self.num_CV_vars != other.num_CV_vars:
            raise TypeError(
                "The number of CV variables of the two ansatze must be the same. "
                f"Got {self.num_CV_vars} and {other.num_CV_vars}."
            )
        # outer product along batch via tile and repeat to get all pairs
        A1 = math.tile(self.A, (other.A.shape[0], 1, 1))
        b1 = math.tile(self.b, (other.b.shape[0], 1))
        A2 = math.repeat(other.A, self.A.shape[0], axis=0)
        b2 = math.repeat(other.b, self.b.shape[0], axis=0)

        batch_size = self.batch_size * other.batch_size
        n = self.num_CV_vars  # alpha
        m1 = self.num_derived_vars  # beta1
        m2 = other.num_derived_vars  # beta2
        newA = math.zeros((batch_size, n + m1 + m2, n + m1 + m2), dtype=math.complex128)
        newb = math.zeros((batch_size, n + m1 + m2), dtype=math.complex128)

        newA[:, :n, :n] = A1[:, :n, :n] - A2[:, :n, :n]
        newb[:, :n] = b1[:, :n] - b2[:, :n]

        if m1 > 0:
            newA[:, :n, n:m1] = A1[:, :n, -m1:]
            newA[:, n:m1, :n] = A1[:, -m1:, :n]
            newA[:, n:m1, n:m1] = A1[:, -m1:, -m1:]
            newb[:, n:m1] = b1[:, -m1:]

        if m2 > 0:
            newA[:, :n, -m2:] = A2[:, :n, -m2:]
            newA[:, -m2:, :n] = A2[:, -m2:, :n]
            newA[:, -m2:, -m2:] = A2[:, -m2:, -m2:]
            newb[:, -m2:] = b2[:, -m2:]

        self_c = math.reshape(
            self.c,
            (self.batch_size, int(math.prod(self.shape_derived_vars))),
        )
        other_c = 1 / math.reshape(
            other.c,
            (other.batch_size, int(math.prod(other.shape_derived_vars))),
        )
        newc = math.einsum("ij,kl->ijl", self_c, other_c)
        newc = math.reshape(
            newc,
            (batch_size,) + self.shape_derived_vars + other.shape_derived_vars,
        )

        return PolyExpAnsatz(A=newA, b=newb, c=newc, num_derived_vars=m1 + m2)
