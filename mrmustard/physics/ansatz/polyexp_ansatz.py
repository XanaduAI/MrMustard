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

# pylint: disable=too-many-instance-attributes,too-many-positional-arguments, too-many-public-methods, inconsistent-return-statements

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

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable

from mrmustard.utils.argsort import argsort_gen

from .base import Ansatz

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""
    This class represents the ansatz function:

        :math:`F^{(i)}(z) = \sum_k c^{(i)}_{k} \partial_y^k \textrm{exp}(\frac{1}{2}(z,y)^T A^{(i)} (z,y) + (z,y)^T b^{(i)})|_{y=0}`

    with ``k`` a multi-index. The ``i`` multi-index is a batch index of shape ``L`` that can be used
    for linear superposition or batching purposes. The ``c^{(i)}_k`` tensors are contracted with the
    array of derivatives :math:`\partial_y^k` to form polynomials of derivatives.
    The matrices :math:`A^{(i)}` and vectors :math:`b^{(i)}` are the parameters of the exponential
    terms in the ansatz, with :math:`z\in\mathbb{C}^{n}` and :math:`y\in\mathbb{C}^{m}`.
    ``A`` and b`` have shape ``(*L, n+m, n+m)`` and ``(*L, n+m)``, respectivly.
    The tensors :math:`c^{(i)}_{k}` contain the coefficients of the polynomial of derivatives and
    have shape ``(*L, *derived)``, where ``*derived`` is the shape of the derived variables, which
    implies ``len(c.shape[1:]) = m``.

    .. code-block::

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz

        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array([1.0])
        >>> F = PolyExpAnsatz(A, b, c, num_derived_vars=1)
        >>> val = F(1.0, 2.0)

    Args:
        A: A batch of quadratic coefficient :math:`A^{(i)}`.
        b: A batch of linear coefficients :math:`b^{(i)}`.
        c: A batch of arrays :math:`c^{(i)}`.
        num_derived_vars: The number of variables :math:`y` that are derived by the polynomial of derivatives.

    TODO: infer num_derived_vars from the shape of c
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix] | None,
        b: Batch[ComplexVector] | None,
        c: Batch[ComplexTensor] | None,
        num_derived_vars: int = 0,  # i.e. size of y
        name: str = "",
    ):
        super().__init__()
        # TODO: consider not using a batch dimension by default
        self._A = math.atleast_3d(math.astensor(A)) if A is not None else None
        self._b = math.atleast_2d(math.astensor(b)) if b is not None else None
        self._c = math.atleast_nd(math.astensor(c), num_derived_vars + 1) if c is not None else None
        self.num_derived_vars = num_derived_vars
        self.name = name
        self._simplified = False
        self._fn = None
        self._fn_kwargs = {}
        self._batch_shape = self._A.shape[:-2] if A is not None else None

    def __repr__(self) -> str:  # TODO: update to show batch shape
        r"""Returns a string representation of the PolyExpAnsatz object."""
        self._generate_ansatz()  # Ensure parameters are generated if needed

        # Get basic information
        batch_size = self.batch_size
        num_cv = self.num_CV_vars
        num_derived = self.num_derived_vars
        total_vars = self.num_vars

        # Format shape information
        A_shape = f"{self.A.shape}" if self._A is not None else "None"
        b_shape = f"{self.b.shape}" if self._b is not None else "None"
        c_shape = f"{self.c.shape}" if self._c is not None else "None"

        # Create a descriptive name
        display_name = f'"{self.name}"' if self.name else "unnamed"

        # Build the representation string
        repr_str = [
            f"PolyExpAnsatz({display_name})",
            f"  Batch size: {batch_size}",
            f"  Variables: {num_cv} CV + {num_derived} derived = {total_vars} total",
            f"  Parameter shapes:",
            f"    A: {A_shape}",
            f"    b: {b_shape}",
            f"    c: {c_shape}",
        ]

        # Add information about simplification status
        if self._simplified:
            repr_str.append("  Status: simplified")

        # Add information about function generation if applicable
        if self._fn is not None:
            fn_name = getattr(self._fn, "__name__", str(self._fn))
            repr_str.append(f"  Generated from: {fn_name}")
            if self._fn_kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in self._fn_kwargs.items())
                repr_str.append(f"  Parameters: {param_str}")

        return "\n".join(repr_str)

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
            self._batch_shape = self._A.shape[:-2]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""
        The shape of the batch of parameters.
        """
        return self._batch_shape

    @property
    def batch_dims(self) -> tuple[int, ...]:
        r"""
        The number of batch dimensions of the parameters.
        """
        return len(self.batch_shape)

    @property
    def _A_vectorized(self) -> Batch[ComplexMatrix]:
        r"""
        A view of self.A with the batch dimension flattened.
        """
        return math.reshape(self.A, (-1, self.num_vars, self.num_vars))

    @property
    def _b_vectorized(self) -> Batch[ComplexVector]:
        r"""
        A view of self.b with the batch dimension flattened.
        """
        return math.reshape(self.b, (-1, self.num_vars))

    @property
    def _c_vectorized(self) -> Batch[ComplexTensor]:
        r"""
        A view of self.c with the batch dimension flattened.
        """
        return math.reshape(self.c, (-1, *self.shape_derived_vars))

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
        The batch of polynomial coefficients :math:`c^{(i)}_{k}`.
        """
        self._generate_ansatz()
        return self._c

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz, i.e. F(0)
        """
        if self.num_derived_vars == 0:
            return self.c
        else:
            return self.eval([])

    @property
    def batch_size(self) -> int:
        if self._batch_shape is None:
            return 1
        return math.prod(self._batch_shape)

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
        return self.A.shape[-1]

    @property
    def shape_derived_vars(self) -> tuple[int, ...]:
        r"""
        The shape of the coefficients of the polynomial of derivatives.
        """
        return self.c.shape[self.batch_dims :]

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

    def to_dict(self) -> dict[str, ArrayLike]:
        r"""Returns a dictionary representation of the ansatz. For serialization purposes."""
        return {"A": self.A, "b": self.b, "c": self.c, "num_derived_vars": self.num_derived_vars}

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> PolyExpAnsatz:
        r"""Creates an ansatz given a function and its kwargs. This ansatz is lazily instantiated, i.e.
        the function is not called until the A,b,c attributes are accessed (even internally)."""
        ansatz = cls(None, None, None, None)
        ansatz._fn = fn
        ansatz._fn_kwargs = kwargs
        return ansatz

    @staticmethod
    def _outer_product_batch_str(*ndims: int) -> str:
        r"""
        Creates the einsum string for the outer product of the given tuple of dimensions.
        E.g. for (2,1,3) it returns ab,c,def->abcdef
        """
        strs = []
        offset = 0
        for ndim in ndims:
            strs.append("".join([chr(97 + i + offset) for i in range(ndim)]))
            offset += ndim
        return ",".join(strs) + "->" + "".join(strs)

    @staticmethod
    def _reshape_args_to_batch_string(
        args: list[ArrayLike], batch_string: str
    ) -> tuple[list[ArrayLike], tuple[int, ...]]:
        r"""
        Reshapes arguments to match the batch string by inserting singleton dimensions where needed
        so that they are broadcastable.
        E.g. given two arrays of shape (2,7) and (3,7) and string ab,cb->abc, it reshapes them to
        shape (2,7,1) and (1,7,3).
        """
        # Parse the batch string
        input_specs, output_spec = batch_string.split("->")
        input_specs = input_specs.split(",")
        if len(input_specs) != len(args):
            raise ValueError(
                f"Number of input specifications ({len(input_specs)}) does not match number of arguments ({len(args)})"
            )

        args = [math.astensor(arg) for arg in args]

        # Determine the size of each dimension in the output
        dim_sizes = {}
        for arg, spec in zip(args, input_specs):
            for dim, label in zip(arg.shape, spec):
                if label in dim_sizes and dim_sizes[label] != dim:
                    raise ValueError(
                        f"Dimension {label} has inconsistent sizes: got {dim_sizes[label]} and {dim}"
                    )
                dim_sizes[label] = dim

        reshaped = []
        for arg, spec in zip(args, input_specs):
            new_shape = [dim_sizes[label] if label in spec else 1 for label in output_spec]
            reshaped.append(math.reshape(arg, new_shape))
        return reshaped

    def contract(
        self,
        other: PolyExpAnsatz,
        batch_str: str = "",
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
    ) -> PolyExpAnsatz:
        r"""
        Contracts two ansatze across the specified CV variables and batch dimensions.
        CV variables are indexed by integers, while for batch dimensions the string has the same
        syntax as in ``np.einsum``.

        Args:
            other: The other PolyExpAnsatz to contract with.
            batch_str: The batch dimensions to contract over with the same syntax as in ``np.einsum``.
                If not indicated, the batch dimensions are taken in outer product
            idx1: The CV variables of the first ansatz to contract.
            idx2: The CV variables of the second ansatz to contract.

        Returns:
            The contracted ansatz.
        """
        if batch_str == "":
            batch_str = self._outer_product_batch_str(self.batch_dims, other.batch_dims)
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

        A, b, c = complex_gaussian_integral_2(self.triple, other.triple, idx1, idx2, batch_str)
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

        TODO: find a way to avoid the loop over the batch dimensions
        """
        if self.num_derived_vars < self.num_CV_vars:
            return self
        A_dec = []
        b_dec = []
        c_dec = []
        for Ai, bi, ci in zip(self._A_vectorized, self._b_vectorized, self._c_vectorized):
            A_dec_i, b_dec_i, c_dec_i = self._decompose_single(Ai, bi, ci)
            A_dec.append(A_dec_i)
            b_dec.append(b_dec_i)
            c_dec.append(c_dec_i)
        A_dec = math.reshape(
            math.concat(A_dec, axis=0), self._batch_shape + (self.num_CV_vars, self.num_CV_vars)
        )
        b_dec = math.reshape(math.concat(b_dec, axis=0), self._batch_shape + (self.num_CV_vars,))
        c_dec = math.reshape(
            math.concat(c_dec, axis=0),
            self._batch_shape + self.shape_derived_vars + self.shape_derived_vars,
        )
        return PolyExpAnsatz(A_dec, b_dec, c_dec, self.num_CV_vars)

    def _decompose_single(self, Ai, bi, ci):
        r"""
        Decomposes a single batch element of the ansatz.
        """
        n = self.num_CV_vars
        A_core = math.block(
            [[math.zeros((n, n), dtype=Ai.dtype), Ai[:n, n:]], [Ai[n:, :n], Ai[n:, n:]]]
        )
        b_core = math.concat((math.zeros((n,), dtype=bi.dtype), bi[n:]), axis=-1)
        pulled_out_input_shape = (math.sum(self.shape_derived_vars),) * n
        poly_shape = pulled_out_input_shape + self.shape_derived_vars
        poly_core = math.hermite_renormalized(A_core, b_core, complex(1), poly_shape).reshape(
            pulled_out_input_shape + (-1,)
        )
        c_prime = math.einsum("...i,i->...", poly_core, ci.reshape(-1))
        block = Ai[:n, :n]
        A_decomp = math.block(
            [[block, math.eye_like(block)], [math.eye_like(block), math.zeros_like(block)]]
        )
        b_decomp = math.concat((bi[:n], math.zeros((n,), dtype=bi.dtype)), axis=-1)
        return A_decomp, b_decomp, c_prime

    def reorder_CV(self, order: Sequence[int]):
        r"""
        Reorders the CV indices of an (A,b,c) triple.
        The length of ``order`` must equal the number of CV variables.
        This method returns a new ansatz object.
        """
        if len(order) != self.num_CV_vars:
            raise ValueError(f"order must have length {self.num_CV_vars}, got {len(order)}")
        # Add derived variable indices after CV indices
        order = list(order) + list(
            range(self.num_CV_vars, self.num_CV_vars + self.num_derived_vars)
        )
        A = math.gather(math.gather(self.A, order, axis=-1), order, axis=-2)
        b = math.gather(self.b, order, axis=-1)
        return self.__class__(A, b, self.c, self.num_derived_vars)

    def simplify(self) -> None:
        r"""
        Simplifies an ansatz by combining terms that have the same exponential part, i.e. two components
        of the batch are considered equal if their ``A`` and ``b`` are equal. In this case only one
        is kept and the corresponding ``c`` are added.

        Will return early if the ansatz has already been simplified, so it is safe to call.

        TODO: consider returning a new ansatz rather than mutating this one
        """
        if self._simplified:
            return

        to_keep = self._find_unique_terms_sorted()
        _A = math.gather(self._A_vectorized, to_keep, axis=0)
        _b = math.gather(self._b_vectorized, to_keep, axis=0)
        _c = math.gather(self._c_vectorized, to_keep, axis=0)  # already added
        self._A = math.reshape(_A, self._batch_shape + (self.num_vars, self.num_vars))
        self._b = math.reshape(_b, self._batch_shape + (self.num_vars,))
        self._c = math.reshape(
            _c, self._batch_shape + self.shape_derived_vars + self.shape_derived_vars
        )
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
        mat, vec = self._A_vectorized[d0], self._b_vectorized[d0]

        for d in range(1, self.batch_size):
            if not (
                np.array_equal(mat, self._A_vectorized[d])
                and np.array_equal(vec, self._b_vectorized[d])
            ):
                to_keep.append(d)
                d0 = d
                mat, vec = self._A_vectorized[d0], self._b_vectorized[d0]
            else:
                d0r = np.unravel_index(d0, self.batch_shape)
                dr = np.unravel_index(d, self.batch_shape)
                self._c = math.update_add_tensor(self.c, [[d0r]], [self.c[dr]])
        return to_keep

    def _order_batch(self):
        r"""
        This method orders the batch dimension by the lexicographical order of the
        flattened arrays (A, b, c). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two PolyExp ansatz.
        """
        generators = [
            itertools.chain(
                math.asnumpy(self._b_vectorized[i]).flat,
                math.asnumpy(self._A_vectorized[i]).flat,
                math.asnumpy(self._c_vectorized[i]).flat,
            )
            for i in range(self.batch_size)
        ]
        sorted_indices = argsort_gen(generators)
        _A = math.gather(self._A_vectorized, sorted_indices, axis=0)
        _b = math.gather(self._b_vectorized, sorted_indices, axis=0)
        _c = math.gather(self._c_vectorized, sorted_indices, axis=0)
        self._A = math.reshape(_A, self._batch_shape + (self.num_vars, self.num_vars))
        self._b = math.reshape(_b, self._batch_shape + (self.num_vars,))
        self._c = math.reshape(
            _c, self._batch_shape + self.shape_derived_vars + self.shape_derived_vars
        )

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]):
        r"""
        Computes the trace of the ansatz across the specified pairs of CV variables.
        TODO: make the measure kw available
        """
        if len(idx_z) != len(idx_zconj):
            raise ValueError("idx_z and idx_zconj must have the same length.")
        if len(set(idx_z + idx_zconj)) != len(idx_z) + len(idx_zconj):
            raise ValueError(
                f"Indices must be unique: {set(idx_z).intersection(idx_zconj)} are repeated."
            )
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
        Evaluates the ansatz at the given batch of points in C^(*b, n).
        If the ansatz itself has batch shape ``L`` then the result has shape (*b, *L).
        If the ansatz batch shape L (or a subset of its dimensions) represents a linear superposition
        of terms rather than independent functions, one should sum the returned array along those
        axes to get the actual ansatz values. For example, the ansatz of a cat state is a linear
        superposition of two gaussian terms.

        Args:
            z: A batch of points where the function is evaluated. The shape should be (*b, n) where:
               - *b represents any number of batch dimensions (even none).
               - n is the number of CV variables in the ansatz

        Returns:
            The evaluated function values with shape (*b, *L) where:
               - *b are the same batch dimensions as the input
               - L is the batch shape of the ansatz itself
        """
        # print(z)
        # z = math.atleast_2d(z)
        # z = math.cast(z, dtype=self.A.dtype)

        z_batch_shape, z_dim = z.shape[:-1], z.shape[-1]
        if z_dim != self.num_CV_vars:
            raise ValueError(
                f"The last dimension of `z` must equal the number of CV variables {self.num_CV_vars}, got {z_dim}."
            )
        z = math.reshape(z, (np.prod(z_batch_shape), z_dim))  # shape (k, num_CV_vars)

        exp_sum = self._compute_exp_part(z)  # shape (batch_size, k)
        if self.num_derived_vars == 0:  # purely gaussian
            ret = math.einsum("ik,i...->ik...", exp_sum, self.c)
        else:
            poly = self._compute_polynomial_part(z)  # shape (batch_size, k, *derived_shape)
            ret = self._combine_exp_and_poly(exp_sum, poly)
        ret = math.transpose(ret, list(range(1, len(ret.shape))) + [0])
        return math.reshape(ret, z_batch_shape + self.batch_shape)

    def _compute_exp_part(self, z: Batch[Vector]) -> Batch[Scalar]:
        r"""Computes the exponential part of the ansatz evaluation. Needed in ``_eval``."""
        n = self.num_CV_vars
        A_part = math.einsum("ka,kb,iab->ik", z, z, self._A_vectorized[:, :n, :n])
        b_part = math.einsum("ka,ia->ik", z, self._b_vectorized[:, :n])
        return math.exp(1 / 2 * A_part + b_part)  # shape (batch_size, k)

    def _compute_polynomial_part(self, z: Batch[Vector]) -> Batch[Scalar]:
        r"""Computes the polynomial part of the ansatz evaluation. Needed in ``_eval``."""
        n = self.num_CV_vars
        b_poly = (
            math.einsum("iab,ka->ikb", self._A_vectorized[:, :n, n:], z)
            + self._b_vectorized[:, None, n:]
        )
        A_poly = self._A_vectorized[:, n:, n:]  # shape (batch_size,derived_vars,derived_vars)
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
        c = math.reshape(self._c_vectorized, (self.batch_size, d))
        poly = math.reshape(poly, (self.batch_size, -1, d))
        return math.einsum("ik,il,inl->in", exp_sum, c, poly)

    def partial_eval(self, z: ArrayLike, indices: tuple[int, ...]) -> PolyExpAnsatz:
        r"""
        Partially evaluates the ansatz by fixing some of its variables to specific values.

        This method creates a new ansatz with fewer variables by substituting the specified
        variables with their given values. The remaining variables keep their original positions
        in the function signature.

        Example:
            If this ansatz represents F(z0,z1,z2), then:
            ```
            new_ansatz = self.partial_eval([2.0,3.0], indices=(0,2))
            ```
            returns a new ansatz G(z1) = F(2.0, z1, 3.0)

        Args:
            z: Values for the variables being fixed. Can be:
               - Shape (r,): A single list of values for r variables
               - Shape (*b, r): Batch of r values of shape *b
               Where r is the number of indices in `indices`
            indices: Indices of the variables to be fixed to the values in z

        Returns:
            A new PolyExpAnsatz with fewer variables. If the original ansatz has batch
            dimensions *L and z has batch dimensions *b, the resulting ansatz will have
            batch dimensions (*b, *L).
        """
        if len(indices) == self.num_CV_vars:
            raise ValueError(
                "The number of indices provided is the same as the number of CV variables."
                "Use the eval() or __call__() method instead."
            )

        z_batch_shape = z.shape[:-1]
        z_batch_size = np.prod(z_batch_shape)
        z = math.reshape(z, (z_batch_size, -1))

        # evaluated, remaining and derived indices
        e = indices
        r = [i for i in range(self.num_CV_vars) if i not in indices]
        d = list(range(self.num_CV_vars, self.num_vars))
        f = len(r) + len(d)  # leftover core dimensions (CV + derived)

        new_A = math.gather(math.gather(self._A_vectorized, r + d, axis=-1), r + d, axis=-2)
        new_A = math.repeat(new_A, z_batch_size, axis=0)  # can reshape to (*b, *L, r+d, r+d)
        new_A = math.reshape(new_A, z_batch_shape + self.batch_shape + (f, f))

        A_er = math.gather(
            math.gather(self._A_vectorized, e, axis=-2), r, axis=-1
        )  # shape (vec(L), e, r)
        b_r = math.einsum("ier,be->bir", A_er, z)  # shape (vec(b), vec(L), r)

        if len(d) > 0:
            A_ed = math.gather(
                math.gather(self._A_vectorized, e, axis=-2), d, axis=-1
            )  # shape (vec(L), e, d)
            b_d = math.einsum("ied,be->bid", A_ed, z)  # shape (vec(b), vec(L), d)
            new_b = math.gather(self._b_vectorized, r + d, axis=-1)[None, :, :] + math.concat(
                (b_r, b_d), axis=-1
            )
        else:
            new_b = math.gather(self._b_vectorized, r, axis=-1)[None, :, :] + b_r

        new_b = math.reshape(new_b, z_batch_shape + self.batch_shape + (f,))

        # new c of shape (batch_size * b,)
        A_ee = math.gather(
            math.gather(self._A_vectorized, e, axis=-1), e, axis=-2
        )  # shape (vec(L), e, e)
        A_part = math.einsum("be,bf,ief->bi", z, z, A_ee)  # shape (vec(b), vec(L))
        b_part = math.einsum(
            "be,ie->bi", z, math.gather(self._b_vectorized, e, axis=-1)
        )  # shape (vec(b), vec(L))
        exp_sum = math.exp(1 / 2 * A_part + b_part)  # shape (vec(b), vec(L))
        new_c = math.einsum("bi,i...->bi...", exp_sum, self.c)
        c_shape = (
            z_batch_shape + self.shape_derived_vars if self.num_derived_vars > 0 else z_batch_shape
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
        return np.array_equal(self.b, other.b) and np.array_equal(self.A, other.A)

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        display(widgets.bargmann(self))

    def __add__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Adds two PolyExp ansatze with same core dimensions together.
        This means concatenating them in the last batch dimension. In the case where ``c`` on self
        and other have different shapes it will add padding zeros to make the shapes fit.
        As a convoluted but compelte example: If ``c1`` has batch shape (4,4,1) and core shape
        (3,4,5) and ``c2`` has batch shape (4,4,10) and core shape (5,4,3) then the combined object
        will have batch shape (4,4,11) and core shape (5,4,5). Note we stack on the last batch
        dimension, and padded the other dimensions. If the batch shapes are incompatible it will
        raise an error.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot add PolyExpAnsatz and {other.__class__}.")
        if self.batch_shape[:-1] != other.batch_shape[:-1]:
            raise ValueError(
                f"Batch shapes must be stackable on the last dimension. Got {self.batch_shape} and {other.batch_shape}."
            )
        if self.num_CV_vars != other.num_CV_vars:
            raise ValueError(
                f"The number of CV variables must match. Got {self.num_CV_vars} and {other.num_CV_vars}."
            )

        def combine_arrays(array1, array2):
            """Combine arrays, padding with zeros if shapes differ in non-derived dimensions."""
            shape1 = array1.shape[self.batch_dims :]
            shape2 = array2.shape[self.batch_dims :]
            max_shapes = tuple(map(max, zip(shape1, shape2)))
            pad_widths1 = [(0, 0)] * (self.batch_dims) + [
                (0, t - s) for t, s in zip(max_shapes, shape1)
            ]
            pad_widths2 = [(0, 0)] * (self.batch_dims) + [
                (0, t - s) for t, s in zip(max_shapes, shape2)
            ]
            padded_array1 = math.pad(array1, pad_widths1, "constant")
            padded_array2 = math.pad(array2, pad_widths2, "constant")
            return math.concat([padded_array1, padded_array2], axis=self.batch_dims)

        n_derived_vars = max(self.num_derived_vars, other.num_derived_vars)

        combined_matrices = math.concat([self.A, other.A], axis=self.batch_dims)
        combined_vectors = math.concat([self.b, other.b], axis=self.batch_dims)
        combined_arrays = combine_arrays(
            math.atleast_nd(self.c, n_derived_vars),
            math.atleast_nd(other.c, n_derived_vars),
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
        As, bs, cs = join_Abc(self.triple, other.triple, self._outer_product_batch_str(other))
        return PolyExpAnsatz(As, bs, cs, self.num_derived_vars + other.num_derived_vars)

    def __call__(
        self, *z: Vector | None, batch_string: str | None = None
    ) -> Scalar | ArrayLike | PolyExpAnsatz:
        r"""
        Evaluates the ansatz at given points or returns a partially evaluated ansatz.
        This method supports passing an einsum-style batch string to specify how the batch dimensions
        of the arguments should be handled. For example, for two arguments, "i,j->ij" means to take
        the outer product  of the batch dimensions. The batch dimensions of the ansatz itself are not
        part of the batch string and are placed after the output batch dimensions in the output.

        1. Partial evaluation: If any argument is None, it returns a new ansatz with those arguments
           unevaluated. For example, if F(z1, z2, z3) is called as F(1.0, None, 3.0), it returns a
           new ansatz G(z2) with z1 and z3 fixed at 1.0 and 3.0.

        2. Full evaluation: If all arguments are provided, it returns the value of the ansatz at
        those points.

        The returned shape depends on the batch shape of the ansatz itself and the batch dimensions of
        the inputs and the batch string.

        Args:
            z: points in C where the function is (partially) evaluated or None if the variable is
            not evaluated.
            batch_string: like einsum string for batch dimensions of the inputs, e.g. "i,j->ij"

        Returns:
            The value of the ansatz or a new ansatz if partial evaluation is performed.
        """
        if len(z) > self.num_CV_vars:
            raise ValueError(
                f"The ansatz was called with {len(z)} variables, "
                f"but it only has {self.num_CV_vars} CV variables."
            )

        evaluated_indices = [i for i, zi in enumerate(z) if zi is not None]
        only_z = [math.atleast_1d(zi) for zi in z if zi is not None]

        if batch_string is None:  # Generate default batch string if none provided
            batch_string = self._outer_product_batch_str(*[len(zi.shape) - 1 for zi in only_z])

        if len(evaluated_indices) == self.num_CV_vars:  # Full evaluation
            reshaped_z = self._reshape_args_to_batch_string(only_z, batch_string)

            return self.eval(reshaped_z)
        else:  # Partial evaluation: some CV variables are not provided
            combined_z = self._combine_args_with_batch_string(only_z, batch_string)
            return self.partial_eval(combined_z, tuple(evaluated_indices))

    def __eq__(self, other: PolyExpAnsatz) -> bool:
        if not isinstance(other, PolyExpAnsatz):
            return False
        return self._equal_no_array(other) and np.array_equal(self.c, other.c)

    def __mul__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c * other, self.num_derived_vars)
            except Exception as e:
                raise TypeError(f"Cannot multiply PolyExpAnsatz and {other.__class__}.") from e

        else:
            raise NotImplementedError(
                "Multiplication of PolyExpAnsatz with other PolyExpAnsatz is not implemented."
            )

    def __neg__(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, -self.c, self.num_derived_vars)

    def __truediv__(self, other: Scalar | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c / other, self.num_derived_vars)
            except Exception as e:
                raise TypeError(f"Cannot divide PolyExpAnsatz and {other.__class__}.") from e
        else:
            raise NotImplementedError(
                "Division of PolyExpAnsatz with other PolyExpAnsatz is not implemented."
            )
