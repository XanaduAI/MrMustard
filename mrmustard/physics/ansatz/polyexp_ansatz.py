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

"""This module contains the PolyExp ansatz."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from IPython.display import display
from numpy.typing import ArrayLike

from mrmustard import math, settings, widgets
from mrmustard.physics.fock_utils import c_in_PS
from mrmustard.physics.utils import (
    batch_indexer_info,
    generate_batch_str,
    join_Abc,
    outer_product_batch_str,
    reshape_args_to_batch_string,
    verify_triple,
)
from mrmustard.utils.argsort import argsort_gen
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexScalar,
    ComplexTensor,
    ComplexVector,
    Matrix,
    Scalar,
    Tensor,
    Vector,
)

from .base import Ansatz

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""This class represents the ansatz of a polynomial exponential function.

    Namely,

        :math:`F^{(i)}(z) = \sum_k c^{(i)}_{k} \partial_y^k \textrm{exp}(\frac{1}{2}(z,y)^T A^{(i)} (z,y) + (z,y)^T b^{(i)})|_{y=0}`

    with ``k`` and ``i`` multi-indices. The ``i`` multi-index is a batch index of shape ``L`` that
    can be used for linear superposition or batching purposes. Each of the ``c^{(i)}_k`` tensors are
    contracted with the array of derivatives :math:`\partial_y^k` to form polynomials of derivatives.
    The tensors :math:`c^{(i)}_{k}` contain the coefficients of the polynomial of derivatives and
    have shape ``(*L, *derived)``, where ``*derived`` is the shape of the derived variables, which
    implies ``len(c.shape[1:]) = m``. The matrices :math:`A^{(i)}` and vectors :math:`b^{(i)}` are
    the parameters of the exponential terms in the ansatz, with :math:`z\in\mathbb{C}^{n}` and
    :math:`y\in\mathbb{C}^{m}`. ``A`` and ``b`` have shape ``(*L, n+m, n+m)`` and ``(*L, n+m)``,
    respectively.

    Args:
        A: A batch of quadratic coefficient :math:`A^{(i)}`.
        b: A batch of linear coefficients :math:`b^{(i)}`.
        c: A batch of arrays :math:`c^{(i)}`.
        name: The name of the ansatz.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    >>> from mrmustard.physics.ansatz import PolyExpAnsatz
    >>> import numpy as np
    >>> A = np.random.random((3,3))  # no batch
    >>> b = np.random.random((3,))
    >>> c = np.random.random()
    >>> F = PolyExpAnsatz(A, b, c)
    >>> assert F(1.0, 2.0, 3.0).shape == ()
    >>> A = np.random.random((10,3,3))  # batch of 10
    >>> b = np.random.random((10,3))
    >>> c = np.random.random((10,))
    >>> F = PolyExpAnsatz(A, b, c)
    >>> assert F(1.0, 2.0, 3.0).shape == (10,)
    >>> A = np.random.random((10,3,3))  # batch of 10
    >>> b = np.random.random((10,3))
    >>> c = np.random.random((10,7))
    >>> F = PolyExpAnsatz(A, b, c)
    >>> assert F(1.0, 2.0).shape == (10,)  # two CV variables, one derived
    >>> A = np.random.random((10,3,3))  # batch of 10
    >>> b = np.random.random((10,3))
    >>> c = np.random.random((10,7,5))
    >>> F = PolyExpAnsatz(A, b, c)
    >>> assert F(1.0).shape == (10,)  # one CV variable, two derived
    >>> assert F([1.0, 2.0, 3.0]).shape == (3,10)  # batch of 3 inputs
    """

    def __init__(
        self,
        A: Matrix,
        b: Vector,
        c: Tensor,
        name: str = "",
        lin_sup: bool = False,
    ):
        super().__init__()
        self.name = name
        self._simplified = False
        self._lin_sup = lin_sup

        self._A = math.astensor(A, dtype=math.complex128)
        self._b = math.astensor(b, dtype=math.complex128)
        self._c = math.astensor(c, dtype=math.complex128)

        verify_triple(self._A, self._b, self._c)

        self._batch_shape = tuple(self._A.shape[:-2])

    @property
    def A(self) -> ComplexMatrix:
        r"""The batch of quadratic coefficient :math:`A^{(i)}`."""
        return self._A

    @property
    def b(self) -> ComplexVector:
        r"""The batch of linear coefficients :math:`b^{(i)}`."""
        return self._b

    @property
    def batch_dims(self) -> int:
        return len(self.batch_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def batch_size(self) -> int:
        return math.prod(self.batch_shape) if self.batch_shape else 0

    @property
    def c(self) -> ComplexTensor:
        r"""The batch of polynomial coefficients :math:`c^{(i)}_{k}`."""
        return self._c

    @property
    def conj(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(
            math.conj(self.A),
            math.conj(self.b),
            math.conj(self.c),
            lin_sup=self._lin_sup,
        )

    @property
    def core_dims(self) -> int:
        r"""The number of core variables of the ansatz. Equivalent to ``self.num_CV_vars``."""
        return self.num_CV_vars

    @property
    def data(
        self,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
        r"""Returns the triple, which is necessary to reinstantiate the ansatz."""
        return self.triple

    @property
    def num_CV_vars(self) -> int:
        r"""The number of continuous variables that remain after the polynomial of derivatives is applied.
        This is the number of continuous variables of the Ansatz function itself.
        """
        return self.num_vars - self.num_derived_vars

    @property
    def num_derived_vars(self) -> int:
        r"""The number of derived variables that are derived by the polynomial of derivatives."""
        return len(self.shape_derived_vars)

    @property
    def num_vars(self):
        return self.A.shape[-1]

    @property
    def PS(self) -> PolyExpAnsatz:
        r"""The ansatz defined using real (i.e., phase-space) variables."""
        n = self.A.shape[-1]
        if n % 2:
            raise ValueError(
                f"A phase space ansatz must have even number of indices. (n={n} is odd)",
            )

        if self.num_derived_vars == 0:
            W = math.conj(math.rotmat(n // 2)) / math.sqrt(settings.HBAR, dtype=math.complex128)

            A = math.einsum("ji,...jk,kl->...il", W, self.A, W)
            b = math.einsum("ij,...j->...i", W, self.b)
            c = self.c / (2 * settings.HBAR) ** (n // 2)
            return PolyExpAnsatz(A, b, c, lin_sup=self._lin_sup)

        if self.num_derived_vars != 2:
            raise ValueError("This transformation supports 2 core and 0 or 2 derived variables")
        A_tmp = self.A

        A_tmp = A_tmp[..., [0, 2, 1, 3], :][..., [0, 2, 1, 3]]
        b = self.b[..., [0, 2, 1, 3]]
        c = c_in_PS(self.c)  # implements PS transformations on ``c``

        W = math.conj(math.rotmat(n // 2)) / math.sqrt(settings.HBAR, dtype=math.complex128)

        A = math.einsum("ji,...jk,kl->...il", W, A_tmp, W)
        b = math.einsum("ij,...j->...i", W, b)
        c = c / (2 * settings.HBAR)

        A_out = A[..., [0, 2, 1, 3], :][..., :, [0, 2, 1, 3]]
        b_out = b[..., [0, 2, 1, 3]]

        return PolyExpAnsatz(A_out, b_out, c, lin_sup=self._lin_sup)

    @property
    def scalar(self) -> Scalar:
        r"""The scalar part of the ansatz, i.e. F(0)."""
        if self.num_CV_vars == 0 and self.num_derived_vars == 0:
            ret = math.einsum("...a->...", self.c) if self._lin_sup else self.c
        elif self.num_CV_vars == 0:
            ret = self()
        else:
            ret = self(*math.zeros(self.num_CV_vars))
        return ret

    @property
    def shape_derived_vars(self) -> tuple[int, ...]:
        r"""The shape of the coefficients of the polynomial of derivatives."""
        return tuple(self.c.shape[self.batch_dims :])

    @property
    def triple(
        self,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
        r"""Returns the triple of parameters of the exponential part of the ansatz."""
        return self.A, self.b, self.c

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> PolyExpAnsatz:
        r"""Creates an ansatz from a dictionary. For deserialization purposes."""
        return cls(**data)

    def concat(self, other: Ansatz, axis: int = 0) -> PolyExpAnsatz:
        r"""Concatenates two PolyExpAnsatz objects along the specified batch axis.

        All batch axes except the concatenation axis must agree in size.
        The lin_sup status must match. Core dimensions (num_CV_vars and num_derived_vars)
        must also match.

        Args:
            other: The other PolyExpAnsatz to concatenate with.
            axis: The batch axis along which to concatenate (default 0).

        Returns:
            A new PolyExpAnsatz with the concatenated batch dimensions.

        Raises:
            TypeError: If ``other`` is not a ``PolyExpAnsatz``.
            ValueError: If batch dimensions don't match (except at axis), if lin_sup status
                doesn't match, or if core dimensions don't match.

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz
        >>> import numpy as np
        >>> A1 = np.random.random((2, 3, 3))
        >>> b1 = np.random.random((2, 3))
        >>> c1 = np.random.random((2,))
        >>> ansatz1 = PolyExpAnsatz(A1, b1, c1)
        >>> A2 = np.random.random((3, 3, 3))
        >>> b2 = np.random.random((3, 3))
        >>> c2 = np.random.random((3,))
        >>> ansatz2 = PolyExpAnsatz(A2, b2, c2)
        >>> concatenated = ansatz1.concat(ansatz2, axis=0)
        >>> assert concatenated.batch_shape == (5,)
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot concatenate with ansatz of type {type(other)}!")

        if self._lin_sup != other._lin_sup:
            raise ValueError(
                f"Cannot concat ansatze with different lin_sup status: {self._lin_sup} vs {other._lin_sup}"
            )

        if self.num_CV_vars != other.num_CV_vars:
            raise ValueError(
                f"Cannot concat ansatze with different num_CV_vars: {self.num_CV_vars} vs {other.num_CV_vars}"
            )

        if self.num_derived_vars != other.num_derived_vars:
            raise ValueError(
                f"Cannot concat ansatze with different num_derived_vars: {self.num_derived_vars} vs {other.num_derived_vars}"
            )

        # Normalize negative axis
        if axis < 0:
            axis = self.batch_dims + axis

        if axis < 0 or axis >= self.batch_dims:
            raise ValueError(f"axis {axis} is out of range for batch_dims {self.batch_dims}")

        # Check that all other batch axes match
        for i in range(self.batch_dims):
            if i != axis and self.batch_shape[i] != other.batch_shape[i]:
                raise ValueError(
                    f"Batch shapes must match except at axis {axis}. "
                    f"Got {self.batch_shape} and {other.batch_shape}"
                )

        # Concatenate the triple components along the batch axis
        A_concat = math.concat([self.A, other.A], axis=axis)
        b_concat = math.concat([self.b, other.b], axis=axis)
        c_concat = math.concat([self.c, other.c], axis=axis)

        return PolyExpAnsatz(A_concat, b_concat, c_concat, lin_sup=self._lin_sup)

    def contract(
        self,
        other: Ansatz,
        idxs: tuple[Sequence[int], Sequence[int]],
    ) -> PolyExpAnsatz:
        r"""Contract along specified core (CV) axes, broadcasting batch dimensions
        and taking the kronecker product of linear-superposition dimensions.

        Note:
            - Batch dims broadcast automatically (no batch string needed).
            - Negative indices are normalized relative to each operand's CV count.

        Args:
            other: The other PolyExpAnsatz to contract with.
            idxs: Tuple ``(idx_self, idx_other)`` of CV-axis indices to integrate over
                (0-based relative to the CV variables). The two sequences must have
                the same length. Negative indices allowed.

        Returns:
            The contracted PolyExpAnsatz with kept core axes ordered as
            ``[self non-contracted] + [other non-contracted]`` and derived axes as
            ``[self derived] + [other derived]``.

        Raises:
            TypeError: If ``other`` is not a ``PolyExpAnsatz``.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot contract with ansatz of type {type(other)}!")

        # Validate indices
        idx1, idx2 = idxs
        n1, n2 = self.num_CV_vars, other.num_CV_vars
        m1, m2 = self.num_derived_vars, other.num_derived_vars
        if len(idx1) != len(idx2):
            raise ValueError(
                f"idxs must have sequences of equal length, got {len(idx1)} and {len(idx2)}."
            )

        if self._lin_sup and other._lin_sup:  # kron lin sup axes
            A1 = math.expand_dims(self.A, axis=-3)
            b1 = math.expand_dims(self.b, axis=-2)
            c1 = math.expand_dims(self.c, axis=-m1 - 1)
            A2 = math.expand_dims(other.A, axis=-4)
            b2 = math.expand_dims(other.b, axis=-3)
            c2 = math.expand_dims(other.c, axis=-m2 - 2)
        elif self._lin_sup or other._lin_sup:  # zip lin sup axes
            A1, b1, c1 = self._ensure_lin_sup_axis().triple
            A2, b2, c2 = other._ensure_lin_sup_axis().triple
        else:
            A1 = self.A
            b1 = self.b
            c1 = self.c
            A2 = other.A
            b2 = other.b
            c2 = other.c
        A_post, b_post, log_c_factor = math.complex_gaussian_integral_2(A1, b1, A2, b2, idx1, idx2)

        # Reorder core indices to [self CV remaining, other CV remaining, self derived, other derived]
        k = len(idx1)
        s1, s2, s3, s4 = n1 - k, m1, n2 - k, m2
        order = (
            list(range(s1))
            + list(range(s1 + s2, s1 + s2 + s3))
            + list(range(s1, s1 + s2))
            + list(range(s1 + s2 + s3, s1 + s2 + s3 + s4))
        )
        A_post = math.gather(math.gather(A_post, order, axis=-1), order, axis=-2)
        b_post = math.gather(b_post, order, axis=-1)

        # Combine polynomial parts: outer product of c's times the scalar (batched) c_factor
        if m1 or m2:
            poly1 = "".join(chr(97 + i) for i in range(m1))
            poly2 = "".join(chr(97 + m1 + j) for j in range(m2))
            log_c12 = math.log(math.einsum(f"...{poly1},...{poly2}->...{poly1}{poly2}", c1, c2))
            c_out = math.exp(log_c_factor[..., *((None,) * (m1 + m2))] + log_c12)
        else:
            c_out = math.exp(log_c_factor + math.log(c1) + math.log(c2))

        # If both have linear-superposition dimensions, collapse them into one
        if self._lin_sup and other._lin_sup:
            batch_shape = A_post.shape[:-2]
            # collapse the last two batch axes
            new_bs = (*batch_shape[:-2], batch_shape[-2] * batch_shape[-1])
            A_post = math.reshape(A_post, new_bs + A_post.shape[-2:])
            b_post = math.reshape(b_post, new_bs + b_post.shape[-1:])
            c_out = math.reshape(c_out, new_bs + c_out.shape[len(batch_shape) :])

        return PolyExpAnsatz(A_post, b_post, c_out, lin_sup=self._lin_sup or other._lin_sup)

    def decompose_ansatz(self) -> PolyExpAnsatz:
        r"""This method decomposes a PolyExp ansatz to make it more efficient to evaluate.
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
        n = self.num_CV_vars
        A, b, c = self.triple
        pulled_out_input_shape = (
            int(math.sum(self.shape_derived_vars)),
        ) * n  # cast to int for jax
        poly_shape = pulled_out_input_shape + self.shape_derived_vars

        batch_shape = A.shape[:-2]
        A_core = math.block(
            [
                [math.zeros((*batch_shape, n, n), dtype=A.dtype), A[..., :n, n:]],
                [A[..., n:, :n], A[..., n:, n:]],
            ],
        )
        b_core = math.concat((math.zeros((*batch_shape, n), dtype=b.dtype), b[..., n:]), axis=-1)

        poly_core = math.hermite_renormalized(
            A_core,
            b_core,
            math.ones(self.batch_shape, dtype=math.complex128),
            shape=poly_shape,
        )

        derived_vars_size = int(math.prod(self.shape_derived_vars))
        poly_core = math.reshape(
            poly_core,
            batch_shape + pulled_out_input_shape + (derived_vars_size,),
        )
        batch_str = generate_batch_str(len(batch_shape))
        c_prime = math.einsum(
            f"{batch_str}...k,{batch_str}...k->{batch_str}...",
            poly_core,
            c.reshape((*batch_shape, derived_vars_size)),
        )
        block = A[..., :n, :n]
        I_matrix = math.broadcast_to(math.eye_like(block), block.shape)
        A_decomp = math.block([[block, I_matrix], [I_matrix, math.zeros_like(block)]])
        b_decomp = math.concat((b[..., :n], math.zeros((*batch_shape, n), dtype=b.dtype)), axis=-1)
        return PolyExpAnsatz(A_decomp, b_decomp, c_prime, lin_sup=self._lin_sup)

    def display(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(repr(self))
            return
        display(widgets.bargmann(self))

    def eval(
        self,
        *z: Vector | None,
        batch_string: str | None = None,
    ) -> Scalar | ArrayLike | PolyExpAnsatz:
        r"""Evaluates the ansatz at given points or returns a partially evaluated ansatz.
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
                f"but it only has {self.num_CV_vars} CV variables.",
            )

        evaluated_indices = [i for i, zi in enumerate(z) if zi is not None]
        only_z = [math.astensor(zi) for zi in z if zi is not None]

        if batch_string is None:  # Generate default batch string if none provided
            batch_string = outer_product_batch_str(*[zi.ndim for zi in only_z])

        reshaped_z = reshape_args_to_batch_string(only_z, batch_string)
        broadcasted_z = math.broadcast_arrays(*reshaped_z)
        if len(evaluated_indices) == self.num_CV_vars:  # Full evaluation: all CV vars specified
            return self(*broadcasted_z)
        # Partial evaluation: some CV variables are not provided
        combined_z = math.stack(broadcasted_z, axis=-1)
        return self._partial_eval(combined_z, tuple(evaluated_indices))

    def reorder(self, order: Sequence[int]) -> PolyExpAnsatz:
        r"""Reorders the CV indices of an (A,b,c) triple.
        The length of ``order`` must equal the number of CV variables.
        This method returns a new ansatz object.
        """
        if len(order) != self.num_CV_vars:
            raise ValueError(f"order must have length {self.num_CV_vars}, got {len(order)}")
        # Add derived variable indices after CV indices
        order = list(order) + list(
            range(self.num_CV_vars, self.num_CV_vars + self.num_derived_vars),
        )
        A = math.gather(math.gather(self.A, order, axis=-1), order, axis=-2)
        b = math.gather(self.b, order, axis=-1)
        return PolyExpAnsatz(A, b, self.c, lin_sup=self._lin_sup)

    def reorder_batch(
        self, order: Sequence[int]
    ) -> PolyExpAnsatz:  # TODO: omit last batch index if lin_sup
        if len(order) != self.batch_dims:
            raise ValueError(
                f"order must have length {self.batch_dims} (number of batch dimensions), got {len(order)}",
            )

        core_dims_indices_A = range(self.batch_dims, self.batch_dims + 2)
        core_dims_indices_b = range(self.batch_dims, self.batch_dims + 1)
        core_dims_indices_c = range(self.batch_dims, self.batch_dims + self.num_derived_vars)

        new_A = math.transpose(self.A, list(order) + list(core_dims_indices_A))
        new_b = math.transpose(self.b, list(order) + list(core_dims_indices_b))
        new_c = math.transpose(self.c, list(order) + list(core_dims_indices_c))

        return PolyExpAnsatz(new_A, new_b, new_c, lin_sup=self._lin_sup)

    # TODO: this should be moved to classes responsible for interpreting a batch dimension as a sum
    def simplify(self) -> PolyExpAnsatz:
        r"""Returns a new ansatz simplified by combining terms that have the
        same exponential part, i.e. two components of the batch are considered equal if their
        ``A`` and ``b`` are equal. In this case only one is kept and the corresponding ``c`` are added.

        Will return immediately if the ansatz has already been simplified, so it is safe to re-call.

        Raises:
            NotImplementedError: If the ``PolyExpAnsatz`` is batched (w/o linear superposition).
        """
        if self._simplified or not self._lin_sup:
            return self
        batch_shape = self.batch_shape[:-1] if self._lin_sup else self.batch_shape
        if batch_shape:
            raise NotImplementedError("Batched simplify is not implemented.")
        (A, b, c), to_keep = self._find_unique_terms_sorted()

        A = math.gather(A, to_keep, axis=0)
        b = math.gather(b, to_keep, axis=0)
        c = math.gather(c, to_keep, axis=0)  # already added

        A = math.reshape(A, (len(to_keep), self.num_vars, self.num_vars))
        b = math.reshape(b, (len(to_keep), self.num_vars))
        c = math.reshape(c, (len(to_keep), *self.shape_derived_vars))

        new_ansatz = PolyExpAnsatz(A, b, c, lin_sup=self._lin_sup)
        new_ansatz._simplified = True

        return new_ansatz

    def to_dict(self) -> dict[str, ArrayLike]:
        r"""Returns a dictionary representation of the ansatz. For serialization purposes."""
        return {"A": self.A, "b": self.b, "c": self.c}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...], measure: float = -1.0):
        r"""Computes the trace of the ansatz across the specified pairs of CV variables.

        Args:
            idx_z: The indices indicating which CV variables to integrate over.
            idx_zconj: The indices indicating which conjugate CV variables to integrate over.
            measure: The measure to use in the complex Gaussian integral.

        Returns:
            A new ansatz with the specified indices traced out.
        """
        if len(idx_z) != len(idx_zconj):
            raise ValueError("idx_z and idx_zconj must have the same length.")
        if len(set(idx_z + idx_zconj)) != len(idx_z) + len(idx_zconj):
            raise ValueError(
                f"Indices must be unique: {set(idx_z).intersection(idx_zconj)} are repeated.",
            )
        if any(i >= self.num_CV_vars for i in idx_z) or any(
            i >= self.num_CV_vars for i in idx_zconj
        ):
            raise ValueError(
                f"All indices must be between 0 and {self.num_CV_vars - 1}. Got {idx_z} and {idx_zconj}.",
            )
        A_in, b_in, c_in = self.triple
        log_c_in = math.log(math.cast(c_in, "complex128"))
        A, b, log_c = math.complex_gaussian_integral_1(A_in, b_in, idx_z + idx_zconj)
        # broadcast log_c and log_c_in to the same shape
        if log_c_in.shape != log_c.shape:
            log_c = math.reshape(log_c, log_c.shape + (1,) * (log_c_in.ndim - log_c.ndim))
        return PolyExpAnsatz(A, b, math.exp(log_c_in + log_c), lin_sup=self._lin_sup)

    def _combine_exp_and_poly(
        self,
        exp_sum: ComplexTensor,
        poly: ComplexTensor,
        c: ComplexTensor,
    ) -> ComplexTensor:
        r"""Combines exponential and polynomial parts using einsum. Needed in ``__call__``."""
        poly_string = "".join(chr(i) for i in range(97, 97 + len(self.shape_derived_vars)))
        return math.einsum(f"...,...{poly_string},...{poly_string}->...", exp_sum, c, poly)

    def _compute_exp_part(
        self,
        z: Vector,
        A: ComplexMatrix,
        b: ComplexVector,
        log_c: ComplexScalar,
    ) -> Scalar:
        r"""Computes the exponential part of the ansatz evaluation. Needed in ``__call__``.
        The exponential part is given by:
        .. math::
            \exp\left(\frac{1}{2} z^T A z + b^T z + \log c\right)
        where :math:`A` is the matrix of the quadratic part of the ansatz, :math:`b` is the vector
        of the linear part that correspond to the vector of given CV variables, and :math:`\log c`
        is a scalar folded into the exponent to prevent overflow when :math:`c` is very small and
        the quadratic/linear terms are very large.
        """
        n = self.num_CV_vars
        A_part = math.einsum("...a,...b,...ab->...", z, z, A[..., :n, :n])
        b_part = math.einsum("...a,...a->...", z, b[..., :n])
        return math.exp(1 / 2 * A_part + b_part + log_c)

    def _compute_polynomial_part(
        self,
        z: Vector,
        A: ComplexMatrix,
        b: ComplexVector,
    ) -> Scalar:
        r"""Computes the polynomial part of the ansatz evaluation. Needed in ``__call__``."""
        n = self.num_CV_vars
        batch_shape = z.shape[:-1]
        b_poly = math.einsum("...ab,...a->...b", A[..., :n, n:], z) + b[..., n:]
        return math.hermite_renormalized(
            A[..., n:, n:],
            b_poly,
            math.ones(batch_shape, dtype=math.complex128),
            shape=self.shape_derived_vars,
        )

    def _ensure_lin_sup_axis(self) -> PolyExpAnsatz:
        if self._lin_sup:
            return PolyExpAnsatz(self._A, self._b, self._c, lin_sup=True)
        return PolyExpAnsatz(
            math.expand_dims(self._A, axis=-3),
            math.expand_dims(self._b, axis=-2),
            math.expand_dims(self._c, axis=-self.num_derived_vars - 1),
            lin_sup=True,
        )

    def _find_unique_terms_sorted(
        self,
    ) -> tuple[tuple[ComplexMatrix, ComplexVector, ComplexTensor], list[int]]:
        r"""Finds unique terms by first sorting the batch dimension and adds the corresponding c values.

        Returns:
            The updated vectorized (A,b,c) triple and a list of indices to keep after simplification.
        """
        A, b, c = self._order_batch()
        to_keep = [d0 := 0]
        mat, vec = A[d0], b[d0]

        for d in range(1, self.batch_size):
            if not (
                math.allclose(mat, A[d], atol=settings.ATOL)
                and math.allclose(vec, b[d], atol=settings.ATOL)
            ):
                to_keep.append(d)
                d0 = d
                mat, vec = A[d0], b[d0]
            else:
                d0r = np.unravel_index(d0, self.batch_shape)
                dr = np.unravel_index(d, self.batch_shape)
                c = math.update_add_tensor(c, [d0r], [c[dr]])
        return (A, b, c), to_keep

    def _order_batch(
        self,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
        r"""This method orders the batch dimension by the lexicographical order of the
        flattened arrays (A, b, c). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two PolyExp ansatz.

        Returns:
            The ordered vectorized (A, b, c) triple.
        """
        if not self.batch_shape:
            return self.A, self.b, self.c
        A_vectorized = math.reshape(self.A, (self.batch_size, self.num_vars, self.num_vars))
        b_vectorized = math.reshape(self.b, (self.batch_size, self.num_vars))
        c_vectorized = math.reshape(self.c, (self.batch_size, *self.shape_derived_vars))
        generators = [
            itertools.chain(
                math.asnumpy(b_vectorized[i]).flat,
                math.asnumpy(A_vectorized[i]).flat,
                math.asnumpy(c_vectorized[i]).flat,
            )
            for i in range(self.batch_size)
        ]
        sorted_indices = argsort_gen(generators)
        A = math.gather(A_vectorized, sorted_indices, axis=0)
        b = math.gather(b_vectorized, sorted_indices, axis=0)
        c = math.gather(c_vectorized, sorted_indices, axis=0)
        return A, b, c

    def _partial_eval(self, z: ArrayLike, indices: tuple[int, ...]) -> PolyExpAnsatz:
        r"""Partially evaluates the ansatz by fixing some of its variables to specific values.

        This method creates a new ansatz with fewer variables by substituting the specified
        variables with their given values. The remaining variables keep their original order
        in the function signature.

        Example:
            If this ansatz represents F(z0,z1,z2), then:
            ```
            new_ansatz = self._partial_eval([2.0,3.0], indices=(0,2))
            ```
            returns a new ansatz G(z1) equal to F(2.0, z1, 3.0).

        Args:
            z: Values for the variables being fixed. Can be:
               - Shape (r,): A single vector of values for r variables
               - Shape (*b, r): Batch of r values of shape *b
               Where r is the number of indices in `indices`
            indices: Indices of the variables to be fixed to the values in z

        Returns:
            A new PolyExpAnsatz with fewer variables. If the original ansatz has batch
            dimensions *L and z has batch dimensions *b, the resulting ansatz will have
            batch dimensions (*b, *L).
        """
        if len(indices) >= self.num_CV_vars:
            raise ValueError(
                "The number of variables and indices must not exceed the number of CV "
                f"variables {self.num_CV_vars}. Use the eval() or __call__() method instead.",
            )
        z_batch_shape = z.shape[:-1]

        # evaluated, remaining and derived indices
        e = indices
        r = [i for i in range(self.num_CV_vars) if i not in indices]
        d = list(range(self.num_CV_vars, self.num_vars))

        ansatz_batch_idxs = tuple(range(self.batch_dims))
        z_batch_idxs = tuple(range(self.batch_dims, self.batch_dims + len(z_batch_shape)))
        z = math.transpose(
            math.broadcast_to(z, self.batch_shape + z.shape),
            z_batch_idxs + ansatz_batch_idxs + (len(z_batch_idxs + ansatz_batch_idxs),),
        )

        A = math.broadcast_to(self.A, z_batch_shape + self.A.shape)
        b = math.broadcast_to(self.b, z_batch_shape + self.b.shape)
        c = math.broadcast_to(self.c, z_batch_shape + self.c.shape)

        new_A = math.gather(math.gather(A, r + d, axis=-1), r + d, axis=-2)
        A_er = math.gather(math.gather(A, e, axis=-2), r, axis=-1)
        b_r = math.einsum("...er,...e->...r", A_er, z)

        if len(d) > 0:
            A_ed = math.gather(math.gather(A, e, axis=-2), d, axis=-1)
            b_d = math.einsum("...ed,...e->...d", A_ed, z)
            new_b = math.gather(b, r + d, axis=-1) + math.concat((b_r, b_d), axis=-1)
        else:
            new_b = math.gather(b, r, axis=-1) + b_r
        A_ee = math.gather(math.gather(A, e, axis=-2), e, axis=-1)
        A_part = math.einsum("...e,...f,...ef->...", z, z, A_ee)
        b_part = math.einsum("...e,...e->...", z, math.gather(b, e, axis=-1))
        exp_sum = math.exp(1 / 2 * A_part + b_part)
        poly_string = "".join(chr(i) for i in range(97, 97 + len(self.shape_derived_vars)))
        new_c2 = math.einsum(f"...,...{poly_string}->...{poly_string}", exp_sum, c)
        return PolyExpAnsatz(
            new_A,
            new_b,
            new_c2,
            lin_sup=self._lin_sup,
        )

    def _squeeze_lin_sup_axis(self) -> PolyExpAnsatz:
        if not self._lin_sup:
            return PolyExpAnsatz(self._A, self._b, self._c, lin_sup=False)
        if self._A.shape[-3] == 1:
            return PolyExpAnsatz(
                math.squeeze(self._A, axis=-3),
                math.squeeze(self._b, axis=-2),
                math.squeeze(self._c, axis=-self.num_derived_vars - 1),
                lin_sup=False,
            )
        raise ValueError("Cannot squeeze lin sup axis if it is not of length 1")

    def __add__(self, other: Ansatz) -> PolyExpAnsatz:
        r"""Adds two ``PolyExpAnsatz`` together.

        This is equivalent to stacking their respective triples
        along a batch dimension, which is to be interpreted to mean a linear superposition.
        In order to use the ``__add__`` method, the ansatze must have the same number of CV variables,
        and zero or one batch dimensions. The reason for this restriction on the number of batch
        dimensions is that if there are multiple batch dimensions, it is not clear which one is used
        as meaning "linear superposition". In that case, the stacking of the triples should be done
        by the user.

        Args:
            other: The other ansatz to add.

        Raises:
            TypeError: If ``other`` is not a ``PolyExpAnsatz``.
            ValueError: If the number of CV variables doesn't match.
            ValueError: If the batch dimensions are incompatible.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot add ansatz of type {type(other)}!")

        if self.num_CV_vars != other.num_CV_vars:
            raise ValueError(
                f"The number of CV variables must match. Got {self.num_CV_vars} and {other.num_CV_vars}.",
            )
        if (self.batch_shape and not self._lin_sup) or (other.batch_shape and not other._lin_sup):
            raise ValueError(
                f"Cannot add PolyExpAnsatz with batch dimensions {self.batch_shape} and {other.batch_shape}.",
            )
        A_self = self.A if self.batch_dims == 1 else math.expand_dims(self.A, axis=0)
        b_self = self.b if self.batch_dims == 1 else math.expand_dims(self.b, axis=0)
        c_self = self.c if self.batch_dims == 1 else math.expand_dims(self.c, axis=0)
        A_other = other.A if other.batch_dims == 1 else math.expand_dims(other.A, axis=0)
        b_other = other.b if other.batch_dims == 1 else math.expand_dims(other.b, axis=0)
        c_other = other.c if other.batch_dims == 1 else math.expand_dims(other.c, axis=0)

        def pad_arrays(array1, array2):
            shape1 = array1.shape[1:]
            shape2 = array2.shape[1:]
            max_shapes = tuple(map(max, zip(shape1, shape2)))
            pad_widths1 = [(0, 0)] + [(0, m - s) for m, s in zip(max_shapes, shape1)]
            pad_widths2 = [(0, 0)] + [(0, m - s) for m, s in zip(max_shapes, shape2)]
            padded_array1 = math.pad(array1, pad_widths1, "constant")
            padded_array2 = math.pad(array2, pad_widths2, "constant")
            return padded_array1, padded_array2

        def pad_and_combine_arrays(array1, array2):
            padded_array1, padded_array2 = pad_arrays(array1, array2)
            return math.concat([padded_array1, padded_array2], axis=0)

        def pad_and_combine_Ab(Ab1, Ab2):
            padded_Ab1, padded_Ab2 = pad_arrays(Ab1, Ab2)
            return math.concat([padded_Ab1, padded_Ab2], axis=0)

        n_derived_vars = max(self.num_derived_vars, other.num_derived_vars)

        combined_matrices = pad_and_combine_Ab(A_self, A_other)
        combined_vectors = pad_and_combine_Ab(b_self, b_other)
        combined_arrays = pad_and_combine_arrays(
            math.atleast_nd(c_self, n_derived_vars + 1),
            math.atleast_nd(c_other, n_derived_vars + 1),
        )
        return PolyExpAnsatz(
            combined_matrices,
            combined_vectors,
            combined_arrays,
            lin_sup=True,
        )

    def __and__(self, other: Ansatz) -> PolyExpAnsatz:
        r"""Tensor product of this PolyExpAnsatz with another. Equivalent to :math:`H(a,b) = F(a) * G(b)`.
        As it distributes over addition on both self and other, the batch shape of the result is the
        outer product of the batch shapes of this ansatz and the other one. Use with moderation.

        Args:
            other: Another PolyExpAnsatz.

        Returns:
            The tensor product of this PolyExpAnsatz and other.

        Raises:
            TypeError: If ``other`` is not a ``PolyExpAnsatz``.
        """
        if not isinstance(other, PolyExpAnsatz):
            raise TypeError(f"Cannot tensor product ansatz of type {type(other)}!")

        A1, b1, c1 = self.triple
        A2, b2, c2 = other.triple

        # Split batch dimensions into regular and linear superposition
        num_reg1 = self.batch_dims - (1 if self._lin_sup else 0)
        num_reg2 = other.batch_dims - (1 if other._lin_sup else 0)

        # For outer product: (r1, [l1]) & (r2, [l2]) -> (r1, r2, l1, l2)
        # We insert singleton dims at specific positions to enable broadcasting

        # Expand A1/b1/c1: insert num_reg2 singletons after regular batch dims
        # If other has lin_sup, also insert singleton after self's lin_sup (if any)
        for _ in range(num_reg2):
            A1 = math.expand_dims(A1, axis=num_reg1)
            b1 = math.expand_dims(b1, axis=num_reg1)
            c1 = math.expand_dims(c1, axis=num_reg1)
        if other._lin_sup:
            # Insert after: reg1 + inserted_reg2 + [linsup1 if exists]
            pos = num_reg1 + num_reg2 + (1 if self._lin_sup else 0)
            A1 = math.expand_dims(A1, axis=pos)
            b1 = math.expand_dims(b1, axis=pos)
            c1 = math.expand_dims(c1, axis=pos)

        # Expand A2/b2/c2: insert num_reg1 singletons at the beginning
        # If self has lin_sup, also insert singleton after those leading ones
        for _ in range(num_reg1):
            A2 = math.expand_dims(A2, axis=0)
            b2 = math.expand_dims(b2, axis=0)
            c2 = math.expand_dims(c2, axis=0)
        if self._lin_sup:
            # Insert after the leading ones, before other's regular batch + lin_sup
            A2 = math.expand_dims(A2, axis=num_reg1 + num_reg2)
            b2 = math.expand_dims(b2, axis=num_reg1 + num_reg2)
            c2 = math.expand_dims(c2, axis=num_reg1 + num_reg2)

        # Join the triples (broadcasting will handle the outer product)
        As, bs, cs = join_Abc(A1, b1, c1, A2, b2, c2)

        # If both have lin_sup, merge the two lin_sup dimensions into one
        if self._lin_sup and other._lin_sup:
            total_reg = num_reg1 + num_reg2
            # Shape is (...reg_batch, lin1, lin2, ...) -> (...reg_batch, lin1*lin2, ...)
            As = math.reshape(As, (*As.shape[:total_reg], -1, *As.shape[total_reg + 2 :]))
            bs = math.reshape(bs, (*bs.shape[:total_reg], -1, *bs.shape[total_reg + 2 :]))
            cs = math.reshape(cs, (*cs.shape[:total_reg], -1))

        return PolyExpAnsatz(As, bs, cs, lin_sup=self._lin_sup or other._lin_sup)

    def __call__(self, *z_inputs: ArrayLike | None) -> ComplexTensor:
        r"""Evaluates the ansatz at the given batch of points. Each point can have arbitray batch dimensions,
        as long as they are broadcastable. If some of the points are not specified (None), the result
        will be a partially evaluated ansatz.
        If the combined shape of the inputs is ``(*b, n)`` where ``n`` is the number of CV variables in the ansatz
        and ``*b`` is the batch dimensions of the combined inputs, then the output will have shape ``(*b, *L)``
        where ``*L`` is the batch shape of the ansatz itself.

        Args:
            *z_inputs: A batch of points where the function is evaluated (or None).
                The shape of each point can be arbitrary, as long as they are broadcastable.

        Returns:
            The evaluated function with shape ``(*b, *L)`` where:
               - ``*b`` are the batch dimensions of the combined inputs.
               - ``*L`` is the batch shape of the ansatz.

        Raises:
            ValueError: If the number of CV variables is not equal to the number of input points.
        """
        z_only = [math.cast(arr, dtype=math.complex128) for arr in z_inputs if arr is not None]
        broadcasted_z = math.broadcast_arrays(*z_only)
        z = (
            math.stack(broadcasted_z, axis=-1)
            if broadcasted_z
            else math.astensor([], dtype=math.complex128)
        )
        if len(z_only) < self.num_CV_vars:
            indices = tuple(i for i, arr in enumerate(z_inputs) if arr is not None)
            return self._partial_eval(z, indices)
        z_batch_shape, z_dim = z.shape[:-1], z.shape[-1]
        if z_dim != self.num_CV_vars:
            raise ValueError(
                f"The last dimension of `z` must equal the number of CV variables {self.num_CV_vars}, got {z_dim}.",
            )

        ansatz_batch_idxs = tuple(range(self.batch_dims))
        z_batch_idxs = tuple(range(self.batch_dims, self.batch_dims + len(z_batch_shape)))
        z = math.transpose(
            math.broadcast_to(z, self.batch_shape + z.shape),
            z_batch_idxs + ansatz_batch_idxs + (len(z_batch_idxs + ansatz_batch_idxs),),
        )

        A = math.broadcast_to(self.A, z_batch_shape + self.A.shape)
        b = math.broadcast_to(self.b, z_batch_shape + self.b.shape)
        c = math.broadcast_to(self.c, z_batch_shape + self.c.shape)

        if self.num_derived_vars == 0:  # purely gaussian
            # Fold log(c) into the exponent to prevent overflow:
            # c * exp(exponent) = exp(exponent + log(c))
            log_c = math.log(c)
            exp_sum = self._compute_exp_part(z, A, b, log_c)
            ret = exp_sum
        else:
            # For the polynomial case, extract a global scale to stabilise the exp:
            # c * exp(exponent) = (c / c_scale) * exp(exponent + log(c_scale))
            # Flatten derived-var dims and take a global max of |c| as scale.
            c_flat = math.reshape(math.abs(c), (*c.shape[: -self.num_derived_vars], -1))
            # c_flat has shape (...batch, prod_derived); take max over last axis
            c_scale = math.max(c_flat)  # global scalar scale
            c_scale = math.maximum(c_scale, 1e-300)
            log_c_scale = math.log(c_scale)
            exp_sum = self._compute_exp_part(z, A, b, log_c_scale)
            c_normalized = c / c_scale
            poly = self._compute_polynomial_part(z, A, b)
            ret = self._combine_exp_and_poly(exp_sum, poly, c_normalized)

        return math.sum(ret, axis=-1) if self._lin_sup else ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyExpAnsatz):
            return False

        if self.num_CV_vars != other.num_CV_vars or self.num_derived_vars != other.num_derived_vars:
            return False

        self_A, self_b, self_c = self._order_batch()
        other_A, other_b, other_c = other._order_batch()
        return (
            math.allclose(self_A, other_A, atol=settings.ATOL)
            and math.allclose(self_b, other_b, atol=settings.ATOL)
            and math.allclose(self_c, other_c, atol=settings.ATOL)
        )

    def __getitem__(self, indexer: Any) -> PolyExpAnsatz:
        r"""Batch-only indexing. Supports integers, slices, None (newaxis), and Ellipsis.
        Indexing is restricted to the first ``batch_dims`` axes. The linear-superposition
        axis (if present) sits immediately before core axes and is not directly indexable.

        The returned object preserves the lin-sup axis position; integers remove batch axes,
        None inserts batch axes.
        """
        batch_index, _, _ = batch_indexer_info(indexer, self.batch_dims)
        A = self.A[batch_index]
        b = self.b[batch_index]
        c = self.c[batch_index]
        return PolyExpAnsatz(A, b, c, lin_sup=self._lin_sup)

    def __mul__(self, other: Scalar) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, self.c * other, lin_sup=self._lin_sup)

    def __neg__(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, -self.c, lin_sup=self._lin_sup)

    def __repr__(self) -> str:
        r"""Returns a string representation of the PolyExpAnsatz object."""
        # Create a descriptive name
        display_name = f'"{self.name}"' if self.name else "unnamed"

        # Build the representation string
        repr_str = [
            f"PolyExpAnsatz({display_name})",
            f"  Batch shape: {self.batch_shape}",
            f"  Linear superposition: {self._lin_sup}",
            f"  Variables: {self.num_CV_vars} CV + {self.num_derived_vars} derived = {self.num_vars} total",
            "  Parameter shapes:",
            f"    A: {self.A.shape}",
            f"    b: {self.b.shape}",
            f"    c: {self.c.shape}",
        ]

        # Add information about simplification status
        if self._simplified:
            repr_str.append("  Status: simplified")

        return "\n".join(repr_str)

    def __str__(self) -> str:
        return f"PolyExpAnsatz(batch_shape={self.batch_shape}, lin_sup={self._lin_sup}, num_CV_vars={self.num_CV_vars}, num_derived_vars={self.num_derived_vars})"

    def __truediv__(self, other: Scalar) -> PolyExpAnsatz:
        # handle the case where other is a batched scalar
        shape = math.shape(other)
        if shape != ():
            delta = len(self.c.shape) - len(shape)
            other = math.reshape(other, shape + (1,) * delta)
        return PolyExpAnsatz(self.A, self.b, self.c / other, lin_sup=self._lin_sup)
