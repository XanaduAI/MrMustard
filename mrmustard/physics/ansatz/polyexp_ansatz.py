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

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from IPython.display import display
from numpy.typing import ArrayLike

from mrmustard import math, settings, widgets
from mrmustard.math.parameters import Variable
from mrmustard.physics.fock_utils import c_in_PS
from mrmustard.physics.gaussian_integrals import (
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
    join_Abc,
)
from mrmustard.physics.utils import generate_batch_str, verify_batch_triple
from mrmustard.utils.argsort import argsort_gen
from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Scalar,
    Vector,
)

from ..utils import outer_product_batch_str, reshape_args_to_batch_string
from .base import Ansatz

__all__ = ["PolyExpAnsatz"]


class PolyExpAnsatz(Ansatz):
    r"""
    This class represents the ansatz function:

        :math:`F^{(i)}(z) = \sum_k c^{(i)}_{k} \partial_y^k \textrm{exp}(\frac{1}{2}(z,y)^T A^{(i)} (z,y) + (z,y)^T b^{(i)})|_{y=0}`

    with ``k`` and ``i`` multi-indices. The ``i`` multi-index is a batch index of shape ``L`` that
    can be used for linear superposition or batching purposes. Each of the ``c^{(i)}_k`` tensors are
    contracted with the array of derivatives :math:`\partial_y^k` to form polynomials of derivatives.
    The tensors :math:`c^{(i)}_{k}` contain the coefficients of the polynomial of derivatives and
    have shape ``(*L, *derived)``, where ``*derived`` is the shape of the derived variables, which
    implies ``len(c.shape[1:]) = m``. The matrices :math:`A^{(i)}` and vectors :math:`b^{(i)}` are
    the parameters of the exponential terms in the ansatz, with :math:`z\in\mathbb{C}^{n}` and
    :math:`y\in\mathbb{C}^{m}`. ``A`` and b`` have shape ``(*L, n+m, n+m)`` and ``(*L, n+m)``,
    respectivly.

    .. code-block::

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

    Args:
        A: A batch of quadratic coefficient :math:`A^{(i)}`.
        b: A batch of linear coefficients :math:`b^{(i)}`.
        c: A batch of arrays :math:`c^{(i)}`.
        name:
    """

    def __init__(
        self,
        A: ComplexMatrix | Batch[ComplexMatrix] | None,
        b: ComplexVector | Batch[ComplexVector] | None,
        c: ComplexTensor | Batch[ComplexTensor] | None,
        name: str = "",
        lin_sup: bool = False,
    ):
        super().__init__()
        self.name = name
        self._simplified = False
        self._lin_sup = lin_sup

        self._A = math.astensor(A) if A is not None else None
        self._b = math.astensor(b) if b is not None else None
        self._c = math.astensor(c) if c is not None else None

        verify_batch_triple(self._A, self._b, self._c)

        if A is not None:
            self._batch_shape = tuple(self._A.shape[:-2])

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
    def batch_dims(self) -> tuple[int, ...]:
        return len(self.batch_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self._A is None:
            self._generate_ansatz()
        return self._batch_shape

    @property
    def batch_size(self) -> int:
        return math.prod(self.batch_shape) if self.batch_shape else 0

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The batch of polynomial coefficients :math:`c^{(i)}_{k}`.
        """
        self._generate_ansatz()
        return self._c

    @property
    def conj(self):
        return PolyExpAnsatz(
            math.conj(self.A),
            math.conj(self.b),
            math.conj(self.c),
            lin_sup=self._lin_sup,
        )

    @property
    def data(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""Returns the triple, which is necessary to reinstantiate the ansatz."""
        return self.triple

    @property
    def num_CV_vars(self) -> int:
        r"""
        The number of continuous variables that remain after the polynomial of derivatives is applied.
        This is the number of continuous variables of the Ansatz function itself.
        """
        return self.A.shape[-1] - self.num_derived_vars

    @property
    def core_dims(self) -> int:
        r"""
        The number of core variables of the ansatz. Equivalent to ``self.num_CV_vars``.
        """
        return self.num_CV_vars

    @property
    def num_derived_vars(self) -> int:
        r"""
        The number of derived variables that are derived by the polynomial of derivatives.
        """
        return len(self.shape_derived_vars)

    @property
    def num_vars(self):
        return self.A.shape[-1]

    @property
    def PS(self) -> PolyExpAnsatz:
        r"""
        The ansatz defined using real (i.e., phase-space) variables.
        """
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

        A_final = A[..., [0, 2, 1, 3], :][..., :, [0, 2, 1, 3]]
        b_final = b[..., [0, 2, 1, 3]]

        return PolyExpAnsatz(A_final, b_final, c, lin_sup=self._lin_sup)

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz, i.e. F(0)
        """
        if self.num_CV_vars == 0 and self.num_derived_vars == 0:
            ret = math.einsum("...a->...", self.c) if self._lin_sup else self.c
        elif self.num_CV_vars == 0:
            ret = self()
        else:
            ret = self(*math.zeros(self.num_CV_vars))
        return ret

    @property
    def shape_derived_vars(self) -> tuple[int, ...]:
        r"""
        The shape of the coefficients of the polynomial of derivatives.
        """
        return tuple(self.c.shape[self.batch_dims :])

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
        r"""
        Creates an ansatz given a function and its kwargs. This ansatz is lazily instantiated,
        i.e. the function is not called until the A,b,c attributes are accessed (even internally).
        """
        ansatz = cls(None, None, None, None)
        ansatz._fn = fn
        ansatz._kwargs = kwargs
        return ansatz

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        (ret._kwargs,) = children
        (
            ret._batch_shape,
            ret._lin_sup,
            ret._fn,
            ret._A,
            ret._b,
            ret._c,
            ret._simplified,
            ret.name,
        ) = aux_data
        return ret

    def contract(
        self,
        other: PolyExpAnsatz,
        idx1: Sequence[str | int],
        idx2: Sequence[str | int],
        idx_out: Sequence[str | int],
    ) -> PolyExpAnsatz:
        r"""Contracts this ansatz with another using einsum-style notation with labels.

        Indices are specified as sequences of labels (str or int). Batch dimensions must
        be strings, core dimensions must be integers. Integer labels refer to the
        index within the CV variables (0 to num_CV_vars-1).

        Unlike ArrayAnsatz, for PolyExpAnsatz:
        1. Only core (CV) variables can be contracted.
        2. Contracted core variable labels cannot appear in the output index list.
        3. Contracted batch labels *must* appear in the output index list.

        Example:
            `self.contract(other, idx1=['b', 0], idx2=['b', 0], idx_out=[0])`
            would raise an error because core label 0 is contracted and appears in the output.
            `self.contract(other, idx1=['a', 0], idx2=['b', 1], idx_out=[0, 1])`
            would raise an error because batch labels 'a', 'b' are not in the output.

        Args:
            other: The other PolyExpAnsatz to contract with.
            idx1: Sequence of labels (str/int) for this ansatz's dimensions (batch_dims + num_CV_vars).
            idx2: Sequence of labels (str/int) for the other ansatz's dimensions (batch_dims + num_CV_vars).
            idx_out: Sequence of labels for the output dimensions.

        Returns:
            The contracted PolyExpAnsatz.

        Raises:
            ValueError: If index sequences have incorrect length, invalid labels, or violate PolyExpAnsatz contraction rules.
        """
        # --- Parse and Validate Inputs ---
        ls1 = int(self._lin_sup)
        ls2 = int(other._lin_sup)
        batch1 = [label for label in idx1 if isinstance(label, str)] + ["__ls1"] * ls1
        core1 = [label for label in idx1 if isinstance(label, int)]
        batch2 = [label for label in idx2 if isinstance(label, str)] + ["__ls2"] * ls2
        core2 = [label for label in idx2 if isinstance(label, int)]
        batch_out = (
            [label for label in idx_out if isinstance(label, str)]
            + ["__ls1"] * ls1
            + ["__ls2"] * ls2
        )
        core_out = [label for label in idx_out if isinstance(label, int)]

        # Check dimensions match expected counts
        for actual, expected, description in [
            (len(batch1) - ls1, self.batch_dims - ls1, "batch labels in idx1"),
            (len(batch2) - ls2, other.batch_dims - ls2, "batch labels in idx2"),
            (len(core1), self.num_CV_vars, "core labels in idx1"),
            (len(core2), other.num_CV_vars, "core labels in idx2"),
            (
                len(batch_out) - ls1 - ls2,
                len(set(batch1) | set(batch2)) - ls1 - ls2,
                "batch labels in idx_out",
            ),
            (len(core_out), len(set(core1) ^ set(core2)), "core labels in idx_out"),
        ]:
            if actual != expected:
                raise ValueError(f"Expected {expected} {description}, found {actual}.")

        # Check contracted core labels don't appear in output
        if not set(core_out).isdisjoint(contracted_cores := set(core1) & set(core2)):
            raise ValueError(
                "idx_out cannot contain core labels that are contracted: "
                f"{set(core_out) & contracted_cores}",
            )

        # --- Prepare for complex_gaussian_integral_2 ---  # TODO: finish fixing this
        contracted_core = set(core1) & set(core2)
        idx1_cv = sorted(core1.index(label) for label in contracted_core)
        idx2_cv = sorted(core2.index(label) for label in contracted_core)

        ls_labels = {f"__ls{i}" for i, ls in [(1, ls1), (2, ls2)] if ls}
        unique_batch_labels = set(batch1) | set(batch2) | ls_labels
        label_to_char = {label: chr(97 + i) for i, label in enumerate(unique_batch_labels)}
        batch1_chars = "".join([label_to_char[label] for label in batch1])
        batch2_chars = "".join([label_to_char[label] for label in batch2])
        batch_out_chars = "".join([label_to_char[label] for label in batch_out])
        batch_str = f"{batch1_chars},{batch2_chars}->{batch_out_chars}"

        # --- Call complex_gaussian_integral_2 ---
        A, b, c = complex_gaussian_integral_2(
            self.triple,
            other.triple,
            idx1_cv,
            idx2_cv,
            batch_str,
        )

        # --- Reorder core dimensions ---
        if self._lin_sup and other._lin_sup:
            batch_shape = self.batch_shape[:-1]
            flattened = self.batch_shape[-1] * other.batch_shape[-1]
            A = math.reshape(A, (*batch_shape, flattened, *tuple(A.shape[-2:])))
            b = math.reshape(b, (*batch_shape, flattened, *tuple(b.shape[-1:])))
            c = math.reshape(c, (*batch_shape, flattened, *self.shape_derived_vars))

        result = PolyExpAnsatz(A, b, c, lin_sup=self._lin_sup or other._lin_sup)
        leftover_core = [i for i in idx1 + idx2 if isinstance(i, int) and i not in contracted_core]

        perm = [leftover_core.index(i) for i in core_out]
        return result.reorder(perm)

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

    def eval(
        self,
        *z: Vector | None,
        batch_string: str | None = None,
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

    def reorder(self, order: Sequence[int]):
        r"""
        Reorders the CV indices of an (A,b,c) triple.
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

    def reorder_batch(self, order: Sequence[int]):  # TODO: omit last batch index if lin_sup
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
    def simplify(self) -> None:
        r"""
        Returns a new ansatz simplified by combining terms that have the
        same exponential part, i.e. two components of the batch are considered equal if their
        ``A`` and ``b`` are equal. In this case only one is kept and the corresponding ``c`` are added.

        Will return immediately if the ansatz has already been simplified, so it is safe to re-call.
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
        r"""
        Computes the trace of the ansatz across the specified pairs of CV variables.

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
        A, b, c = complex_gaussian_integral_1(self.triple, idx_z, idx_zconj, measure=measure)
        return PolyExpAnsatz(A, b, c, lin_sup=self._lin_sup)

    def _combine_exp_and_poly(
        self,
        exp_sum: Batch[ComplexTensor],
        poly: Batch[ComplexTensor],
        c: Batch[ComplexTensor],
    ) -> Batch[ComplexTensor]:
        r"""
        Combines exponential and polynomial parts using einsum. Needed in ``__call__``.
        """
        poly_string = "".join(chr(i) for i in range(97, 97 + len(self.shape_derived_vars)))
        return math.einsum(f"...,...{poly_string},...{poly_string}->...", exp_sum, c, poly)

    def _compute_exp_part(
        self,
        z: Batch[Vector],
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
    ) -> Batch[Scalar]:
        r"""
        Computes the exponential part of the ansatz evaluation. Needed in ``__call__``.
        The exponential part is given by:
        .. math::
            \exp\left(\frac{1}{2} z^T A z + b^T z\right)
        where :math:`A` is the matrix of the quadratic part of the ansatz and :math:`b` is the vector of the linear part
        that correspond to the vector of given CV variables.
        """
        n = self.num_CV_vars
        A_part = math.einsum("...a,...b,...ab->...", z, z, A[..., :n, :n])
        b_part = math.einsum("...a,...a->...", z, b[..., :n])
        return math.exp(1 / 2 * A_part + b_part)

    def _compute_polynomial_part(
        self,
        z: Batch[Vector],
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
    ) -> Batch[Scalar]:
        r"""
        Computes the polynomial part of the ansatz evaluation. Needed in ``__call__``.
        """
        n = self.num_CV_vars
        batch_shape = z.shape[:-1]
        b_poly = math.einsum("...ab,...a->...b", A[..., :n, n:], z) + b[..., n:]
        return math.hermite_renormalized(
            A[..., n:, n:],
            b_poly,
            math.ones(batch_shape, dtype=math.complex128),
            shape=self.shape_derived_vars,
        )

    def _find_unique_terms_sorted(
        self,
    ) -> tuple[tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]], list[int]]:
        r"""
        Finds unique terms by first sorting the batch dimension and adds the corresponding c values.

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

    def _generate_ansatz(self):
        r"""
        This method computes and sets the (A, b, c) triple given a function and its kwargs.
        """
        if self._should_regenerate():
            params = {}
            for name, param in self._kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param

            A, b, c = self._fn(**params)
            self._A = math.astensor(A)
            self._b = math.astensor(b)
            self._c = math.astensor(c)
            verify_batch_triple(self._A, self._b, self._c)
            self._batch_shape = tuple(self._A.shape[:-2])

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        display(widgets.bargmann(self))

    def _order_batch(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        This method orders the batch dimension by the lexicographical order of the
        flattened arrays (A, b, c). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two PolyExp ansatz.

        Returns:
            The ordered vectorized (A, b, c) triple.
        """
        if not self.batch_shape:
            return self.A, self.b, self.c
        A_vectorized = math.reshape(self.A, (-1, self.num_vars, self.num_vars))
        b_vectorized = math.reshape(self.b, (-1, self.num_vars))
        c_vectorized = math.reshape(self.c, (-1, *self.shape_derived_vars))
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
        r"""
        Partially evaluates the ansatz by fixing some of its variables to specific values.

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
            z_batch_idxs
            + ansatz_batch_idxs
            + (len(z_batch_idxs + ansatz_batch_idxs),),  # tensorflow
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

    def _should_regenerate(self):
        return (
            self._A is None
            or self._b is None
            or self._c is None
            or Variable in {type(param) for param in self._kwargs.values()}
        )

    def _tree_flatten(self):  # pragma: no cover
        children, aux_data = super()._tree_flatten()
        aux_data += (self._A, self._b, self._c, self._simplified, self.name)
        return (children, aux_data)

    def __add__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Adds two PolyExp ansatze together. This is equivalent to stacking their respective triples
        along a batch dimension, which is to be interpreted to mean a linear superposition.
        In order to use the __add__ method, the ansatze must have the same number of CV variables,
        and zero or one batch dimensions. The reason for this restriction on the number of batch
        dimensions is that if there are multiple batch dimensions, it is not clear which one is used
        as meaning "linear superposition". In that case, the stacking of the triples should be done
        by the user.
        """
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

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Tensor product of this PolyExpAnsatz with another. Equivalent to :math:`H(a,b) = F(a) * G(b)`.
        As it distributes over addition on both self and other, the batch shape of the result is the
        outer product of the batch shapes of this ansatz and the other one. Use with moderation.

        Args:
            other: Another PolyExpAnsatz.

        Returns:
            The tensor product of this PolyExpAnsatz and other.
        """
        As, bs, cs = join_Abc(
            self.triple,
            other.triple,
            outer_product_batch_str(
                self.batch_dims,
                other.batch_dims,
                lin_sup=[0] * self._lin_sup + [1] * other._lin_sup,
            ),
        )
        if self._lin_sup and other._lin_sup:  # we have two linear superposition dimensions
            As = math.reshape(As, (*As.shape[:-4], As.shape[-4] * As.shape[-3], *As.shape[-2:]))
            bs = math.reshape(bs, (*bs.shape[:-3], bs.shape[-3] * bs.shape[-2], *bs.shape[-1:]))
            cs = math.reshape(cs, (*cs.shape[:-2], cs.shape[-2] * cs.shape[-1]))
        return PolyExpAnsatz(As, bs, cs, lin_sup=self._lin_sup or other._lin_sup)

    def __call__(self: PolyExpAnsatz, *z_inputs: ArrayLike | None) -> Batch[ComplexTensor]:
        r"""
        Evaluates the ansatz at the given batch of points. Each point can have arbitray batch dimensions,
        as long as they are broadcastable. If some of the points are not specified (None), the result
        will be a partially evaluated ansatz.
        If the combined shape of the inputs is ``(*b, n)`` where ``n`` is the number of CV variables in the ansatz
        and ``*b`` is the batch dimensions of the combined inputs, then the output will have shape ``(*b, *L)``
        where ``*L`` is the batch shape of the ansatz itself.

        Args:
            z: A batch of points where the function is evaluated (or None).
                The shape of each point can be arbitrary, as long as they are broadcastable.

        Returns:
            The evaluated function with shape (*b, *L) where:
               - *b are the batch dimensions of the combined inputs.
               - *L is the batch shape of the ansatz.
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
            z_batch_idxs
            + ansatz_batch_idxs
            + (len(z_batch_idxs + ansatz_batch_idxs),),  # tensorflow
        )

        A = math.broadcast_to(self.A, z_batch_shape + self.A.shape)
        b = math.broadcast_to(self.b, z_batch_shape + self.b.shape)
        c = math.broadcast_to(self.c, z_batch_shape + self.c.shape)

        exp_sum = self._compute_exp_part(z, A, b)
        if self.num_derived_vars == 0:  # purely gaussian
            ret = math.einsum("...,...->...", exp_sum, c)
        else:
            poly = self._compute_polynomial_part(z, A, b)
            ret = self._combine_exp_and_poly(exp_sum, poly, c)

        return math.sum(ret, axis=-1) if self._lin_sup else ret

    def __eq__(self, other) -> bool:
        if not isinstance(other, PolyExpAnsatz):
            return False
        self_A, self_b, self_c = self._order_batch()
        other_A, other_b, other_c = other._order_batch()
        return (
            math.allclose(self_A, other_A, atol=settings.ATOL)
            and math.allclose(self_b, other_b, atol=settings.ATOL)
            and math.allclose(self_c, other_c, atol=settings.ATOL)
        )

    def __mul__(self, other: Scalar | ArrayLike | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c * other, lin_sup=self._lin_sup)
            except Exception as e:
                raise TypeError(f"Cannot multiply PolyExpAnsatz and {other.__class__}.") from e

        else:
            raise NotImplementedError(
                "Multiplication of PolyExpAnsatz with other PolyExpAnsatz is not implemented.",
            )

    def __neg__(self) -> PolyExpAnsatz:
        return PolyExpAnsatz(self.A, self.b, -self.c, lin_sup=self._lin_sup)

    def __repr__(self) -> str:
        r"""Returns a string representation of the PolyExpAnsatz object."""
        self._generate_ansatz()  # Ensure parameters are generated if needed

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

        # Add information about function generation if applicable
        if self._fn is not None:
            fn_name = getattr(self._fn, "__name__", str(self._fn))
            repr_str.append(f"  Generated from: {fn_name}")
            if self._kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in self._kwargs.items())
                repr_str.append(f"  Parameters: {param_str}")

        return "\n".join(repr_str)

    def __truediv__(self, other: Scalar | ArrayLike | PolyExpAnsatz) -> PolyExpAnsatz:
        if not isinstance(other, PolyExpAnsatz):  # could be a number
            try:
                return PolyExpAnsatz(self.A, self.b, self.c / other, lin_sup=self._lin_sup)
            except Exception as e:
                raise TypeError(f"Cannot divide PolyExpAnsatz and {other.__class__}.") from e
        else:
            raise NotImplementedError(
                "Division of PolyExpAnsatz with other PolyExpAnsatz is not implemented.",
            )
