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
This module contains the Exp ansatz.
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

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable

from mrmustard.utils.argsort import argsort_gen

from .base import Ansatz

__all__ = ["ExpAnsatz"]


class ExpAnsatz(Ansatz):
    r"""
    This class represents the ansatz function:

        :math:`F^{(i)}(z) = c_i \textrm{exp}(\frac{1}{2}z^T A^{(i)} z + z^T b^{(i)})`

    The ``i`` index is a batch index that can be used for linear superposition or batching purposes.
    The ``c_i`` tensor is the coefficient of the exponential term in the ansatz.
    The matrices :math:`A^{(i)}` and vectors :math:`b^{(i)}` are the parameters of the exponential
    term in the ansatz, with :math:`z` a vector of continuous complex variables.
    They have shape ``(L, n, n)`` and ``(L, n)``, respectively for ``n`` continuous variables.

    .. code-block:: python

        >>> from mrmustard.physics.ansatz import ExpAnsatz
        >>> import numpy as np
        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = 1.0
        >>> F = ExpAnsatz(A, b, c)
        >>> z1 = np.array([1.0, 2.0])
        >>> z2 = np.array([3.0, 4.0])
        >>> # calculate the value of the function on a 2x2 grid
        >>> val = F(z1,z2)
        >>> assert val.shape == (2,2)
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix] | None,
        b: Batch[ComplexVector] | None,
        c: Batch[ComplexTensor] | None = np.ones([], dtype=np.complex128),
        name: str = "",
    ):
        super().__init__()
        self._init_with_batch = len(math.astensor(A).shape) > 2 if A is not None else False
        self._A = math.atleast_3d(math.astensor(A)) if A is not None else None
        self._b = math.atleast_2d(math.astensor(b)) if b is not None else None
        self._c = math.atleast_1d(math.astensor(c)) if c is not None else None
        self.name = name
        self._simplified = False
        self._fn = None
        self._fn_kwargs = {}
        self._batch_size = self._A.shape[0] if A is not None else None

    def _should_recompute(self) -> bool:
        r"""
        Checks if the ansatz should be recomputed when accessing the A, b, c attributes.
        """
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
        if self._should_recompute():
            params = {}
            for name, param in self._fn_kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param

            data = self._fn(**params)
            A, b, c = data
            self._A = math.atleast_3d(A)
            self._b = math.atleast_2d(b)
            self._c = math.atleast_1d(c)
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
    def data(self) -> tuple[int, ...]:
        r"""
        The shape of the data (A, b, c) tensor.
        """
        return self.triple

    @property
    def scalar(self) -> np.ndarray:
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
        return ExpAnsatz(math.conj(self.A), math.conj(self.b), math.conj(self.c))

    @property
    def num_CV_vars(self) -> int:
        r"""
        The number of continuous variables that remain after the polynomial of derivatives is applied.
        This is the number of continuous variables of the Ansatz function itself, i.e. the size of ``z``
        in :math:`F^{(i)}(z)`.
        """
        return self.A.shape[-1]

    @property
    def num_vars(self):
        r"""
        The total number of continuous variables of this ansatz.
        """
        return self.num_CV_vars

    @property
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""Returns the triple of parameters of the exponential part of the ansatz."""
        return self.A, self.b, self.c

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> ExpAnsatz:
        r"""Creates an ansatz from a dictionary. For deserialization purposes."""
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> ExpAnsatz:
        r"""Creates an ansatz given a function and its kwargs. This ansatz is lazily instantiated, i.e.
        the function is not called until the A,b,c attributes are accessed (even internally)."""
        ansatz = cls(None, None, None)
        ansatz._fn = fn
        ansatz._fn_kwargs = kwargs
        return ansatz

    def contract(
        self,
        other: ExpAnsatz,
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
        mode: Literal["zip", "kron"] = "kron",
    ) -> ExpAnsatz:
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
        return ExpAnsatz(A, b, c)

    def reorder(self, order: Sequence[int]):
        r"""
        Reorders the CV indices of an (A,b,c) triple.
        The length of ``order`` must be the number of CV variables.
        """
        if len(order) != self.num_CV_vars:
            raise ValueError(f"order must have length {self.num_CV_vars}, got {len(order)}")
        A = math.gather(math.gather(self.A, order, axis=-1), order, axis=-2)
        b = math.gather(self.b, order, axis=-1)
        return self.__class__(A, b, self.c)

    def simplify(self) -> None:
        r"""
        Simplifies an ansatz by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the ansatz has already been simplified, so it is always safe to call.
        """
        if self._simplified:
            return

        if self._should_recompute():
            raise ValueError("Cannot simplify an ansatz that has been generated from a function.")

        to_keep = self._find_unique_terms_sorted()
        self._A = math.gather(self.A, to_keep, axis=0)
        self._b = math.gather(self.b, to_keep, axis=0)
        self._c = math.gather(self.c, to_keep, axis=0)  # already added
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
                self._c = math.update_add_tensor(self._c, [[d0]], [self.c[d]])
        return to_keep

    def _order_batch(self):
        r"""
        This method orders the batch dimension by the lexicographical order of the
        flattened arrays (A, b, c). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two PolyExp ansatz.
        Needed in ``simplify``.
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
        self._A = math.gather(self.A, sorted_indices, axis=0)
        self._b = math.gather(self.b, sorted_indices, axis=0)
        self._c = math.gather(self.c, sorted_indices, axis=0)

    def to_dict(self) -> dict[str, ArrayLike]:
        r"""Returns a dictionary representation of the ansatz. For serialization purposes."""
        return {"A": self.A, "b": self.b, "c": self.c}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]):
        r"""
        Computes the trace of the ansatz across the specified CV variables.
        """
        if len(idx_z) != len(idx_zconj):
            raise ValueError("idx_z and idx_zconj must have the same length.")
        if max(idx_z) >= self.num_CV_vars or max(idx_zconj) >= self.num_CV_vars:
            raise ValueError(
                f"All indices must be between 0 and {self.num_CV_vars-1}. Got {idx_z} and {idx_zconj}."
            )
        A, b, c = complex_gaussian_integral_1(self.triple, idx_z, idx_zconj, measure=-1.0)
        return self.__class__(A, b, c)

    def _eval(self: ExpAnsatz, z: Batch[Vector]) -> Batch[ComplexTensor]:
        r"""
        Evaluates the ansatz at a batch of points ``z`` in C^(*b, n), where ``b`` is the batch shape
        and ``n`` is the number of CV variables.
        Note that since the ansatz may itself have a batch size `L`, the output will have shape
        ``(L, *b)``.

        Args:
            z: Point(s) in C^(*b, n) where the function is evaluated, ``b`` stands for batch shape.

        Returns:
            The value of the function at the point(s) with the same batch dimensions as ``z``.
            The output has shape (L, *b) where L is the batch size of the ansatz.
        """
        z = math.atleast_2d(z, dtype=math.complex128)
        z_batch_shape, z_dim = z.shape[:-1], z.shape[-1]
        if z_dim != self.num_CV_vars:
            raise ValueError(
                f"The last dimension of `z` must equal the number of CV variables {self.num_CV_vars}, got {z_dim}."
            )
        z = math.reshape(z, (-1, z_dim))  # shape (k, num_CV_vars)

        A_part = math.einsum("ka,kb,iab->ik", z, z, self.A)
        b_part = math.einsum("ka,ia->ik", z, self.b)
        exp_sum = math.exp(1 / 2 * A_part + b_part)  # shape (batch_size, k)
        result = math.reshape(
            math.einsum("ik,i->ik", exp_sum, self.c), (self.batch_size,) + z_batch_shape
        )
        return result if self._init_with_batch else math.squeeze(result, 0)

    def _partial_eval(self, z: Vector, indices: tuple[int, ...]) -> ExpAnsatz:
        r"""
        Returns a new ansatz that corresponds to currying (partially evaluate) the current one.
        For example, if ``self`` represents the function ``F(z0,z1,z2)``, the call
        ``self._partial_eval(np.array([2.0,3.0]), (0,2))`` returns
        ``G(z1) = F(2.0, z1, 3.0)`` as a new ansatz of a single variable.
        The vector ``z`` must have shape (r,), where ``r`` is the number of indices in ``indices``.
        It cannot have batch dimensions.

        Args:
            z: vector in ``C^(*b, r)`` where the function is evaluated.
            indices: indices of the variables of the ansatz to be evaluated.

        Returns:
            A new ansatz.
        """
        if len(indices) == self.num_CV_vars:
            raise ValueError(
                "Cannot curry a function of the same number of variables as the ansatz. "
                "Use the _eval or __call__ method instead."
            )

        # evaluated and remaining indices
        e = indices
        r = [i for i in range(self.num_CV_vars) if i not in indices]
        z = math.atleast_1d(z, dtype=math.complex128)
        # new A of shape (batch_size, r, r)
        new_A = math.gather(math.gather(self.A, r, axis=-1), r, axis=-2)

        # new b of shape (batch_size, r)
        A_er = math.gather(math.gather(self.A, e, axis=-2), r, axis=-1)  # shape (batch_size, e, r)
        b_r = math.einsum("ier,e->ir", A_er, z)  # shape (batch_size, r)
        new_b = math.gather(self.b, r, axis=-1) + b_r

        # new c of shape (batch_size,)
        A_ee = math.gather(math.gather(self.A, e, axis=-1), e, axis=-2)  # shape (batch_size, e, e)
        A_part = math.einsum("e,f,ief->i", z, z, A_ee)  # shape (batch_size,)
        b_part = math.einsum("e,ie->i", z, math.gather(self.b, e, axis=-1))  # shape (batch_size,)
        exp_sum = math.exp(1 / 2 * A_part + b_part)  # shape (batch_size,)
        new_c = exp_sum * self.c

        return ExpAnsatz(new_A, new_b, new_c)

    def __add__(self, other: ExpAnsatz) -> ExpAnsatz:
        r"""
        Adds two Exp ansatz together. This means concatenating them in the batch dimension.
        In the case where ``c`` on self and other have different shapes it will add padding zeros to make
        the shapes fit. Example: If the shape of ``c1`` is (1,3,4,5) and the shape of ``c2`` is (10,5,4,3) then the
        shape of the combined object will be (11,5,4,5).
        """
        if not isinstance(other, ExpAnsatz):
            raise TypeError(f"Cannot add ExpAnsatz and {other.__class__}.")
        if self.num_CV_vars != other.num_CV_vars:
            raise ValueError(
                f"The number of CV variables must match. Got {self.num_CV_vars} and {other.num_CV_vars}."
            )

        combined_matrices = math.concat([self.A, other.A], axis=0)
        combined_vectors = math.concat([self.b, other.b], axis=0)
        combined_arrays = math.concat([self.c, other.c], axis=0)

        return ExpAnsatz(
            combined_matrices,
            combined_vectors,
            combined_arrays,
        )

    def __and__(self, other: ExpAnsatz) -> ExpAnsatz:
        r"""
        Tensor product of this ExpAnsatz with another. Equivalent to :math:`H(a,b) = F(a) * G(b)`.
        As it distributes over addition on both self and other, the batch size of the result is the
        product of the batch size of this ansatz and the other one.

        Args:
            other: Another ExpAnsatz.

        Returns:
            The tensor product of this ExpAnsatz and other.
        """
        As, bs, cs = join_Abc(self.triple, other.triple, mode="kron")
        return ExpAnsatz(As, bs, cs)

    def __call__(
        self, *z: Vector | None, mode: Literal["zip", "kron"] = "kron"
    ) -> Scalar | ExpAnsatz:
        r"""
        Returns either the value of the ansatz or a new ansatz depending on the arguments.

        If an argument is None, the corresponding variable is not evaluated, and the method
        returns a new ansatz with the remaining variables unevaluated.
        For example, if the ansatz is a function of 3 variables F(z1, z2, z3) and we want to
        evaluate it at a point in C^2, we would get a new ansatz with one variable unevaluated:
        e.g. F(z1, z2, None).

        The ``mode`` argument can be used to specify how the vectors of arguments are broadcast together.
        The default is "zip", which is to broadcast the vectors pairwise. The alternative is "kron",
        which is to broadcast the vectors Kronecker-style. For example, if z1 and z2 are vectors,
        ``F(z1, z2, mode="zip")`` returns the array of values ``[F(z1[0], z2[0]), F(z1[1], z2[1]), …]``.
        On the other hand, ``F(z1, z2, mode="kron")`` returns the Kronecker product of the vectors,
        i.e. ``[[F(z1[0], z2[0]), F(z1[0], z2[1]), …], [F(z1[1], z2[0]), F(z1[1], z2[1]), …], …]``.
        The 'kron' style is useful if we want to pass points along each axis independently from each other.
        In `zip` mode the batch dimensions of the z vectors must match, while in `kron` mode they can differ,
        and the result will have a batch dimension equal to the product of the batch dimensions of the ansatz, followed
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

        if len(evaluated_indices) == self.num_CV_vars:  # Full evaluation: all variables provided.
            if mode == "zip":
                only_z = [math.atleast_2d(zi) for zi in z]
                batch_sizes = [zi.shape for zi in only_z]
                if not all(bs == batch_sizes[0] for bs in batch_sizes):
                    raise ValueError(
                        f"In mode 'zip' all z vectors must have the same batch size, got {batch_sizes}."
                    )
                # Concatenate along the last axis to form an array of shape (batch, n)
                z_input = math.concat(only_z, axis=0)
                return self._eval(math.transpose(z_input))
            elif mode == "kron":
                z = [math.astensor(zi) for zi in z]
                only_z = [math.atleast_1d(zi) for zi in z]
                if any(zi.ndim > 1 for zi in only_z):
                    raise ValueError("No more than one batch dimension is allowed.")
                # Create a meshgrid from the provided arrays; they may have different batch sizes.
                grid = np.meshgrid(*only_z, indexing="ij")
                z_combined = math.astensor(np.stack(grid, axis=-1))  # shape (b0, b1, …, b_n, n)
                z_flat = math.reshape(z_combined, (-1, self.num_CV_vars))  # shape (prod(b_i), n)
                result_flat = self._eval(z_flat)
                rest = tuple(s for zi in z for s in zi.shape)
                result = math.reshape(
                    result_flat, (self.batch_size,) + rest
                )  # shape (batch_size, b0, b1, …, b_n)
                return result if self._init_with_batch else math.squeeze(result, 0)
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:  # Partial evaluation: some CV variables are not provided.
            only_z = [math.atleast_1d(zi) for zi in z if zi is not None]  # no batch dimension
            z_input = math.concat(only_z, axis=-1)  # shape (r,) with r evaluated indices.
            return self._partial_eval(z_input, evaluated_indices)

    def __eq__(self, other: ExpAnsatz) -> bool:
        if not isinstance(other, ExpAnsatz):
            return False
        self._order_batch()
        other._order_batch()
        return (
            np.allclose(self.A, other.A, atol=1e-9)
            and np.allclose(self.b, other.b, atol=1e-9)
            and np.allclose(self.c, other.c, atol=1e-9)
        )

    def __mul__(self, other: Scalar | ExpAnsatz) -> ExpAnsatz:
        if not isinstance(other, ExpAnsatz):  # could be a number
            try:
                return ExpAnsatz(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot multiply ExpAnsatz and {other.__class__}.") from e

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

        newA = A1 + A2
        newb = b1 + b2
        newc = math.kron(self.c, other.c)

        return ExpAnsatz(A=newA, b=newb, c=newc)

    def __neg__(self) -> ExpAnsatz:
        return ExpAnsatz(self.A, self.b, -self.c)

    def __truediv__(self, other: Scalar | ExpAnsatz) -> ExpAnsatz:
        if not isinstance(other, ExpAnsatz):  # could be a number
            try:
                return ExpAnsatz(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot multiply ExpAnsatz and {other.__class__}.") from e
        raise NotImplementedError("Division of ExpAnsatz is not implemented.")

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        display(widgets.bargmann(self))
