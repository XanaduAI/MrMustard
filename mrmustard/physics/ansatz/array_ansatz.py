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
This module contains the array ansatz.
"""

from __future__ import annotations
from typing import Any, Callable, Sequence, Tuple, List, Optional, Union, Dict
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from IPython.display import display

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable
from mrmustard.utils.typing import Batch, Scalar, Tensor

from .base import Ansatz
from ..utils import outer_product_batch_str

__all__ = ["ArrayAnsatz"]


class ArrayAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz as a multidimensional array.

    .. code-block::

          >>> from mrmustard.physics.ansatz import ArrayAnsatz

          >>> array = np.random.random((2, 4, 5))
          >>> ansatz = ArrayAnsatz(array)

    Args:
        array: A (potentially) batched array.
        batch_dims: The number of batch dimensions.
    """

    def __init__(self, array: Batch[Tensor] | None, batch_dims: int = 0):
        super().__init__()
        self._array = math.astensor(array) if array is not None else None
        self._batch_dims = batch_dims
        self._batch_shape = self._array.shape[:batch_dims] if array is not None else ()
        self._original_abc_data = None

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        return self._array

    @array.setter
    def array(self, value: Tensor):
        self._array = math.astensor(value)
        self._batch_shape = value.shape[: self.batch_dims]

    @property
    def batch_dims(self) -> int:
        return self._batch_dims

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def batch_size(self):
        return int(np.prod(self.batch_shape)) if self.batch_shape != () else 0  # tensorflow

    @property
    def core_dims(self) -> int:
        r"""
        The number of core dimensions of this ansatz.
        """
        return len(self.core_shape)

    @property
    def conj(self):
        return ArrayAnsatz(math.conj(self.array), self.batch_dims)

    @property
    def core_shape(self) -> tuple[int, ...] | None:
        r"""
        The core dimensions of this ansatz.
        """
        return self.array.shape[self.batch_dims :]

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - self.batch_dims

    @property
    def scalar(self) -> Scalar | ArrayLike:
        r"""
        The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        """
        return self.array[(...,) + (0,) * self.core_dims]

    @property
    def triple(self) -> tuple:
        r"""
        The data of the original PolyExpAnsatz if it exists.
        """
        if self._original_abc_data is None:
            raise AttributeError("This ArrayAnsatz does not have (A,b,c) data.")
        return self._original_abc_data

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> ArrayAnsatz:
        return cls(**data)

    @classmethod
    def from_function(cls, fn: Callable, batch_dims: int = 0, **kwargs: Any) -> ArrayAnsatz:
        ret = cls(None, batch_dims=batch_dims)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def _validate_and_prepare_indices(
        self,
        idx1: Union[int, Tuple[int, ...], List[int], None],
        idx2: Union[int, Tuple[int, ...], List[int], None],
        core_str: Optional[str],
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """
        Validates contraction inputs and ensures idx1/idx2 are tuples if provided.
        Returns the validated/converted idx1 and idx2 as tuples or None.
        """
        if core_str is not None:
            if idx1 is not None or idx2 is not None:
                raise ValueError("Cannot specify both `core_str` and `idx1`/`idx2`.")
            return None, None  # Not using idx1/idx2

        if idx1 is None or idx2 is None:
            raise ValueError("Either `core_str` or both `idx1` and `idx2` must be provided.")

        # Convert to tuples for consistent handling
        _idx1 = tuple(idx1) if isinstance(idx1, (list, tuple)) else (idx1,)
        _idx2 = tuple(idx2) if isinstance(idx2, (list, tuple)) else (idx2,)

        if len(_idx1) != len(_idx2):
            raise ValueError("idx1 and idx2 must have the same length when provided.")

        return _idx1, _idx2

    def _get_einsum_components(
        self,
        other: ArrayAnsatz,
        core_str: Optional[str],
        idx1: Optional[Tuple[int, ...]],
        idx2: Optional[Tuple[int, ...]],
        batch_str: Optional[str],
    ) -> Tuple[List[str], str, List[str], str, Dict[str, Tuple[int, int]]]:
        """
        Determines batch and core einsum components based on input method.

        Returns:
            tuple: (batch_input_parts, batch_output_str,
                    core_input_parts, core_output_str,
                    contracted_indices_map)
        """
        # --- Batch String Processing ---
        if batch_str is None:
            batch_str = outer_product_batch_str(self.batch_shape, other.batch_shape)
        try:
            batch_input_str, batch_output_str = batch_str.split("->")
            batch_input_parts = batch_input_str.split(",")
            if len(batch_input_parts) != 2:
                raise ValueError("Batch string must have exactly two input parts.")
            used_batch_chars = set(batch_input_str) | set(batch_output_str)
            start_ord = (max(ord(c) for c in used_batch_chars) + 1) if used_batch_chars else 97
        except ValueError as e:
            raise ValueError(f"Invalid batch_str format: '{batch_str}'. {e}") from e

        # --- Core String Processing ---
        contracted_indices_map = {}
        if core_str is not None:
            # Parse core_str
            try:
                core_input_str, core_output_str = core_str.split("->")
                core_input_parts = core_input_str.split(",")
                if len(core_input_parts) != 2:
                    raise ValueError("Core string must have exactly two input parts.")

                # Validate characters and map dimensions
                map1 = {}
                for i, char in enumerate(core_input_parts[0]):
                    if i >= self.core_dims:
                        raise IndexError(
                            f"Index for core char '{char}' ({i}) out of bounds for self (dims={self.core_dims})"
                        )
                    map1[char] = i
                map2 = {}
                for i, char in enumerate(core_input_parts[1]):
                    if i >= other.core_dims:
                        raise IndexError(
                            f"Index for core char '{char}' ({i}) out of bounds for other (dims={other.core_dims})"
                        )
                    map2[char] = i

                # Identify contracted indices and populate map
                chars1, chars2 = set(core_input_parts[0]), set(core_input_parts[1])
                contracted_chars = chars1 & chars2
                for char in contracted_chars:
                    contracted_indices_map[char] = (map1[char], map2[char])

            except ValueError as e:
                raise ValueError(f"Invalid core_str format: '{core_str}'. {e}") from e
            except IndexError as e:  # Re-raise index errors from validation
                raise e
        else:
            # Generate core string components from idx1/idx2
            # Input idx1/idx2 are guaranteed to be tuples here by _validate_and_prepare_indices
            if any(i >= self.core_dims for i in idx1) or any(j >= other.core_dims for j in idx2):
                raise IndexError(
                    f"Index in idx1/idx2 out of bounds for core dims ({self.core_dims}, {other.core_dims})"
                )

            core_indices1 = [chr(start_ord + i) for i in range(self.core_dims)]
            core_indices2 = [chr(start_ord + self.core_dims + i) for i in range(other.core_dims)]

            for i_self, i_other in zip(idx1, idx2):
                contracted_char = core_indices1[i_self]
                core_indices2[i_other] = contracted_char
                contracted_indices_map[contracted_char] = (i_self, i_other)

            core_input_parts = ["".join(core_indices1), "".join(core_indices2)]
            survivors1 = [c for i, c in enumerate(core_indices1) if i not in idx1]
            survivors2 = [c for i, c in enumerate(core_indices2) if i not in idx2]
            core_output_str = "".join(survivors1 + survivors2)

        return (
            batch_input_parts,
            batch_output_str,
            core_input_parts,
            core_output_str,
            contracted_indices_map,
        )

    def _reduce_arrays_if_needed(
        self, other: ArrayAnsatz, contracted_indices_map: dict[str, tuple[int, int]]
    ) -> tuple:
        r"""Reduces array dimensions if necessary for contraction.
        Needed in contract method.
        """
        shape_s = list(self.core_shape)
        shape_o = list(other.core_shape)
        needs_reduction = False
        for s_idx, o_idx in contracted_indices_map.values():
            min_dim = min(shape_s[s_idx], shape_o[o_idx])
            if shape_s[s_idx] != min_dim or shape_o[o_idx] != min_dim:
                needs_reduction = True
            shape_s[s_idx] = min_dim
            shape_o[o_idx] = min_dim

        # Use tuple for reduce shape argument
        array_self = self.reduce(tuple(shape_s)).array if needs_reduction else self.array
        array_other = other.reduce(tuple(shape_o)).array if needs_reduction else other.array
        return array_self, array_other

    def contract(
        self,
        other: ArrayAnsatz,
        idx1: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        idx2: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        core_str: Optional[str] = None,
        batch_str: Optional[str] = None,
    ) -> ArrayAnsatz:
        r"""
        Contracts two ansatze across the specified variables and batch dimensions.
        Either `idx1` and `idx2` OR a more flexible `core_str` must be provided.

        If `idx1` and `idx2` are used, they specify the core dimensions to contract (sum over).
        All other core dimensions survive in their original relative order (self's survivors
        followed by other's survivors).

        If `core_str` is used, it specifies the contraction using einsum notation for the
        core dimensions, allowing for index survival and reordering in a single call.

        Args:
            other: The other ArrayAnsatz to contract with.
            idx1: The variables of the first ansatz to contract (used if core_str is None).
            idx2: The variables of the second ansatz to contract (used if core_str is None).
            core_str: The einsum string defining the core contraction (used if idx1/idx2 are None).
                      Example: "ab,bc->ac" contracts the 2nd dim of self with the 1st dim of other.
            batch_str: The (optional) batch dimensions to contract over with the
                same syntax as in ``np.einsum``. If not indicated, the batch dimensions
                are taken in outer product.

        Returns:
            The contracted ansatz.
        """
        # 1. Validate inputs and ensure idx1/idx2 are tuples if used
        _idx1, _idx2 = self._validate_and_prepare_indices(idx1, idx2, core_str)

        # 2. Determine all einsum string components and contraction map
        (
            batch_input_parts,
            batch_output_str,
            core_input_parts,
            core_output_str,
            contracted_indices_map,
        ) = self._get_einsum_components(other, core_str, _idx1, _idx2, batch_str)

        # 3. Reduce arrays if shapes mismatch on contracted dimensions
        array_self, array_other = self._reduce_arrays_if_needed(other, contracted_indices_map)

        # 4. Construct the final einsum string
        einsum_str = (
            f"{batch_input_parts[0]}{core_input_parts[0]},"
            f"{batch_input_parts[1]}{core_input_parts[1]}->"
            f"{batch_output_str}{core_output_str}"
        )

        # 5. Execute einsum
        try:
            result = math.einsum(einsum_str, array_self, array_other)
        except Exception as e:
            raise ValueError(f"Failed to execute einsum with string '{einsum_str}'.") from e

        return ArrayAnsatz(result, batch_dims=len(batch_output_str))

    def reduce(self, shape: Sequence[int]) -> ArrayAnsatz:
        r"""
        Returns a new ``ArrayAnsatz`` with a sliced core shape.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.ansatz import ArrayAnsatz

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = ArrayAnsatz(array1)

            >>> fock2 = fock1.reduce((3, 3, 3))
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce((2, 2, 2))
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == ArrayAnsatz(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == ArrayAnsatz(array4)

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.
        """
        if shape == self.core_shape:
            return self
        if len(shape) != self.core_dims:
            raise ValueError(f"Expected shape of length {self.core_dims}, got {len(shape)}.")

        if any(s > t for s, t in zip(shape, self.core_shape)):
            warn("Warning: the fock array is being padded with zeros. Is this really necessary?")
            padded = math.pad(
                self.array,
                [(0, 0)] * self.batch_dims + [(0, s - t) for s, t in zip(shape, self.core_shape)],
            )
            return ArrayAnsatz(padded, self.batch_dims)

        return ArrayAnsatz(self.array[(...,) + tuple(slice(0, s) for s in shape)], self.batch_dims)

    def reorder(self, order: tuple[int, ...] | list[int]) -> ArrayAnsatz:
        order = list(range(self.batch_dims)) + [i + self.batch_dims for i in order]
        return ArrayAnsatz(math.transpose(self.array, order), self.batch_dims)

    def reorder_batch(self, order: Sequence[int]):
        if len(order) != self.batch_dims:
            raise ValueError(
                f"order must have length {self.batch_dims} (number of batch dimensions), got {len(order)}"
            )

        core_dims_indices = range(self.batch_dims, self.batch_dims + self.core_dims)
        new_array = math.transpose(self.array, list(order) + list(core_dims_indices))
        return ArrayAnsatz(new_array, self.batch_dims)

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data, "batch_dims": self.batch_dims}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> ArrayAnsatz:
        if len(idx_z) != len(idx_zconj) or not set(idx_z).isdisjoint(idx_zconj):
            raise ValueError("The idxs must be of equal length and disjoint.")
        order = (
            list(range(self.batch_dims))
            + [self.batch_dims + i for i in range(self.core_dims) if i not in idx_z + idx_zconj]
            + [self.batch_dims + i for i in idx_z]
            + [self.batch_dims + i for i in idx_zconj]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idx_z) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idx_z)] + (n, n))
        trace = math.trace(new_array)
        return ArrayAnsatz(trace, self.batch_dims)

    def _generate_ansatz(self):
        r"""
        Computes and sets the array given a function and its kwargs.
        """
        if self._should_regenerate():
            params = {}
            for name, param in self._kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param
            self.array = self._fn(**params)

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL or (w := widgets.fock(self)) is None:
            print(self)
            return
        display(w)

    def _should_regenerate(self):
        r"""
        Determines if the ansatz needs to be regenerated based on its current state
        and parameter types.
        """
        return self._array is None or Variable in {type(param) for param in self._kwargs.values()}

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        r"""
        Adds two ArrayAnsatz together. In order to use the __add__ method, the ansatze must have
        the same batch dimensions. The shape of the core arrays must be such that one can be reduced
        to the other. Other will be reduced to the shape of self. If you want the opposite use other + self.
        """
        if self.batch_dims != other.batch_dims:
            raise ValueError("Batch dimensions must match.")
        if self.core_shape != other.core_shape:
            if math.prod(self.core_shape) > math.prod(other.core_shape):
                self_array = self.array
                other_array = other.reduce(self.core_shape).array
            else:
                self_array = self.reduce(other.core_shape).array
                other_array = other.array
        else:
            self_array = self.array
            other_array = other.array
        return ArrayAnsatz(array=self_array + other_array, batch_dims=self.batch_dims)

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        if self.batch_shape != other.batch_shape:
            raise ValueError("Batch shapes must match.")
        batch_size = int(np.prod(self.batch_shape))
        self_reshaped = math.reshape(self.array, (batch_size, -1))
        other_reshaped = math.reshape(other.array, (batch_size, -1))
        new = math.einsum("ab,ac -> abc", self_reshaped, other_reshaped)
        new = math.reshape(new, self.batch_shape + self.core_shape + other.core_shape)
        return ArrayAnsatz(array=new, batch_dims=self.batch_dims)

    def __call__(self, _: Any):
        raise AttributeError("Cannot call an ArrayAnsatz.")

    def __eq__(self, other: Ansatz) -> bool:
        if self.batch_shape != other.batch_shape:
            return False
        slices = tuple(slice(0, min(si, oi)) for si, oi in zip(self.core_shape, other.core_shape))
        return np.allclose(self.array[(...,) + slices], other.array[(...,) + slices], atol=1e-10)

    def __mul__(self, other: Scalar) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array * other, batch_dims=self.batch_dims)

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array, batch_dims=self.batch_dims)

    def __truediv__(self, other: Scalar) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array / other, batch_dims=self.batch_dims)
