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

# pylint: disable=redefined-outer-name

"""
This module contains the implementation of a tensor wrapper class.
"""
import string
from numbers import Number
from typing import List, Optional, Union

from mrmustard.math.backend_manager import BackendManager

math = BackendManager()


class MMTensor:
    r"""A Mr Mustard tensor (a wrapper around an array that implements the numpy array API)."""

    def __init__(self, array, axis_labels=None):
        # If the input array is an MMTensor,
        # use its tensor and axis labels (or the provided ones if specified)
        if isinstance(array, MMTensor):
            self.tensor = array.tensor
            self.axis_labels = axis_labels or array.axis_labels
        else:
            self.tensor = array
            self.axis_labels = axis_labels

        # If axis labels are not provided, generate default labels
        if self.axis_labels is None:
            self.axis_labels = [str(n) for n in range(len(self.tensor.shape))]

        # Validate the number of axis labels
        if len(self.axis_labels) != len(self.tensor.shape):
            raise ValueError("The number of axis labels must be equal to the number of axes.")

    def __array__(self):
        r"""Implement the NumPy array interface."""
        return self.tensor

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        r"""
        Implement the NumPy ufunc interface.
        """
        if method == "__call__":
            inputs = [i.tensor if isinstance(i, MMTensor) else i for i in inputs]
            return MMTensor(ufunc(*inputs, **kwargs), self.axis_labels)
        else:
            return NotImplemented(f"Cannot call {method} on {ufunc}.")

    def __mul__(self, other):
        r"""implement the * operator"""
        if isinstance(other, Number):
            return MMTensor(self.tensor * other, self.axis_labels)
        if isinstance(other, MMTensor):
            self.check_axis_labels_match(other)
            return MMTensor(self.tensor * other.tensor, self.axis_labels)
        return NotImplemented(f"Cannot multiply {type(self)} and {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        r"""implement the / operator"""
        if isinstance(other, MMTensor):
            self.check_axis_labels_match(other)
            return MMTensor(self.tensor / other.tensor, self.axis_labels)
        try:
            return MMTensor(self.tensor / other, self.axis_labels)
        except TypeError:
            return NotImplemented(f"Cannot divide {type(self)} by {type(other)}")

    def __add__(self, other):
        r"""implement the + operator"""
        if isinstance(other, MMTensor):
            self.check_axis_labels_match(other)
            return MMTensor(self.tensor + other.tensor, self.axis_labels)
        try:
            return MMTensor(self.tensor + other, self.axis_labels)
        except TypeError:
            return NotImplemented(f"Cannot add {type(self)} and {type(other)}")

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return MMTensor(-self.tensor, self.axis_labels)

    def check_axis_labels_match(self, other):
        r"""
        Check that the axis labels of *this* tensor match those of another tensor.
        """
        if self.axis_labels != other.axis_labels:
            raise ValueError(
                f"Axis labels must match (got {self.axis_labels} and {other.axis_labels})"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        r"""Overload the @ operator to perform tensor contractions."""
        # if not isinstance(other, MMTensor):
        #     raise TypeError(f"Cannot contract with object of type {type(other)}")

        # Find common axis labels
        common_labels = set(self.axis_labels) & set(other.axis_labels)

        if not common_labels:
            raise ValueError("No common axis labels found")

        # Determine the indices to contract along
        left_indices = [self.axis_labels.index(label) for label in common_labels]
        right_indices = [other.axis_labels.index(label) for label in common_labels]

        # Create a list of the new axis labels
        new_axis_labels = [label for label in self.axis_labels if label not in common_labels] + [
            label for label in other.axis_labels if label not in common_labels
        ]

        return MMTensor(
            math.tensordot(self.tensor, other.tensor, axes=(left_indices, right_indices)),
            new_axis_labels,
        )

    def contract(self, relabeling: Optional[List[str]] = None):
        r"""
        Contract *this* tensor along the specified indices using einsum.

        Args:
            relabeling (list[str]): An optional list of new axis labels.
            The tensor is contracted along all groups of axes with matching labels.
        """
        # check that labels are valid
        if relabeling is None:
            relabeling = self.axis_labels
        elif len(relabeling) != len(self.axis_labels):
            raise ValueError("The number of labels must be equal to the number of axes.")

        self.axis_labels = relabeling

        # Find all unique labels but keep the order
        unique_labels = []
        for label in relabeling:
            if label not in unique_labels:
                unique_labels.append(label)
        repeated = [label for label in unique_labels if self.axis_labels.count(label) > 1]

        # Turn labels into consecutive ascii lower-case letters,
        # with same letters corresponding to the same label
        label_map = {label: string.ascii_lowercase[i] for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in self.axis_labels]

        # create einsum string from labels
        einsum_str = "".join(labels)

        # Contract the tensor and assign new axis labels (unique except for the contracted ones)
        return MMTensor(
            math.einsum(einsum_str, self.tensor),
            [label for label in unique_labels if label not in repeated],
        )

    def transpose(self, perm: Union[List[int], List[str]]):
        """Transpose the tensor using a list of axis labels or indices."""
        if set(perm) == set(self.axis_labels):
            perm = [self.axis_labels.index(label) for label in perm]
        return MMTensor(math.transpose(self.tensor, perm), [self.axis_labels[i] for i in perm])

    def reshape(self, shape, axis_labels=None):
        """Reshape the tensor. Allows to change the axis labels."""
        return MMTensor(math.reshape(self.tensor, shape), axis_labels or self.axis_labels)

    def __getitem__(self, indices):
        """Implement indexing into the tensor."""
        indices = indices if isinstance(indices, tuple) else (indices,)
        axis_labels = self.axis_labels.copy()
        offset = 0
        for i, ind in enumerate(indices):
            if isinstance(ind, int):
                axis_labels.pop(i + offset)
                offset -= 1
            elif ind is Ellipsis and i == 0:
                offset = len(self.tensor.shape) - len(indices)
            elif ind is Ellipsis and i > 0:
                break

        return MMTensor(self.tensor[indices], axis_labels)

    def __repr__(self):
        return f"MMTensor({self.tensor}, {self.axis_labels})"

    def __getattribute__(self, name):
        r"""
        Overrides built-in getattribute method for fallback attribute lookup.
        Tries to get attribute from self, then from self.tensor

        Args:
            self (object): instance
            name (str): attribute name

        Returns:
            attribute value or raises AttributeError"""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.tensor, name)
