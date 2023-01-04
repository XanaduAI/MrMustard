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

from typing import List
import string
from mrmustard.math import Math

math = Math()


class MMTensor:
    r"""A Mr Mustard tensor (a wrapper around an array that implements the numpy array API)."""

    def __init__(self, array, axis_labels=None):
        if axis_labels is not None:
            if len(axis_labels) != len(array.shape):
                raise ValueError("The number of axis labels must be equal to the number of axes.")
        if isinstance(array, MMTensor):
            self.array = array.array
            self.axis_labels = array.axis_labels
        else:
            self.array = array
            self.axis_labels = axis_labels

    def __array__(self):
        """
        Implement the NumPy array interface.
        """
        return self.array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Implement the NumPy ufunc interface.
        """
        if method == "__call__":
            return MMTensor(ufunc(*inputs, **kwargs), self.axis_labels)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """
        Overload the @ operator to perform tensor contractions.
        """
        if not isinstance(other, MMTensor):
            raise TypeError(f"Cannot contract with object of type {type(other)}")

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
            math.tensordot(self.array, other.array, axes=(left_indices, right_indices)),
            new_axis_labels,
        )

    def contract(self, relabeling: List[str]):
        """
        Contract the tensor along the specified indices using einsum.

        Args:
            relabeling (list[str]): A list of axis labels. The tensor is contracted along all
                groups of axes with matching labels.
        """
        # check that labels are valid
        if not len(relabeling) == len(self.axis_labels):
            raise ValueError("The number of labels must be equal to the number of axes.")

        # Find all unique labels
        unique_labels = set(relabeling)

        # Find the indices of the axes to contract
        indices_to_contract = {}
        for label in unique_labels:
            indices_to_contract[label] = [i for i, l in enumerate(relabeling) if l == label]

        # Create the new axis labels
        new_axis_labels = [label for label in self.axis_labels if label not in unique_labels]

        # create einsum string from indices_to_contract using letters a-z
        einsum_string = "".join([string.ascii_lowercase[i] for i in indices_to_contract])

        # Contract the tensor
        return MMTensor(math.einsum(einsum_string, self.array), new_axis_labels)

    def transpose(self, perm):
        """
        Transpose the tensor. Permutes also the axis labels accordingly.
        """
        return MMTensor(math.transpose(self.array, perm), [self.axis_labels[i] for i in perm])

    def reshape(self, shape, axis_labels=None):
        """
        Reshape the tensor. Allows to change the axis labels.
        """
        return MMTensor(math.reshape(self.array, shape), axis_labels or self.axis_labels)

    def __array__(self):
        """
        Implement the NumPy array interface.
        """
        return self.array

    def __getitem__(self, indices):
        """
        Implement indexing into the tensor.
        """
        if isinstance(indices, tuple):
            axis_labels = []
            for i, ind in enumerate(indices):
                if ind is Ellipsis and i == 0:
                    axis_labels += self.axis_labels[:i]
                elif isinstance(ind, slice):
                    axis_labels += self.axis_labels[i]
                elif ind is Ellipsis and i > 0:
                    axis_labels += self.axis_labels[i:]
                    break
            return MMTensor(self.array[indices], axis_labels)
        else:
            # Index along a single axis and take care of the axis labels
            return MMTensor(
                self.array[indices], self.axis_labels[:indices] + self.axis_labels[indices + 1 :]
            )

    def __repr__(self):
        return f"MMTensor({self.array}, {self.axis_labels})"

    def __getattribute__(self, name):
        """
        Implement the underlying array's methods.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.array, name)
