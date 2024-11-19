# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains
"""

from __future__ import annotations

from mrmustard import math

from .base import Coefficient

__all__ = ["A"]


class A(Coefficient):
    r""" """

    def __and__(self, other: A) -> A:
        r""" """
        dim_beta1 = self.num_derived_vars
        dim_beta2 = other.num_derived_vars

        dim_alpha1 = self.data.shape[-1] - dim_beta1
        dim_alpha2 = other.data.shape[-1] - dim_beta2

        A3 = math.block(
            [
                [
                    self.data[:dim_alpha1, :dim_alpha1],
                    math.zeros((dim_alpha1, dim_alpha2), dtype=math.complex128),
                    self.data[:dim_alpha1, dim_alpha1:],
                    math.zeros((dim_alpha1, dim_beta2), dtype=math.complex128),
                ],
                [
                    math.zeros((dim_alpha2, dim_alpha1), dtype=math.complex128),
                    other.data[:dim_alpha2:, :dim_alpha2],
                    math.zeros((dim_alpha2, dim_beta1), dtype=math.complex128),
                    other.data[:dim_alpha2, dim_alpha2:],
                ],
                [
                    self.data[dim_alpha1:, :dim_alpha1],
                    math.zeros((dim_beta1, dim_alpha2), dtype=math.complex128),
                    self.data[dim_alpha1:, dim_alpha1:],
                    math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                ],
                [
                    math.zeros((dim_beta2, dim_alpha1), dtype=math.complex128),
                    other.data[dim_alpha2:, :dim_alpha2],
                    math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                    other.data[dim_alpha2:, dim_alpha2:],
                ],
            ]
        )
        num_derived_vars = dim_beta1 * dim_beta2
        return A(A3, num_derived_vars)

    def __mul__(self, other: A) -> A:
        r""" """
        dim_beta1 = self.num_derived_vars
        dim_beta2 = other.num_derived_vars

        dim_alpha1 = self.data.shape[-1] - dim_beta1
        dim_alpha2 = other.data.shape[-1] - dim_beta2
        if dim_alpha1 != dim_alpha2:
            raise TypeError("The dimensionality of the two As must be the same.")
        dim_alpha = dim_alpha1

        A1 = math.cast(self.data, "complex128")
        A2 = math.cast(other.data, "complex128")
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
        num_derived_vars = dim_beta1 * dim_beta2
        return A(A3, num_derived_vars)

    def __truediv__(self, other: A) -> A:
        r""" """
        dim_beta1 = self.num_derived_vars
        dim_beta2 = other.num_derived_vars
        if dim_beta1 == 0 and dim_beta2 == 0:
            dim_alpha1 = self.data.shape[-1] - dim_beta1
            dim_alpha2 = other.data.shape[-1] - dim_beta2
            if dim_alpha1 != dim_alpha2:
                raise TypeError("The dimensionality of the two As must be the same.")
            dim_alpha = dim_alpha1

            A1 = math.cast(self.data, "complex128")
            A2 = -math.cast(other.data, "complex128")
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
            return A(A3, 0)
        else:
            raise NotImplementedError("Only implemented for ``num_derived_vars == 0``.")
