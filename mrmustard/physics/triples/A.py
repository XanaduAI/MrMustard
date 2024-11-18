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

from .base import TempName

__all__ = ["A"]


class A(TempName):
    r""" """

    def __init__(self, array, poly_shape):
        super().__init__(array)
        self._poly_shape = poly_shape

    @property
    def polynomial_shape(self):
        r""" """
        return self._poly_shape

    def __and__(self, other: A):
        r""" """
        dim_beta1, _ = self.polynomial_shape
        dim_beta2, _ = other.polynomial_shape

        dim_alpha1 = self.array.shape[-1] - dim_beta1
        dim_alpha2 = other.array.shape[-1] - dim_beta2

        A3 = math.block(
            [
                [
                    self.array[:dim_alpha1, :dim_alpha1],
                    math.zeros((dim_alpha1, dim_alpha2), dtype=math.complex128),
                    self.array[:dim_alpha1, dim_alpha1:],
                    math.zeros((dim_alpha1, dim_beta2), dtype=math.complex128),
                ],
                [
                    math.zeros((dim_alpha2, dim_alpha1), dtype=math.complex128),
                    other.array[:dim_alpha2:, :dim_alpha2],
                    math.zeros((dim_alpha2, dim_beta1), dtype=math.complex128),
                    other.array[:dim_alpha2, dim_alpha2:],
                ],
                [
                    self.array[dim_alpha1:, :dim_alpha1],
                    math.zeros((dim_beta1, dim_alpha2), dtype=math.complex128),
                    self.array[dim_alpha1:, dim_alpha1:],
                    math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                ],
                [
                    math.zeros((dim_beta2, dim_alpha1), dtype=math.complex128),
                    other.array[dim_alpha2:, :dim_alpha2],
                    math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                    other.array[dim_alpha2:, dim_alpha2:],
                ],
            ]
        )
        return A(A3, self.polynomial_shape + other.polynomial_shape)

    def __eq__(self, other):
        r""" """

    def __getitem__(self, idx):
        r""" """

    def __mul__(self, other):
        r""" """

    def __neg__(self):
        r""" """

    def __truediv__(self, other):
        r""" """
