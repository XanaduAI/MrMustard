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

__all__ = ["b"]


class b(Coefficient):
    r""" """

    def __and__(self, other: b) -> b:
        r""" """
        dim_beta1, _ = self.num_derived_vars
        dim_beta2, _ = other.num_derived_vars

        dim_alpha1 = self.data.shape[-1] - dim_beta1
        dim_alpha2 = other.data.shape[-1] - dim_beta2

        b1 = self.data
        b2 = other.data
        b3 = math.reshape(
            math.block(
                [
                    [
                        b1[:dim_alpha1],
                        b2[:dim_alpha2],
                        b1[dim_alpha1:],
                        b2[dim_alpha2:],
                    ]
                ]
            ),
            -1,
        )
        num_derived_vars = dim_beta1 * dim_beta2
        return b(b3, num_derived_vars)

    def __mul__(self, other: b) -> b:
        r""" """
        dim_beta1 = self.num_derived_vars
        dim_beta2 = other.num_derived_vars

        dim_alpha1 = self.data.shape[-1] - dim_beta1
        dim_alpha2 = other.data.shape[-1] - dim_beta2
        if dim_alpha1 != dim_alpha2:
            raise TypeError("The dimensionality of the two bs must be the same.")
        dim_alpha = dim_alpha1

        b1 = self.data
        b2 = other.data
        b3 = math.reshape(
            math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]),
            -1,
        )
        num_derived_vars = dim_beta1 * dim_beta2
        return b(b3, num_derived_vars)

    def __truediv__(self, other: b) -> b:
        r""" """
        dim_beta1 = self.num_derived_vars
        dim_beta2 = other.num_derived_vars
        if dim_beta1 == 0 and dim_beta2 == 0:
            dim_alpha1 = self.data.shape[-1] - dim_beta1
            dim_alpha2 = other.data.shape[-1] - dim_beta2
            if dim_alpha1 != dim_alpha2:
                raise TypeError("The dimensionality of the two bs must be the same.")
            dim_alpha = dim_alpha1

            b1 = self.data
            b2 = other.data
            b3 = math.reshape(
                math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]),
                -1,
            )
            return b(b3, 0)
        else:
            raise NotImplementedError("Only implemented for ``num_derived_vars == 0``.")
