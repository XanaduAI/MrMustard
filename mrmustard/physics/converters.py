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
This module contains the functions to convert between different representations.
"""
from typing import Iterable, Union
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard import math, settings
from mrmustard.utils.typing import Matrix, Vector, Scalar
import numpy as np


import numpy as np
import warnings


def to_fock(rep: Representation, cutoffs: Union[int, Iterable[int]] = None):
    r"""This functions takes a Representaion and returns a new Fock Representation object."""
    if isinstance(rep, Bargmann):
        shape_of_cutoffs_without_batch = len(rep.b[0])
        if not cutoffs:
            cutoffs = (settings.AUTOCUTOFF_MAX_CUTOFF,) * shape_of_cutoffs_without_batch
        elif isinstance(cutoffs, int):
            cutoffs = (cutoffs,) * shape_of_cutoffs_without_batch
        else:
            if len(cutoffs) != shape_of_cutoffs_without_batch:
                raise ValueError(
                    f"Given cutoffs ``{cutoffs}`` is incompatible with the representation."
                    cutoffs,
                    "is not compatiable with the object's representation, it should be ",
                    shape_of_cutoffs_without_batch,
                    ".",
                )
        return Fock(
            math.astensor(
                [
                    math.hermite_renormalized(A, b, c, cutoffs)
                    for A, b, c in zip(rep.A, rep.b, rep.c)
                ]
            ),
            batched=True,
        )
    elif isinstance(rep, Fock):
        if not cutoffs:
            warnings.warn("The cutoffs here hasn't been used!")
        return rep
