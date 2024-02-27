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
from typing import Iterable, Union, Optional
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard import math, settings


def to_fock(rep: Representation, cutoffs: Optional[Union[int, Iterable[int]]] = None):
    r"""A function to map ``Representation``s to ``Fock`` representations.

    Note that if the rep is ``Fock``, this function maps nothing but return the original rep object directly.

    Args:
        rep: the orginal representation of the object.
        cutoffs: the cutoffs of the transformed Fock representation object.

    Raises:
        ValueError: If the size of the cutoffs given is not compatible with the representation.

    Returns:
        A ``Fock`` representation object.

    .. code-block::

    >>> import numpy as np
    >>> from mrmustard.physics.converters import to_fock
    >>> from mrmustard.physics.representations import Bargmann

    >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> b = np.array([1.0, 1.0])
    >>> c = np.array(1.0)

    >>> bargmann = Bargmann(A, b, c)
    >>> fock = to_fock(bargmann_test).array

    >>> # show the final fock array
    >>> fock.array

    """
    if isinstance(rep, Bargmann):
        len_cutoff = len(rep.b[0])

        if not cutoffs:
            cutoffs = settings.AUTOCUTOFF_MAX_CUTOFF

        cutoffs = (cutoffs,) * len_cutoff if isinstance(cutoffs, int) else cutoffs

        if len_cutoff != len(cutoffs):
            raise ValueError(
                f"Given cutoffs ``{cutoffs}`` is incompatible with the representation."
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
        return rep
