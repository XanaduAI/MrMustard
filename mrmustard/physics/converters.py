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


def to_fock(rep: Representation, shape: Optional[Union[int, Iterable[int]]] = None) -> Fock:
    r"""A function to map ``Representation``s to ``Fock`` representations.

    If the given ``rep`` is ``Fock``, this function simply returns ``rep``.

    Args:
        rep: The orginal representation of the object.
        shape: The shape of the transformed Fock representation object.

    Raises:
        ValueError: If the size of the shape given is not compatible with the representation.

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
        >>> fock = to_fock(bargmann)
        >>> # One can show the final fock array by using fock.array

    """
    if isinstance(rep, Bargmann):
        len_shape = len(rep.b[0])
        if not shape:
            shape = settings.AUTOCUTOFF_MAX_CUTOFF
        shape = (shape,) * len_shape if isinstance(shape, int) else shape
        if len_shape != len(shape):
            raise ValueError(f"Given shape ``{shape}`` is incompatible with the representation.")
        
        array = [math.hermite_renormalized(A, b, c, shape) for A, b, c in zip(rep.A, rep.b, rep.c)]
        return Fock(math.astensor(array), batched=True)
    return rep
