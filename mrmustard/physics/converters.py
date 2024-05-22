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


def to_fock(
    rep: Representation, shape: Optional[Union[int, Iterable[int]]] = None
) -> Fock:
    r"""A function to map ``Representation``\s to ``Fock`` representations.

    If the given ``rep`` is ``Fock``, this function simply returns ``rep``.

    Args:
        rep: The orginal representation of the object.
        shape: The shape of the returned representation. If ``shape``is given as an ``int``, it is broadcasted
            to all the dimensions. If ``None``, it defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` in
            the settings.

    Raises:
        ValueError: If the size of the shape given is not compatible with the representation.

    Returns:
        A ``Fock`` representation object.

    .. code-block::

        >>> from mrmustard.physics.converters import to_fock
        >>> from mrmustard.physics.representations import Bargmann, Fock
        >>> from mrmustard.physics.triples import displacement_gate_Abc

        >>> bargmann = Bargmann(*displacement_gate_Abc(x=0.1, y=[0.2, 0.3]))
        >>> fock = to_fock(bargmann, shape=10)
        >>> assert isinstance(fock, Fock)

    """
    if isinstance(rep, Bargmann):
        if not shape:
            shape = (settings.AUTOCUTOFF_MAX_CUTOFF,) * rep.ansatz.num_vars
        else:
            shape = (shape,) * rep.ansatz.num_vars if isinstance(shape, int) else shape
        if rep.ansatz.num_vars != len(shape):
            msg = f"Given shape ``{shape}`` has length {len(shape)} which is "
            msg += f"{'less' if len(shape) < rep.ansatz.num_vars else 'more'} than "
            msg += f"the number of variables of this ansatz ({rep.ansatz.num_vars})."
            raise ValueError(msg)

        array = [
            math.hermite_renormalized(A, b, c, shape)
            for A, b, c in zip(rep.A, rep.b, rep.c)
        ]
        fock = Fock(math.astensor(array), batched=True)
        fock._original_bargmann_data = rep.data
        return fock
    return rep
