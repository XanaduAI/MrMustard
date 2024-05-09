# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
The avaliable quantum states.

Mr Mustard supports a variety of built-in states, such as:

    .. code-block::

        >>> from mrmustard.lab_dev.states import Coherent, SqueezedVacuum, Vacuum

        >>> # the vacuum state
        >>> vac = Vacuum(modes=[0])

        >>> # coherent states
        >>> coh = Coherent(modes=[1, 2], x=1, y=[0, 1])

        >>> # squeezed states
        >>> sq = SqueezedVacuum(modes=[3], r=1)

All these states are of one of two types, namely :class:`~mrmustard.lab_dev.states.Ket` or
:class:`~mrmustard.lab_dev.states.DM`.

    .. code-block::

        >>> from mrmustard.lab_dev.states import Coherent
        >>> from mrmustard.lab_dev.wires import Wires

        >>> # coherent states are of type ``Ket``
        >>> ket = Coherent(modes=[1, 2], x=1)
        >>> assert isinstance(ket, Ket)

        >>> # this means that their wires only have support on the ket side, and their internal
        >>> # representations have consistent shapes
        >>> assert ket.wires == Wires(modes_out_ket={1, 2})
        >>> assert ket.representation.A.shape == (1, 2, 2)
        >>> assert ket.representation.b.shape == (1, 2)
        >>> assert ket.representation.c.shape == (1,)

        >>> # they can be easily converted into objects of type ``DM``
        >>> dm = ket.dm()

        >>> # ``DM``'s wires have support on both the ket and bra sides, and their internal
        >>> # representations have a larger shapes
        >>> assert dm.wires == Wires(modes_out_bra={1, 2}, modes_out_ket={1, 2})
        >>> assert dm.representation.A.shape == (1, 4, 4)
        >>> assert dm.representation.b.shape == (1, 4)
        >>> assert dm.representation.c.shape == (1,)

In addition to providing these built-in states, Mr Mustard allows initializing custom
:class:`~mrmustard.lab_dev.states.Ket`\s and :class:`~mrmustard.lab_dev.states.DM`\s with the
desired representation. The snippet belowe shows how to initialize a ``Ket`` from
the Bargmann triple of the squeezed vacuum state. Analogous methods exist to initialize
``Ket``\s and ``DM``\s from Fock arrays and from quadrature.

    .. code-block::

        >>> from mrmustard.lab_dev.states import Ket, SqueezedVacuum, Vacuum
        >>> from mrmustard.lab_dev.transformations import Sgate
        >>> from mrmustard.physics import triples

        >>> # the A, b, c triple of the squeezed vacuum state
        >>> A, b, c = triples.squeezed_vacuum_state_Abc(r=0.8)

        >>> # use the tripls to generate a ``Ket``
        >>> my_ket = Ket.from_bargmann(modes=[0], triple=(A, b, c), name="my_ket")

        >>> assert my_ket == Vacuum([0]) >> Sgate([0], r=0.8)
        >>> assert my_ket == SqueezedVacuum([0], r=0.8)

Every state supports an ``expectation`` method which allows calculating the expectation value of
``Ket``\s, ``DM``\s, and ``Unitary``\s on the given state, as well as other useful methods such
as ``probability`` and ``purity``.

    .. code-block::

        >>> from mrmustard.lab_dev.states import Coherent
        >>> from mrmustard.lab_dev.transformations import Dgate
        >>> import numpy as np

        >>> state = Coherent(modes=[0], x=1)

        >>> ket = state
        >>> dm = state.dm()
        >>> unitary = Dgate([0], x=1)

        >>> assert np.allclose(state.expectation(ket), 1)
        >>> assert np.allclose(state.expectation(dm), 1)
        >>> assert np.allclose(state.expectation(unitary), 0.60653066)

        >>> assert np.allclose(Coherent(modes=[0], x=1).probability, 1)
        >>> assert np.allclose(Coherent(modes=[0], x=1).purity, 1)
"""

from .base import *
from .states import *
