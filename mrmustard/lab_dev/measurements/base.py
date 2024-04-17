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

"""
This module contains the base classes for the available measurements.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mrmustard.physics.representations import Fock
from ..states import Ket
from ..circuit_components import CircuitComponent

__all__ = ["Measurement", "Detector"]


class Measurement(CircuitComponent):
    r"""
    Base class for all measurements.
    """


class Detector(Measurement):
    r"""
    Base class for all detectors.

    Detectors are specified by a sequence of ``Ket``\s
    :math:`|K_0\rangle, \ldots,|K_N\rangle` referred to as "measurement operators."
    Their internal representation can be of type ``Bargmann`` or ``Fock``, and is obtained by
    stacking the representations of each :math:`|K_i\rangle\langle K_i|`.

    When a detector is applied to a state :math:`\rho`) via ``>>``, the resulting component is a
    ``DM``. Its representation contains ``N`` batches, and could equivalently (but less
    efficiently) be obtained by stacking the representations of all the components of the type
    :math:`\langle K_i|\rho|K_i\rangle`.

    .. code-block::

        >>> from mrmustard.lab_dev import *
        >>> from mrmustard import settings
        >>> import numpy as np

        >>> settings.AUTOCUTOFF_MAX_CUTOFF = 10

        >>> # the measurement operators of a PNR detector with cutoff at n=4
        >>> meas_op = [Number([0], n, 4) for n in range(5)]

        >>> # apply the detector to mode `0` of a two-mode Coherent state
        >>> result = Coherent([0, 1], x=1) >> Detector("my_pnr", [0], meas_op)
        >>> arr = result.representation.array

        >>> # the resulting component has a five-batch representation
        >>> assert arr.shape == (5, 10, 10)

        >>> # the `i`-th batch could equivalently be obtained projecting the input state into the dual of
        >>> # the `i`-th measurement operatos
        >>> dm = Coherent([0, 1], x=1).dm()
        >>> assert np.allclose(arr[0], (dm >> Number([0], 0, 4).dual).representation.array)
        >>> assert np.allclose(arr[1], (dm >> Number([0], 1, 4).dual).representation.array)
        >>> assert np.allclose(arr[2], (dm >> Number([0], 2, 4).dual).representation.array)
        >>> assert np.allclose(arr[3], (dm >> Number([0], 3, 4).dual).representation.array)
        >>> assert np.allclose(arr[4], (dm >> Number([0], 4, 4).dual).representation.array)

    When a detector with measurement operators :math:`|K_0\rangle, \ldots,|K_N\rangle` is applied after
    a transformation :math:`U`, the resulting components is equivalent to a detector with measurement
    operators :math:`U|K_0\rangle, \ldots,U|K_N\rangle`.

    .. code-block::

        >>> # a `Dgate` followed by a PNR detector
        >>> meas_op = [Number([0], n, 4) for n in range(5)]
        >>> d1 = Dgate([0], x=1) >> Detector("my_pnr", [0], meas_op)

        >>> # a detector with displaced measurement operators
        >>> displaced_meas_op = [op >> Dgate([0], x=1).dual for op in meas_op]
        >>> d2 = Detector("my__transformed_pnr", [0], displaced_meas_op)

        >>> # the two components are identical
        >>> assert d1 == d2

    Arguments:
        name: The name of this detector.
        modes: The modes that this detector acts on.
        meas_op: A sequence of ket-like circuit components representing the set of operators
            for this measurement.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        modes: tuple[int, ...] = (),
        meas_op: Optional[Sequence[Ket]] = None,
    ):
        super().__init__(
            name or "D" + "".join(str(m) for m in modes), modes_in_bra=modes, modes_in_ket=modes
        )
        self._meas_op = meas_op
        self._representation = None

    @property
    def meas_op(self):
        return self._meas_op

    @property
    def representation(self):
        if not self._representation:
            array = [k.representation.array for k in self.meas_op]
            self._representation = Fock(array, True).outer(Fock(array, True))
        return self._representation

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Detector")
