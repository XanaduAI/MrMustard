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

from typing import Optional

from ..circuit_components import CircuitComponent

__all__ = ["Measurement", "Detector"]


class Measurement(CircuitComponent):
    r"""
    Base class for all measurements.
    """


class Detector(Measurement):
    r"""
    Base class for all detectors.

    Arguments:
        name: The name of this detector.
        modes: The modes that this detector acts on.
    """

    def __init__(self, name: Optional[str] = None, modes: tuple[int, ...] = ()):
        super().__init__(
            name or "D" + "".join(str(m) for m in modes), modes_in_ket=modes
        )

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
        ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        # add logic to handle ret type
        return ret

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Detector")