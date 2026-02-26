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
The class representing a coherent state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.lab.states.ket import Ket
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import coherent_state

__all__ = ["Coherent"]


class Coherent(Ket):
    r"""
    The coherent state in Bargmann representation.

    >>> from mrmustard.lab import Coherent, Vacuum, Dgate
    >>> state = Coherent(mode=0, alpha=0.3 + 0.2j)
    >>> assert state == Vacuum(0) >> Dgate(0, alpha=0.3 + 0.2j)

    Args:
        mode: The mode of the coherent state.
        alpha: The `alpha` displacement of the coherent state.

    Returns:
        A ``Ket`` object representing a coherent state.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        coherent state displaced :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \sqrt{2\hbar}[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b=\bar{\alpha}\text{, and }c=\text{exp}\big(-|\bar{\alpha}^2|/2\big).

        Note that vector of means in phase space for a coherent state with parameters ``x,y`` is
        ``np.sqrt(2)*x, np.sqrt(2)*y`` (with units ``settings.HBAR=1``).

    """

    short_name = "Coh"

    def __init__(
        self,
        mode: int,
        alpha: complex | Sequence[complex] | Parameter = 0.0 + 0.0j,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (coherent_state, ("alpha", "lin_sup"))}
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["alpha"] = Parameter.from_cc_init(alpha, "complex128", f"{self.name}/alpha")
