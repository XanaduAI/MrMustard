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

"""The class representing a Bargmann eigenstate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import bargmann_eigenstate
from .ket import Ket

__all__ = ["BargmannEigenstate"]


class BargmannEigenstate(Ket):
    r"""The Bargmann eigenstate.

    >>> from mrmustard.lab import BargmannEigenstate
    >>> state = BargmannEigenstate(mode=1, alpha=0.1 + 0.5j)
    >>> assert state.modes == (1,)

    Args:
        mode: The mode of the Bargmann eigenstate.
        alpha: The displacement of the state (i.e., the eigen-value).
        name: A name for the state. If not provided, the class name will be used.

    Note:
        The only difference with ``Coherent(mode, alpha)`` is in its `c` parameter (and hence, does not have unit norm).

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = 0 , b = \alpha, c = 1.

    """

    short_name = "Be"

    def __init__(
        self,
        mode: int | tuple[int],
        alpha: complex | Sequence[complex] | Parameter = 0.0 + 0.0j,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (bargmann_eigenstate, ("alpha", "lin_sup"))}
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["alpha"] = Parameter.from_cc_init(alpha, "complex128", f"{self.name}/alpha")
