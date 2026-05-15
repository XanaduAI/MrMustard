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

r"""
The Sauron state is an approximation of the `n`-th Fock states using a ring of `n+1` coherent states.
"""

from __future__ import annotations

from mrmustard.lab.states.ket import Ket
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import sauron_state


class Sauron(Ket):
    r"""
    The `n`-th Sauron state is an approximation of the `n`-th Fock states using
    a ring of `n+1` coherent states.

    >>> from mrmustard.lab import Sauron
    >>> psi = Sauron(0, 1)
    >>> assert psi.modes == (0,)

    Args:
        mode: The mode of the Sauron state.
        n: The Fock state that is approximated.
        epsilon: The radius of the ring of coherent states, default is 0.1.

    Note:
        The reference to the Lord of the Rings comes from
        the approximation becoming perfect in the limit for the radius of the ring going
        to zero where vacuum (= darkness) is.
        The formula for the Sauron state as a superposition of coherent states on a ring
        is given in https://arxiv.org/abs/2305.17099:

        .. math::

            |\text{Sauron}(n)\rangle = \frac{1}{\mathcal{N}}\sum_{k=0}^{n} e^{i 2\pi k/(n+1)} |\epsilon e^{2\pi k/(n+1)}\rangle_c,
    """

    short_name = "Saur"

    def __init__(self, mode: int | tuple[int], n: int, epsilon: float = 0.1):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (sauron_state, ("n", "epsilon", "lin_sup"))}
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=f"{self.__class__.__name__}-{n}",
        )
        self.parameters["n"] = Parameter.from_cc_init(n, "int64", f"{self.name}/n")
        self.parameters["epsilon"] = Parameter.from_cc_init(
            epsilon, "float64", f"{self.name}/epsilon"
        )
