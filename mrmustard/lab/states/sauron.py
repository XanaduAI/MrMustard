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

from mrmustard import math
from mrmustard.lab.states.ket import Ket
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ..utils import make_parameter


class Sauron(Ket):
    r"""
    The `n`-th Sauron state is an approximation of the `n`-th Fock states using
    a ring of `n+1` coherent states.

    Args:
        mode: The mode of the Sauron state.
        n: The Fock state that is approximated.
        epsilon: The radius of the ring of coherent states, default is 0.1.

    Notes:
        The reference to the Lord of the Rings comes from
        the approximation becoming perfect in the limit for the radius of the ring going
        to zero where vacuum (= darkness) is.
        The formula for the Sauron state as a superposition of coherent states on a ring
        is given in https://arxiv.org/abs/2305.17099:

        .. math::

            |\text{Sauron}(n)\rangle = \frac{1}{\mathcal{N}}\sum_{k=0}^{n} e^{i 2\pi k/(n+1)} |\epsilon e^{2\pi k/(n+1)}\rangle_c,

    .. code-block::

        >>> from mrmustard.lab import Sauron

        >>> psi = Sauron(0, 1)
        >>> assert psi.modes == (0,)
    """

    def __init__(self, mode: int | tuple[int], n: int, epsilon: float = 0.1):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name=f"Sauron-{n}")

        self.parameters.add_parameter(make_parameter(False, n, "n", (None, None), dtype=math.int64))
        self.parameters.add_parameter(
            make_parameter(False, epsilon, "epsilon", (None, None), dtype=math.float64)
        )

        self._ansatz = PolyExpAnsatz.from_function(
            triples.sauron_state_Abc,
            n=self.parameters.n,
            epsilon=self.parameters.epsilon,
        )
        self._wires = Wires(modes_out_ket=set(mode))
        self.ansatz._lin_sup = True
