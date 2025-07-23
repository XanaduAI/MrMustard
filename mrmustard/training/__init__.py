# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The optimizer module contains all logic for parameter and circuit optimization
in Mr Mustard.

The :class:`Optimizer` uses Adam underneath the hood for Euclidean parameters and
a custom Symplectic optimizer for Gaussian gates and states and an Orthogonal
optimizer for interferometers.

We can turn any simulation in Mr Mustard into an optimization by marking which
parameters we wish to be trainable. Let's take a simple example:
Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip on the
many-photon setting.

.. code-block::

    import numpy as np

    from mrmustard import math
    from mrmustard.lab import BSgate, Circuit, S2gate, Vacuum
    from mrmustard.training import Optimizer

    r = np.arcsinh(1.0)
    s2_0 = S2gate(modes=(0,1), r=r, phi=0.0, phi_trainable=True)
    s2_1 = S2gate(modes=(2,3), r=r, phi=0.0, phi_trainable=True)
    bs = BSgate(
            modes=(1, 2),
            theta=np.arccos(np.sqrt(k / (i + k))) + 0.1 * np.random.normal(),
            phi=np.random.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )

    circ = Circuit([s2_0, s2_1, bs])
    state_in = Vacuum(modes=(0,1,2,3))

    i, k = 2, 2
    cutoff = 1 + i + k

    def cost_fn(circ):
        return math.abs((state_in >> circ).ket(cutoffs=[cutoff] * 4)[i, 1, i + k - 1, k]) ** 2

    opt = Optimizer(euclidean_lr=0.01)
    (circ,) =opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)

Then, we can see the optimized value of the parameters, for example

    .. code-block::

        np.cos(circ.components[2].parameters.theta.value) ** 2

"""

from .optimizer import Optimizer as Optimizer
