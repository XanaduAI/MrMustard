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

"""callbacks tests"""

import numpy as np
import tensorflow as tf

from mrmustard import math, settings
from mrmustard.lab.circuit import Circuit
from mrmustard.lab.gates import (
    BSgate,
    S2gate,
)
from mrmustard.lab.states import Vacuum
from mrmustard.training import Optimizer, TensorboardCallback

from ..conftest import skip_np


def test_tensorboard_callback(tmp_path):
    """Tests tensorboard callbacks on hong-ou-mandel optimization."""
    skip_np()

    settings.SEED = 42
    i, k = 2, 3
    r = np.arcsinh(1.0)
    s2_0, s2_1, bs = (
        S2gate(r=r, phi=0.0, phi_trainable=True)[0, 1],
        S2gate(r=r, phi=0.0, phi_trainable=True)[2, 3],
        BSgate(
            theta=np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.rng.normal(),
            phi=settings.rng.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )[1, 2],
    )
    circ = Circuit([s2_0, s2_1, bs])
    state_in = Vacuum(num_modes=4)
    cutoff = 1 + i + k

    free_var = math.new_variable([1.1, -0.2], None, "free_var")

    def cost_fn():
        return tf.abs(
            (state_in >> circ).ket(cutoffs=[cutoff] * 4)[i, 1, i + k - 1, k]
        ) ** 2 + tf.reduce_sum(free_var**2)

    tbcb = TensorboardCallback(
        steps_per_call=2,
        root_logdir=tmp_path,
        cost_converter=np.log10,
        track_grads=True,
    )

    opt = Optimizer(euclidean_lr=0.01)
    opt.minimize(
        cost_fn, by_optimizing=[circ, free_var], max_steps=300, callbacks={"tb": tbcb}
    )

    assert np.allclose(np.cos(bs.theta.value) ** 2, k / (i + k), atol=1e-2)
    assert tbcb.logdir.exists()
    assert len(list(tbcb.writter_logdir.glob("events*"))) > 0
    assert (
        len(opt.callback_history["tb"])
        == (len(opt.opt_history) - 1) // tbcb.steps_per_call
    )
