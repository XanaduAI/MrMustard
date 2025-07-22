# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the jitting functionality within JAX backend."""

import time

import jax
import jax.numpy as jnp
import pytest

from mrmustard import math
from mrmustard.lab import Attenuator, BSgate, SqueezedVacuum


def evaluate_circuit(params):
    r"""
    Defines and evaluates a sample circuit with the given parameters.
    """
    params = jnp.asarray(params)
    BS_01 = BSgate(
        modes=(0, 1),
        theta=params[0],
        phi=params[1],
        theta_trainable=False,
        phi_trainable=False,
    )
    BS_12 = BSgate(
        modes=(1, 2),
        theta=params[2],
        phi=params[3],
        theta_trainable=False,
        phi_trainable=False,
    )
    att = Attenuator(mode=0, transmissivity=params[4], transmissivity_trainable=False)
    initial_state = SqueezedVacuum(
        mode=0,
        r=params[5],
        phi=params[6],
        r_trainable=False,
        phi_trainable=False,
    )
    state_out = (
        initial_state
        >> initial_state.on(1)
        >> initial_state.on(2)
        >> BS_01
        >> BS_12
        >> att
        >> att.on(1)
        >> att.on(2)
    )
    output_fock_state = state_out.fock_array(shape=(20, 5, 5, 20, 5, 5))
    marginal = output_fock_state[:, 4, 4, :, 4, 4]
    return math.real(math.trace(marginal))


@pytest.mark.requires_backend("jax")
def test_jit_complete_circuit():
    r"""Tests if entire circuit with component definitions can be jitted."""
    unjitted_evaluate_circuit = evaluate_circuit
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (7,), minval=0.1, maxval=0.8)
        _ = unjitted_evaluate_circuit(params)
    end_time = time.time()
    unjitted_routine_time = end_time - start_time

    jitted_evaluate_circuit = jax.jit(evaluate_circuit)
    _ = jitted_evaluate_circuit(jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (7,), minval=0.1, maxval=0.8)
        _ = jitted_evaluate_circuit(params)
    end_time = time.time()
    jitted_routine_time = end_time - start_time

    assert jitted_routine_time < unjitted_routine_time, (
        "Jitting should be make circuit evaluation faster."
    )


@pytest.mark.requires_backend("jax")
def test_jit_circuit_with_parameters():
    r"""Tests if circuit with pre-defined elements can be jitted."""
    initial_state = SqueezedVacuum(mode=0, r=0.5, phi=0.5, r_trainable=True, phi_trainable=True)
    BS_01 = BSgate(modes=(0, 1), theta=0.5, phi=0.5, theta_trainable=True, phi_trainable=True)
    BS_12 = BSgate(modes=(1, 2), theta=0.5, phi=0.5, theta_trainable=True, phi_trainable=True)
    att = Attenuator(mode=0, transmissivity=0.5, transmissivity_trainable=True)

    def evaluate_parameters(params):
        r"""
        Evaluate pre-defined circuit elements with the given parameters.
        """
        BS_01.parameters.all_parameters["theta"].value = params[0]
        BS_01.parameters.all_parameters["phi"].value = params[1]
        BS_12.parameters.all_parameters["theta"].value = params[2]
        BS_12.parameters.all_parameters["phi"].value = params[3]
        state_out = (
            initial_state
            >> initial_state.on(1)
            >> initial_state.on(2)
            >> BS_01
            >> BS_12
            >> att
            >> att.on(1)
            >> att.on(2)
        )
        output_fock_state = state_out.fock_array(shape=(20, 5, 5, 20, 5, 5))
        marginal = output_fock_state[:, 4, 4, :, 4, 4]
        return math.real(math.trace(marginal))

    unjitted_evaluate_parameters = evaluate_parameters
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (4,), minval=0.1, maxval=0.8)
        _ = unjitted_evaluate_parameters(params)
    end_time = time.time()
    unjitted_routine_time = end_time - start_time

    jitted_evaluate_parameters = jax.jit(evaluate_parameters)
    _ = jitted_evaluate_parameters(jnp.array([0.5, 0.5, 0.5, 0.5]))
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (4,), minval=0.1, maxval=0.8)
        _ = jitted_evaluate_parameters(params)
    end_time = time.time()
    jitted_routine_time = end_time - start_time

    assert jitted_routine_time < unjitted_routine_time, (
        "Jitting should be make circuit evaluation faster."
    )
