import jax
import time
import jax.numpy as jnp
from mrmustard import math
from mrmustard.lab_dev import SqueezedVacuum, Attenuator, BSgate
from ..conftest import skip_np_and_tf


def evaluate_circuit(params):
    params = jnp.asarray(params)
    BS_01 = BSgate(
        modes=(0, 1), theta=params[0], phi=params[1], theta_trainable=False, phi_trainable=False
    )
    BS_12 = BSgate(
        modes=(1, 2), theta=params[2], phi=params[3], theta_trainable=False, phi_trainable=False
    )
    att = Attenuator(modes=(0, 1, 2), transmissivity=params[4], transmissivity_trainable=False)
    initial_state = SqueezedVacuum(
        modes=(0, 1, 2), r=params[5], phi=params[6], r_trainable=False, phi_trainable=False
    )
    state_out = initial_state >> BS_01 >> BS_12 >> att
    output_fock_state = state_out.fock_array(shape=(20, 5, 5, 20, 5, 5))
    marginal = output_fock_state[:, 4, 4, :, 4, 4]
    return math.real(math.trace(marginal))


def test_jit_complete_circuit():
    r"""Tests if entire circuit with component definitions can be jitted."""
    skip_np_and_tf()

    unjitted_evaluate_circuit = evaluate_circuit
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (7,), minval=0.1, maxval=0.8)
        _ = unjitted_evaluate_circuit(params)
    end_time = time.time()
    unjitted_routine_time = end_time - start_time

    math.JIT_FLAG = True  # turns off checks on parameter values
    jitted_evaluate_circuit = jax.jit(evaluate_circuit)
    _ = jitted_evaluate_circuit(jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (7,), minval=0.1, maxval=0.8)
        _ = jitted_evaluate_circuit(params)
    end_time = time.time()
    jitted_routine_time = end_time - start_time

    print(f"Jitted routine time: {jitted_routine_time}")
    print(f"Unjitted routine time: {unjitted_routine_time}")

    assert (
        jitted_routine_time < unjitted_routine_time
    ), "Jitting should be make circuit evaluation faster."


def test_jit_circuit_with_parameters():
    skip_np_and_tf()

    initial_state = SqueezedVacuum(
        modes=(0, 1, 2), r=0.5, phi=0.5, r_trainable=True, phi_trainable=True
    )
    BS_01 = BSgate(modes=(0, 1), theta=0.5, phi=0.5, theta_trainable=True, phi_trainable=True)
    BS_12 = BSgate(modes=(1, 2), theta=0.5, phi=0.5, theta_trainable=True, phi_trainable=True)
    att = Attenuator(modes=(0, 1, 2), transmissivity=0.5, transmissivity_trainable=True)

    def evaluate_parameters(params):
        BS_01.parameters.all_parameters["theta"].value = params[0]
        BS_01.parameters.all_parameters["phi"].value = params[1]
        BS_12.parameters.all_parameters["theta"].value = params[2]
        BS_12.parameters.all_parameters["phi"].value = params[3]
        state_out = initial_state >> BS_01 >> BS_12 >> att
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

    math.JIT_FLAG = True  # turns off checks on parameter values
    jitted_evaluate_parameters = jax.jit(evaluate_parameters)
    _ = jitted_evaluate_parameters(jnp.array([0.5, 0.5, 0.5, 0.5]))
    start_time = time.time()
    for k in range(100):
        rng = jax.random.PRNGKey(k)
        params = jax.random.uniform(rng, (4,), minval=0.1, maxval=0.8)
        _ = jitted_evaluate_parameters(params)
    end_time = time.time()
    jitted_routine_time = end_time - start_time

    print(f"Jitted routine time: {jitted_routine_time}")
    print(f"Unjitted routine time: {unjitted_routine_time}")

    assert (
        jitted_routine_time < unjitted_routine_time
    ), "Jitting should be make circuit evaluation faster."
