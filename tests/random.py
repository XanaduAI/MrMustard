# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mrmustard import settings
from mrmustard.lab import (
    TMSV,
    AdditiveNoise,
    Amplifier,
    Attenuator,
    BSgate,
    Coherent,
    CXgate,
    CZgate,
    Dgate,
    DisplacedSqueezed,
    Ggate,
    Interferometer,
    MZgate,
    Pgate,
    Rgate,
    S2gate,
    Sgate,
    SqueezedVacuum,
    Thermal,
    Vacuum,
)

# numbers
integer32bits = st.integers(min_value=0, max_value=2**31 - 1)
real = st.floats(allow_infinity=False, allow_nan=False)
positive = st.floats(
    min_value=0, exclude_min=True, allow_infinity=False, allow_nan=False
)
negative = st.floats(
    max_value=0, exclude_max=True, allow_infinity=False, allow_nan=False
)
real_not_zero = st.one_of(negative, positive)
small_float = st.floats(
    min_value=-0.1, max_value=0.1, allow_infinity=False, allow_nan=False
)
medium_float = st.floats(
    min_value=-1.0, max_value=1.0, allow_infinity=False, allow_nan=False
)
complex_nonzero = st.complex_numbers(
    allow_infinity=False, allow_nan=False, min_magnitude=1e-9, max_magnitude=1e2
)

# physical parameters
nmodes = st.integers(min_value=1, max_value=10)
angle = st.floats(min_value=0, max_value=2 * np.pi)
r = st.floats(min_value=0, max_value=1.25, allow_infinity=False, allow_nan=False)
prob = st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False)
gain = st.floats(min_value=1, max_value=2, allow_infinity=False, allow_nan=False)

# Complex number strategy
complex_number = st.complex_numbers(
    min_magnitude=1e-9, max_magnitude=1, allow_infinity=False, allow_nan=False
)

# Size strategy
size = st.integers(min_value=1, max_value=9)


def Abc_triple(n: int):
    r"""
    Produces a random ``(A, b, c)`` triple for ``n`` modes.
    """
    min_magnitude = 1e-9
    max_magnitude = 1

    # complex symmetric matrix A
    A = np.random.uniform(
        min_magnitude, max_magnitude, (n, n)
    ) + 1.0j * np.random.uniform(min_magnitude, max_magnitude, (n, n))
    A = 0.5 * (A + A.T)  # make it symmetric

    # complex vector b
    b = np.random.uniform(
        min_magnitude, max_magnitude, (n,)
    ) + 1.0j * np.random.uniform(min_magnitude, max_magnitude, (n,))

    # complex scalar c
    c = np.random.uniform(
        min_magnitude, max_magnitude, (1,)
    ) + 1.0j * np.random.uniform(min_magnitude, max_magnitude, (1,))

    return A, b, c


@st.composite
def vector(draw, length):
    r"""Return a vector of length `length`."""
    return draw(
        arrays(np.float64, (length,), elements=st.floats(min_value=-1.0, max_value=1.0))
    )


@st.composite
def list_of_ints(draw, N):
    r"""Return a list of N unique integers between 0 and N-1."""
    return draw(
        st.lists(
            st.integers(min_value=0, max_value=N), min_size=N, max_size=N, unique=True
        )
    )


@st.composite
def matrix(draw, rows, cols):
    """Return a strategy for generating matrices of shape `rows` x `cols`."""
    elements = st.floats(
        allow_infinity=False, allow_nan=False, max_value=1e10, min_value=-1e10
    )
    return draw(arrays(np.float64, (rows, cols), elements=elements))


@st.composite
def complex_matrix(draw, rows, cols):
    """Return a strategy for generating matrices of shape `rows` x `cols` with complex numbers."""
    max_abs_value = 1e10
    elements = st.complex_numbers(
        min_magnitude=0,
        max_magnitude=max_abs_value,
        allow_infinity=False,
        allow_nan=False,
    )
    return draw(arrays(np.complex, (rows, cols), elements=elements))


@st.composite
def complex_vector(draw, length=None):
    """Return a strategy for generating vectors of length `length` with complex numbers."""
    elements = st.complex_numbers(
        min_magnitude=0, max_magnitude=1, allow_infinity=False, allow_nan=False
    )
    if length is None:
        length = draw(st.integers(min_value=1, max_value=10))
    return draw(arrays(np.complex, (length,), elements=elements))


def array_of_(strategy, minlen=0, maxlen=100):
    r"""Return a strategy that returns an array of values from `strategy`."""
    return arrays(
        shape=(st.integers(minlen, maxlen).example(),),
        elements=strategy,
        dtype=type(strategy.example()),
    )


def none_or_(strategy):
    r"""Return a strategy that returns either None or a value from `strategy`."""
    return st.one_of(st.just(None), strategy)


# bounds
def bounds_check(t):
    return t[0] < t[1] if t[0] is not None and t[1] is not None else True


angle_bounds = st.tuples(none_or_(angle), none_or_(angle)).filter(bounds_check)
positive_bounds = st.tuples(none_or_(positive), none_or_(positive)).filter(bounds_check)
real_bounds = st.tuples(none_or_(real), none_or_(real)).filter(bounds_check)
gain_bounds = st.tuples(none_or_(gain), none_or_(gain)).filter(bounds_check)
prob_bounds = st.tuples(none_or_(prob), none_or_(prob)).filter(bounds_check)


# gates
@st.composite
def random_Rgate(draw, trainable=False):
    r"""Return a random Rgate."""
    return Rgate(
        angle=draw(angle),
        angle_bounds=draw(angle_bounds),
        angle_trainable=trainable,
    )


@st.composite
def random_Sgate(draw, trainable=False):
    r"""Return a random Sgate."""
    return Sgate(
        r=draw(r),
        phi=draw(angle),
        r_bounds=draw(positive_bounds),
        phi_bounds=draw(angle_bounds),
        r_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_Dgate(draw, trainable=False):
    r"""Return a random Dgate."""
    x = draw(small_float)
    y = draw(small_float)
    return Dgate(
        x=x,
        y=y,
        x_bounds=draw(real_bounds),
        y_bounds=draw(real_bounds),
        x_trainable=trainable,
        y_trainable=trainable,
    )


@st.composite
def random_Pgate(draw, trainable=False):
    r"""Return a random Pgate."""
    return Pgate(
        shearing=draw(prob),
        shearing_bounds=draw(prob_bounds),
        shearing_trainable=trainable,
    )


@st.composite
def random_Attenuator(draw, trainable=False):
    r"""Return a random Attenuator."""
    return Attenuator(
        transmissivity=draw(prob),
        transmissivity_bounds=draw(prob_bounds),
        transmissivity_trainable=trainable,
    )


@st.composite
def random_Amplifier(draw, trainable=False):
    r"""Return a random Amplifier."""
    return Amplifier(
        gain=draw(gain),
        gain_bounds=draw(gain_bounds),
        gain_trainable=trainable,
    )


@st.composite
def random_AdditiveNoise(draw, trainable=False):
    r"""Return a random AdditiveNoise."""
    return AdditiveNoise(
        noise=draw(prob),
        noise_bounds=draw(prob_bounds),
        noise_trainable=trainable,
    )


@st.composite
def random_S2gate(draw, trainable=False):
    r"""Return a random S2gate."""
    return S2gate(
        r=draw(r),
        phi=draw(angle),
        r_bounds=draw(positive_bounds),
        phi_bounds=draw(angle_bounds),
        r_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_CXgate(draw, trainable=False):
    r"""Return a random CXgate."""
    return CXgate(
        s=draw(medium_float),
        s_bounds=draw(real_bounds),
        s_trainable=trainable,
    )


@st.composite
def random_CZgate(draw, trainable=False):
    r"""Return a random CZgate."""
    return CZgate(
        s=draw(medium_float),
        s_bounds=draw(real_bounds),
        s_trainable=trainable,
    )


@st.composite
def random_BSgate(draw, trainable=False):
    r"""Return a random BSgate."""
    return BSgate(
        theta=draw(angle),
        phi=draw(angle),
        theta_bounds=draw(angle_bounds),
        phi_bounds=draw(angle_bounds),
        theta_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_MZgate(draw, trainable=False):
    r"""Return a random MZgate."""
    return MZgate(
        phi_a=draw(angle),
        phi_b=draw(angle),
        phi_a_bounds=draw(angle_bounds),
        phi_b_bounds=draw(angle_bounds),
        phi_a_trainable=trainable,
        phi_b_trainable=trainable,
        internal=draw(st.booleans()),
    )


@st.composite
def random_Interferometer(draw, num_modes, trainable=False):
    r"""Return a random Interferometer."""
    settings.SEED = draw(integer32bits)
    return Interferometer(num_modes=num_modes, unitary_trainable=trainable)


@st.composite
def random_Ggate(draw, num_modes, trainable=False):
    r"""Return a random Ggate."""
    settings.SEED = draw(integer32bits)
    return Ggate(num_modes=num_modes, symplectic_trainable=trainable)


@st.composite
def single_mode_unitary_gate(draw):
    r"""Return a random single mode unitary gate."""
    return draw(
        st.one_of(
            random_Rgate(),
            random_Sgate(),
            random_Dgate(),
            random_Pgate(),
            random_Interferometer(num_modes=1),  # like Rgate
        )
    )


@st.composite
def single_mode_cv_channel(draw):
    r"""Return a random single mode unitary gate."""
    return draw(
        st.one_of(
            random_Attenuator(),
            random_Amplifier(),
            random_AdditiveNoise(),
        )
    )


@st.composite
def two_mode_unitary_gate(draw):
    r"""Return a random two mode unitary gate."""
    return draw(
        st.one_of(
            random_S2gate(),
            random_BSgate(),
            random_MZgate(),
            random_CXgate(),
            random_CZgate(),
            random_Ggate(num_modes=2),
            random_Interferometer(num_modes=2),
        )
    )


@st.composite
def n_mode_unitary_gate(draw, num_modes=None):
    r"""Return a random n mode unitary gate."""
    return draw(st.one_of(random_Interferometer(num_modes), random_Ggate(num_modes)))


## states
@st.composite
def squeezed_vacuum(draw, num_modes):
    r"""Return a random squeezed vacuum state."""
    r_vec = array_of_(r, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    return SqueezedVacuum(r=draw(r_vec), phi=draw(phi))


@st.composite
def displacedsqueezed(draw, num_modes):
    r"""Return a random displaced squeezed state."""
    r_vec = array_of_(r, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    x = array_of_(medium_float, num_modes, num_modes)
    y = array_of_(medium_float, num_modes, num_modes)
    return DisplacedSqueezed(r=draw(r_vec), phi=draw(phi), x=draw(x), y=draw(y))


@st.composite
def coherent(draw, num_modes):
    r"""Return a random coherent state."""
    x = array_of_(medium_float, num_modes, num_modes)
    y = array_of_(medium_float, num_modes, num_modes)
    return Coherent(x=draw(x), y=draw(y))


@st.composite
def tmsv(draw, phi):
    r"""Return a random two-mode squeezed vacuum state."""
    return TMSV(r=draw(r), phi=draw(phi))


@st.composite
def thermal(draw, num_modes):
    r"""Return a random thermal state."""
    n_mean = array_of_(r, num_modes, num_modes)  # using r here
    return Thermal(n_mean=draw(n_mean))


# generic states
@st.composite
def n_mode_separable_pure_state(draw, num_modes):
    r"""Return a random n mode separable pure state."""
    return draw(
        st.one_of(
            squeezed_vacuum(num_modes),
            displacedsqueezed(num_modes),
            coherent(num_modes),
        )
    )


@st.composite
def n_mode_separable_mixed_state(draw, num_modes):
    r"""Return a random n mode separable mixed state."""
    attenuator = Attenuator(draw(st.floats(min_value=0.2, max_value=0.9)))
    return (
        draw(
            st.one_of(
                squeezed_vacuum(num_modes),
                displacedsqueezed(num_modes),
                coherent(num_modes),
                thermal(num_modes),
            )
        )
        >> attenuator
    )


@st.composite
def n_mode_pure_state(draw, num_modes=1):
    r"""Return a random n mode pure state."""
    S = draw(random_Sgate(num_modes))
    I = draw(random_Interferometer(num_modes))
    D = draw(random_Dgate(num_modes))
    return Vacuum(num_modes) >> S >> I >> D


@st.composite
def n_mode_mixed_state(draw, num_modes=1):
    r"""Return a random n mode pure state."""
    S = draw(random_Sgate(num_modes))
    I = draw(random_Interferometer(num_modes))
    D = draw(random_Dgate(num_modes))
    return Thermal([0.5] * num_modes) >> S >> I >> D
