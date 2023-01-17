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
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from mrmustard.lab import *
from mrmustard import settings

# numbers
integer32bits = st.integers(min_value=0, max_value=2 ** 31 - 1)
real = st.floats(allow_infinity=False, allow_nan=False)
positive = st.floats(min_value=0, exclude_min=True, allow_infinity=False, allow_nan=False)
negative = st.floats(max_value=0, exclude_max=True, allow_infinity=False, allow_nan=False)
real_not_zero = st.one_of(negative, positive)
small_float = st.floats(min_value=-0.1, max_value=0.1, allow_infinity=False, allow_nan=False)
medium_float = st.floats(min_value=-1.0, max_value=1.0, allow_infinity=False, allow_nan=False)

# physical parameters
num_modes = st.integers(min_value=0, max_value=10)
angle = st.floats(min_value=0, max_value=2 * np.pi)
r = st.floats(
    min_value=0, max_value=1.25, allow_infinity=False, allow_nan=False
)  # reasonable squeezing magnitude


@st.composite
def vector(draw, length):
    r"""Return a vector of length `length`."""
    return draw(arrays(np.float, (length,), elements=st.floats(min_value=-1.0, max_value=1.0)))


@st.composite
def list_of_ints(draw, N):
    r"""Return a list of N unique integers between 0 and N-1."""
    return draw(st.lists(st.integers(min_value=0, max_value=N), min_size=N, max_size=N, unique=True))


def array_of_(strategy, minlen=0, maxlen=100):
    r"""Return a strategy that returns an array of values from `strategy`."""
    return arrays(shape=(st.integers(minlen, maxlen).example(),), elements=strategy, dtype=type(strategy.example()))


def none_or_(strategy):
    r"""Return a strategy that returns either None or a value from `strategy`."""
    return st.one_of(st.just(None), strategy)

# bounds
angle_bounds = st.tuples(none_or_(angle), none_or_(angle)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)
positive_bounds = st.tuples(none_or_(positive), none_or_(positive)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)
real_bounds = st.tuples(none_or_(real), none_or_(real)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)

# gates
@st.composite
def random_Rgate(draw, num_modes=None, trainable=False):
    return Rgate(
        angle=draw(angle),
        angle_bounds=draw(angle_bounds),
        angle_trainable=trainable,
    )


@st.composite
def random_Sgate(draw, num_modes=None, trainable=False):
    return Sgate(
        r=draw(r),
        phi=draw(angle),
        r_bounds=draw(positive_bounds),
        phi_bounds=draw(angle_bounds),
        r_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_Dgate(draw, num_modes=None, trainable=False):
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
def random_S2gate(draw, trainable=False):
    return S2gate(
        r=draw(r),
        phi=draw(angle),
        r_bounds=draw(positive_bounds),
        phi_bounds=draw(angle_bounds),
        r_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_BSgate(draw, trainable=False):
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
    settings.SEED = draw(integer32bits)
    return Interferometer(num_modes=num_modes, orthogonal_trainable=trainable)


@st.composite
def random_Ggate(draw, num_modes, trainable=False):
    settings.SEED = draw(integer32bits)
    return Ggate(num_modes=num_modes, symplectic_trainable=trainable)


@st.composite
def single_mode_unitary_gate(draw):
    return draw(st.one_of(random_Rgate(1), random_Sgate(1), random_Dgate(1)))


@st.composite
def two_mode_unitary_gate(draw):
    return draw(
        st.one_of(
            random_S2gate(),
            random_BSgate(),
            random_MZgate(),
            random_Ggate(num_modes=2),
            random_Interferometer(num_modes=2),
        )
    )


@st.composite
def n_mode_gate(draw, num_modes=None):
    return draw(st.one_of(random_Interferometer(num_modes), random_Ggate(num_modes)))


## states
@st.composite
def squeezed_vacuum(draw, num_modes):
    r = array_of_(r, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    return SqueezedVacuum(r=draw(r), phi=draw(phi))


@st.composite
def displacedsqueezed(draw, num_modes):
    r = array_of_(r, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    x = array_of_(medium_float, num_modes, num_modes)
    y = array_of_(medium_float, num_modes, num_modes)
    return DisplacedSqueezed(r=draw(r), phi=draw(phi), x=draw(x), y=draw(x))


@st.composite
def coherent(draw, num_modes):
    x = array_of_(medium_float, num_modes, num_modes)
    y = array_of_(medium_float, num_modes, num_modes)
    return Coherent(x=draw(x), y=draw(y))


@st.composite
def tmsv(draw):
    return TMSV(r=draw(r), phi=draw(phi))


@st.composite
def thermal(draw, num_modes):
    n_mean = array_of_(r, num_modes, num_modes) # using r here
    return Thermal(n_mean=draw(n_mean))


# generic states
@st.composite
def n_mode_separable_pure_state(draw, num_modes):
    return draw(
        st.one_of(
            squeezed_vacuum(num_modes),
            displacedsqueezed(num_modes),
            coherent(num_modes),
        )
    )


@st.composite
def n_mode_separable_mixed_state(draw, num_modes):
    return draw(
        st.one_of(
            squeezed_vacuum(num_modes) ,
            displacedsqueezed(num_modes),
            coherent(num_modes),
            thermal(num_modes),
        )
    ) >> Attenuator(0.9)


@st.composite
def n_mode_pure_state(draw, num_modes=1):
    S = draw(random_Sgate(num_modes))
    I = draw(random_Interferometer(num_modes))
    D = draw(random_Dgate(num_modes))
    return Vacuum(num_modes) >> S >> I >> D
