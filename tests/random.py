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


angle = st.floats(min_value=0, max_value=2 * np.pi)
positive = st.floats(min_value=0, allow_infinity=False, allow_nan=False)
real = st.floats(allow_infinity=False, allow_nan=False)
r = st.floats(
    min_value=0, max_value=0.5, allow_infinity=False, allow_nan=False
)  # reasonable squeezing magnitude
real_not_zero = st.one_of(st.floats(max_value=-0.00001), st.floats(min_value=0.00001))
integer = st.integers(min_value=0, max_value=2 ** 32 - 1)
small_float = st.floats(min_value=-0.1, max_value=0.1, allow_infinity=False, allow_nan=False)
medium_float = st.floats(min_value=-1.0, max_value=1.0, allow_infinity=False, allow_nan=False)
large_float = st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False)
num_modes = st.integers(min_value=0, max_value=10)


@st.composite
def vector(draw, length):
    return draw(
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=length, max_size=length)
    )


# a strategy to produce a list of integers of length num_modes. the integers are all different and between 0 and num_modes
@st.composite
def modes(draw, num_modes):
    return draw(
        st.lists(
            st.integers(min_value=0, max_value=num_modes), min_size=num_modes, max_size=num_modes
        ).filter(lambda x: len(set(x)) == len(x))
    )


def array_of_(strategy, minlen=0, maxlen=None):
    return arrays(dtype=np.float64, shape=(st.integers(minlen, maxlen),), elements=strategy)


def none_or_(strategy):
    return st.one_of(st.just(None), strategy)


angle_bounds = st.tuples(none_or_(angle), none_or_(angle)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)
positive_bounds = st.tuples(none_or_(positive), none_or_(positive)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)
real_bounds = st.tuples(none_or_(real), none_or_(real)).filter(
    lambda t: t[0] < t[1] if t[0] is not None and t[1] is not None else True
)


@st.composite
def random_Rgate(draw, num_modes=None, trainable=False):
    return Rgate(
        angle=draw(angle),
        angle_bounds=draw(angle_bounds),
        angle_trainable=trainable,
    )


@st.composite
def random_Sgate(draw, num_modes=None, trainable=False, small=False):
    return Sgate(
        r=np.abs(draw(small_float)) if small else draw(r),
        phi=draw(angle),
        r_bounds=draw(positive_bounds),
        phi_bounds=draw(angle_bounds),
        r_trainable=trainable,
        phi_trainable=trainable,
    )


@st.composite
def random_Dgate(draw, num_modes=None, trainable=False, small=False):
    if small:
        x = draw(small_float)
        y = draw(small_float)
    else:
        x = draw(medium_float)
        y = draw(medium_float)
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
    return Interferometer(num_modes=num_modes, orthogonal_trainable=trainable)


@st.composite
def random_Ggate(draw, num_modes, trainable=False):
    displacement = vector(2 * num_modes)
    return Ggate(
        num_modes=num_modes,
        displacement=draw(displacement),
        displacement_trainable=trainable,
    )


@st.composite
def single_mode_unitary(draw, small=False):
    return draw(
        st.one_of(random_Rgate(1), random_Sgate(1, small=small), random_Dgate(1, small=small))
    )


@st.composite
def two_mode_gate(draw):
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
    r = array_of_(small_float.filter(lambda r: r > 0.0), num_modes, num_modes)
    phi_ = array_of_(angle, num_modes, num_modes)
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
    r = array_of_(medium_float.filter(lambda r: r > 0.0), 2, 2)
    phi = array_of_(angle, 2, 2)
    return TMSV(r=draw(r), phi=draw(phi))


@st.composite
def thermal(draw, num_modes):
    n_mean = array_of_(medium_float.filter(lambda r: r > 0.0), num_modes, num_modes)
    return Thermal(n_mean=draw(n_mean))


@st.composite
def default_state(draw, num_modes):
    return draw(
        st.one_of(
            squeezed_vacuum(num_modes),
            displacedsqueezed(num_modes),
            coherent(num_modes),
            tmsv(num_modes),
            thermal(num_modes),
        )
    )


@st.composite
def default_pure_state(draw, num_modes):
    return draw(
        st.one_of(
            squeezed_vacuum(num_modes),
            displacedsqueezed(num_modes),
            coherent(num_modes),
            tmsv(num_modes),
        )
    )


@st.composite
def pure_state(draw, num_modes=1, small=False):
    S = draw(random_Sgate(num_modes, small=small))
    I = draw(random_Interferometer(num_modes))
    D = draw(random_Dgate(num_modes, small=small))
    return Vacuum(num_modes) >> S >> I >> D
