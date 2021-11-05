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
from mrmustard import *


angle = st.floats(min_value=0, max_value=2 * np.pi)
positive = st.floats(min_value=0, allow_infinity=False, allow_nan=False)
real = st.floats(allow_infinity=False, allow_nan=False)
real_not_zero = st.one_of(st.floats(max_value=-0.00001), st.floats(min_value=0.00001))
integer = st.integers(min_value=0, max_value=2 ** 32 - 1)

num_modes = st.integers(min_value=0, max_value=10)
modes = st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=10).filter(lambda x: len(set(x)) == len(x))

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

vector = st.lists(st.floats(min_value=-1, max_value=1), min_size=0, max_size=10)


@st.composite
def random_Rgate(draw, num_modes=None, trainable=False):
    return Rgate(
        modes=draw(modes),
        phi=draw(angle),
        angle_bounds=draw(random_angle_bounds),
        angle_trainable=trainable,
        )

@st.composite
def random_Sgate(draw, num_modes=None):
    return Sgate(
        modes=draw(modes),
        r=draw(positive),
        phi=draw(angle),
        r_bounds=draw(random_positive_bounds),
        phi_bounds=draw(random_angle_bounds),
        r_trainable=draw(st.booleans()),
        phi_trainable=draw(st.booleans()),
    )


@st.composite
def random_Dgate(draw, num_modes=None):
    return Dgate(
        modes=draw(random_modes.filter(lambda v: len(v) == num_modes if num_modes is not None else True)),
        x=draw(real),
        y=draw(real),
        x_bounds=draw(random_real_bounds),
        y_bounds=draw(random_real_bounds),
        x_trainable=draw(st.booleans()),
        y_trainable=draw(st.booleans()),
    )


@st.composite
def random_S2gate(draw):
    return S2gate(
        modes=[draw(random_int), draw(random_int)],
        r=draw(positive),
        phi=draw(angle),
        r_bounds=draw(random_positive_bounds),
        phi_bounds=draw(random_angle_bounds),
        r_trainable=draw(st.booleans()),
        phi_trainable=draw(st.booleans()),
    )


@st.composite
def random_BSgate(draw):
    return BSgate(
        modes=[draw(random_int), draw(random_int)],
        theta=draw(angle),
        phi=draw(angle),
        theta_bounds=draw(random_angle_bounds),
        phi_bounds=draw(random_angle_bounds),
        theta_trainable=draw(st.booleans()),
        phi_trainable=draw(st.booleans()),
    )


@st.composite
def random_MZgate(draw):
    return MZgate(
        modes=[draw(random_int), draw(random_int)],
        phi_a=draw(angle),
        phi_b=draw(angle),
        phi_a_bounds=draw(random_angle_bounds),
        phi_b_bounds=draw(random_angle_bounds),
        phi_a_trainable=draw(st.booleans()),
        phi_b_trainable=draw(st.booleans()),
        internal=draw(st.booleans()),
    )


@st.composite
def random_Interferometer(draw, num_modes=None):
    return Interferometer(modes=draw(random_int_list_modes(num_modes)), orthogonal_trainable=draw(st.booleans()))


@st.composite
def random_Ggate(draw, num_modes):
    displacement = st.one_of(random_vector.filter(lambda v: len(v) == 2 * num_modes), st.just(None))
    return Ggate(
        modes=draw(random_modes.filter(lambda v: len(v) == num_modes if num_modes is not None else True)),
        displacement=draw(displacement),
        displacement_trainable=draw(st.booleans()) if displacement is not None else False,
    )


@st.composite
def random_single_mode_gate(draw, num_modes=None):
    return st.one_of(random_Rgate(num_modes), random_Sgate(num_modes), random_Dgate(num_modes))


@st.composite
def random_two_mode_gate(draw, num_modes=None):
    return st.one_of(random_S2gate(), random_BSgate(), random_MZgate(), random_Ggate(num_modes=2), random_Interferometer(num_modes=2))


@st.composite
def random_n_mode_gate(draw, num_modes=None):
    return st.one_of(random_Interferometer(num_modes), random_Ggate(num_modes))


## states
@st.composite
def random_squeezed_vacuum(draw, num_modes):
    r = array_of_(positive, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    return SqueezedVacuum(r=draw(r), phi=draw(phi))


@st.composite
def random_displacedsqueezed(draw, num_modes):
    r = array_of_(positive, num_modes, num_modes)
    phi = array_of_(angle, num_modes, num_modes)
    x = array_of_(real, num_modes, num_modes)
    y = array_of_(real, num_modes, num_modes)
    return DisplacedSqueezed(r=draw(r), phi=draw(phi), x=draw(x), y=draw(x))


@st.composite
def random_coherent(draw, num_modes):
    x = array_of_(real, num_modes, num_modes)
    y = array_of_(real, num_modes, num_modes)
    return Coherent(x=draw(x), y=draw(y))


@st.composite
def random_TMSV(draw):
    r = array_of_(positive, 2, 2)
    phi = array_of_(angle, 2, 2)
    return TMSV(r=draw(r), phi=draw(phi))


@st.composite
def random_thermal(draw, num_modes):
    n_mean = array_of_(positive, num_modes, num_modes)
    return Thermal(n_mean=draw(n_mean))


@st.composite
def random_default_state(draw, num_modes):
    return st.one_of(
        random_squeezed_vacuum(num_modes),
        random_displacedsqueezed(num_modes),
        random_coherent(num_modes),
        random_TMSV(num_modes),
        random_thermal(num_modes),
    )


@st.composite
def random_default_pure_state(draw, num_modes):
    return st.one_of(
        random_squeezed_vacuum(num_modes), random_displacedsqueezed(num_modes), random_coherent(num_modes), random_TMSV(num_modes)
    )


@st.composite
def random_pure_state(draw, num_modes=None):
    if num_modes is None:
        num_modes = draw(st.integers(min_value=1, max_value=10))
    G = draw(random_Ggate(num_modes))
    D = draw(random_Dgate(num_modes))
    return D(G(Vacuum(1 + max(G._modes))))
