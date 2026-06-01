# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import get_args, get_origin

import numpy as np
import pytest

from mrmustard.utils.typing import (
    Array,
    ComplexArray,
    ComplexMatrix,
    ComplexScalar,
    ComplexScalarValue,
    ComplexTensor,
    ComplexVector,
    IntArray,
    IntMatrix,
    IntScalar,
    IntScalarValue,
    IntTensor,
    IntVector,
    Matrix,
    RealArray,
    RealMatrix,
    RealScalar,
    RealScalarValue,
    RealTensor,
    RealVector,
    Scalar,
    ScalarValue,
    Tensor,
    Vector,
)


def _get_ndarray_dtype_family(alias):
    """Return the numpy abstract dtype class from an NDArray-backed alias.

    E.g. ``ComplexVector`` -> ``np.complexfloating``.
    Works for aliases defined as ``type X = NDArray[np.T]`` (two levels of
    ``__value__``: the first unwraps the outer ``type`` statement, the second
    unwraps the intermediate ``*Array`` alias).
    """
    inner = alias.__value__  # e.g. ComplexArray (TypeAliasType)
    ndarray_parameterized = inner.__value__  # e.g. NDArray[np.complexfloating]
    dtype_arg = get_args(ndarray_parameterized)[1]  # np.dtype[np.complexfloating]
    return get_args(dtype_arg)[0]  # np.complexfloating


def _get_scalar_value_types(alias):
    """Return the tuple of concrete / abstract types inside a *ScalarValue alias.

    E.g. ``ComplexScalarValue`` -> ``(complex, np.complexfloating)``.
    """
    return get_args(alias.__value__)


@pytest.mark.parametrize(
    "alias, expected_members",
    [
        (ComplexScalarValue, (complex, np.complexfloating)),
        (RealScalarValue, (float, np.floating)),
        (IntScalarValue, (int, np.signedinteger)),
        (ScalarValue, (complex, float, int, np.number)),
    ],
)
def test_scalar_value_structure(alias, expected_members):
    """*ScalarValue aliases should be unions of Python builtins and numpy abstract types."""
    members = _get_scalar_value_types(alias)
    for expected_member in expected_members:
        assert expected_member in members


@pytest.mark.parametrize(
    "alias, expected_members",
    [
        (ComplexScalar, (ComplexScalarValue, ComplexArray)),
        (RealScalar, (RealScalarValue, RealArray)),
        (IntScalar, (IntScalarValue, IntArray)),
        (Scalar, (ScalarValue, Array)),
    ],
)
def test_scalar_structure(alias, expected_members):
    """*Scalar aliases should be unions of *ScalarValue and *Array."""
    members = get_args(alias.__value__)
    for expected_member in expected_members:
        assert expected_member in members


@pytest.mark.parametrize(
    "alias, expected_dtype",
    [
        (ComplexVector, np.complexfloating),
        (ComplexMatrix, np.complexfloating),
        (ComplexTensor, np.complexfloating),
        (RealVector, np.floating),
        (RealMatrix, np.floating),
        (RealTensor, np.floating),
        (IntVector, np.signedinteger),
        (IntMatrix, np.signedinteger),
        (IntTensor, np.signedinteger),
        (Vector, np.number),
        (Matrix, np.number),
        (Tensor, np.number),
    ],
)
def test_array_structure(alias, expected_dtype):
    """Vector / Matrix / Tensor aliases should be backed by NDArray with the correct dtype."""
    assert get_origin(alias.__value__.__value__) is np.ndarray
    assert _get_ndarray_dtype_family(alias) is expected_dtype
