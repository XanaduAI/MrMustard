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

"""Test the MrMustard serialization library."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mrmustard.utils.serialize import save, load


@dataclass
class Dummy:
    """A dummy class for testing."""

    foo: int
    bar: str

    @classmethod
    def deserialize(cls, data):
        return cls(**data)


@dataclass
class DummyOneNP:
    """A dummy class with numpy data."""

    name: str
    array: np.ndarray

    @classmethod
    def deserialize(cls, data):
        return cls(**data)


@dataclass
class DummyTwoNP:
    """Another dummy class with more numpy data."""

    name: str
    array1: np.ndarray
    array2: np.ndarray

    @classmethod
    def deserialize(cls, data):
        return cls(**data)


@pytest.fixture()
def cache_dir(tmpdir, monkeypatch):
    cache = Path(tmpdir)
    monkeypatch.setattr("mrmustard.utils.serialize.CACHE", cache)
    return cache


def test_basic(cache_dir):
    """Test basic save and load functionality."""
    path = save(Dummy, foo=5, bar="hello")
    assert path.exists() and path.parent == cache_dir
    assert load(path) == Dummy(foo=5, bar="hello")
    assert not list(cache_dir.glob("*"))  # removed by load


def test_one_numpy_obj(cache_dir):
    """Test save and load functionality with numpy data."""
    path = save(DummyOneNP, name="foobar", array=np.array([1.1, 2.2]))
    assert path.exists() and path.with_suffix(".npy").exists()
    loaded = load(path)

    assert isinstance(loaded, DummyOneNP)
    assert loaded.name == "foobar"
    assert np.array_equal(loaded.array, np.array([1.1, 2.2]))
    assert not list(cache_dir.glob("*"))


def test_two_numpy_obj(cache_dir):
    """Test save and load functionality with more numpy data."""
    a1 = np.array([1.1, 2.2])
    a2 = np.array([3.3 + 4.4j, 5.5 + 6.6j])
    path = save(DummyTwoNP, name="foobar", array1=a1, array2=a2)
    assert path.exists() and path.with_suffix(".npz").exists()
    loaded = load(path)

    assert isinstance(loaded, DummyTwoNP)
    assert loaded.name == "foobar"
    assert np.array_equal(loaded.array1, a1)
    assert np.array_equal(loaded.array2, a2)
    assert not list(cache_dir.glob("*"))
