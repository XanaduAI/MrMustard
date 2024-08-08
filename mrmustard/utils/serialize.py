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

"""A serialization library for MrMustard."""

from contextlib import contextmanager
import json
from pathlib import Path
from pydoc import locate
from uuid import uuid4

import numpy as np

from mrmustard import math, settings


def save(cls: type, arrays=None, **data) -> Path:
    r"""
    Save a serialized set of data to file for later deserialization.

    This function can serialize any object that implements the MrMustard
    serialization interface, which is the following:

    .. code-block:: python

        @classmethod
        def deserialize(cls, data: Dict) -> cls

    Though not required to satisfy the interface, it is conventional for a
    class to implement an instance method called ``serialize`` that takes no
    arguments, calls this function with its data, and returns the Path result.

    Args:
        cls (type): The object type being serialized

    Kwargs:
        arrays (Dict[str, NDArray]): A dict of array-like objects to be serialized
        data (Dict[Any, Any]): A JSON-serializable dict of arbitrary data to save

    Returns:
        Path: the path to the saved object, to be retrieved later with ``load``
    """
    file = settings.CACHE_DIR / f"{cls.__qualname__}_{uuid4().hex}.json"  # random filename
    data["class"] = f"{cls.__module__}.{cls.__qualname__}"

    if arrays:
        if overlap := set(arrays).intersection(set(data)):
            raise ValueError(f"Arrays cannot have the same name as generic data: {overlap}")
        backend = math.backend_name
        npz = file.with_suffix(".npz")
        np.savez(npz, **{k: math.asnumpy(v) for k, v in arrays.items()})
        data["arrays"] = str(npz)
        data["backend"] = backend

    with file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    return file


def load(file: Path, remove_after=True):
    r"""
    The deserializer entrypoint for objects saved with the ``save`` method.

    Args:
        file (Path): The json file to load from
        remove_after (Optional[bool]): Once load is complete, delete the saved file
    """
    file = Path(file)
    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "arrays" in data:
        npz_file = data.pop("arrays")
        if (backend := data.pop("backend")) != math.backend_name:
            raise TypeError(
                f"Data serialized with {backend} backend, cannot deserialize to the currently active {math.backend_name} backend"
            )
        data.update(**{k: math.astensor(v) for k, v in np.load(npz_file).items()})
        if remove_after:
            Path(npz_file).unlink()

    if remove_after:
        file.unlink()
    return locate(data.pop("class")).deserialize(data)


def get_zipfile(name: str = None) -> Path:
    r"""
    Returns a randomly named zipfile in the cache folder. If a name is
    provided, returns a Path for that base name in the cache folder.
    """
    return settings.CACHE_DIR / (name or f"collection_{uuid4().hex}.zip")


@contextmanager
def cache_subdir(name=None):
    r"""Context manager to have calls to ``save`` write to a cache subdirectory."""
    old_cache = settings.CACHE_DIR
    settings.CACHE_DIR = old_cache / (name or f"subcache_{uuid4().hex}")
    yield settings.CACHE_DIR
    settings.CACHE_DIR = old_cache
