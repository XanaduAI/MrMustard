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

import json
from pathlib import Path
from pydoc import locate
from uuid import uuid4

import numpy as np

from mrmustard import math


CACHE = Path(__file__).parents[2].absolute() / ".serialize_cache"


def save(cls, arrays=None, **data) -> Path:
    r"""Save a serialized set of data to file for later deserialization."""
    file = CACHE / f"{cls.__qualname__}_{uuid4()}.json"  # random filename
    data["class"] = f"{cls.__module__}.{cls.__qualname__}"

    if arrays:
        if overlap := set(arrays).intersection(set(data)):
            raise ValueError(f"Arrays cannot have the same name as generic data: {overlap}")
        backend = math.backend_name
        npz = file.with_suffix(".npz")
        np.savez(npz, **{k: math.asnumpy(v) for k, v in arrays.items()})
        data["arrays"] = str(npz)
        data["backend"] = backend

    with file.open("w") as f:
        json.dump(data, f)

    return file


def load(file: Path, remove_after=True):
    r"""
    The deserializer entrypoint for MrMustard.

    Args:
        file (Path): The file to load from
        remove_after (Optional[bool]): Once load is complete, delete the saved file
    """
    file = Path(file)
    with file.open() as f:
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
