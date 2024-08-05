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


CACHE = Path(__file__).parents[2].absolute() / ".serialize_cache"


def save(cls, **data) -> Path:
    r"""Save a serialized set of data to file for later deserialization."""
    file = CACHE / f"{cls.__qualname__}_{uuid4()}.json"  # random filename
    data["class"] = f"{cls.__module__}.{cls.__qualname__}"

    # use numpy serialization tools, save filename in json
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    if len(arrays) > 1:  # many arrays, use zipped serialization
        # save to file
        npz = file.with_suffix(".npz")
        np.savez(npz, **arrays)
        # update json
        _ = list(map(data.pop, arrays))
        data["npz"] = str(npz)
    elif len(arrays) == 1:
        # save to file
        npy = file.with_suffix(".npy")
        [(k, arr)] = list(arrays.items())
        np.save(npy, arr)
        # update json
        del data[k]
        data["npy"] = (k, str(npy))

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
    if remove_after:
        file.unlink()

    if "npy" in data:
        (k, npy_file) = data.pop("npy")
        data[k] = np.load(npy_file)
        if remove_after:
            Path(npy_file).unlink()
    elif "npz" in data:
        npz_file = data.pop("npz")
        data.update(**dict(np.load(npz_file)))
        if remove_after:
            Path(npz_file).unlink()

    return locate(data.pop("class")).deserialize(data)
