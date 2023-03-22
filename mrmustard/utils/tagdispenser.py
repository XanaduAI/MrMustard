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

"""
This file contains a singleton class (TagDispenser) that generates unique tags (ints).
"""

from typing import Optional

import numpy as np


class TagDispenser:
    r"""A singleton class that generates unique tags (ints).
    It can be given back tags to reuse them.

    Example:
        >>> dispenser = TagDispenser()
        >>> dispenser.get_tag()
        0
        >>> dispenser.get_tag()
        1
        >>> dispenser.give_back_tag(0)
        >>> dispenser.get_tag()
        0
        >>> dispenser.get_tag()
        2
    """
    _instance = None
    _tags = []
    _counter = 0

    def __new__(cls):
        if TagDispenser._instance is None:
            TagDispenser._instance = object.__new__(cls)
        return TagDispenser._instance

    def get_tag(self) -> int:
        """Returns a new unique tag."""
        if len(self._tags) > 0:
            return self._tags.pop(0)
        else:
            self._counter += 1
            return self._counter - 1

    def give_back_tag(self, *tags: Optional[int]):
        """Give back a tag to the dispenser to be reused. Ignores None tags."""
        for tag in tags:
            if isinstance(tag, int) and tag not in self._tags and tag < self._counter:
                self._tags.append(tag)
            elif tag is None:
                pass
            else:
                raise ValueError(
                    f"Cannot accept tag {tag}: self._tags={self._tags}, self._counter={self._counter}"
                )

    def reset(self):
        """Resets the dispenser."""
        self._tags = []
        self._counter = 0

    def __repr__(self):
        _next = self._tags[0] if len(self._tags) > 0 else self._counter
        return f"TagDispenser(returned={self._tags}, next={_next})"
