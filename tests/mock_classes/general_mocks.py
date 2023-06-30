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
"""
This module provides general mock classes.
"""

all = [ 'MockAnimal', 'MockFruit']

class MockAnimal():
    def __init__(self) -> None:
        self.age = 42
        self.alive = True
        self.colour = 'blue'


class MockFruit():
    def __init__(self) -> None:
        self.creation_date = 1954
        self.at_war = False
        self.flag_colour = 'blue'