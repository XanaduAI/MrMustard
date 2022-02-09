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

import os
import random
from hypothesis import settings, Verbosity
import numpy as np
from tensorflow.python.framework import random_seed as tf_random_seed

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf_random_seed.set_seed(seed_value)

try:
    import torch
except (ImportError, ModuleNotFoundError) as e:
    pass
else:
    torch.manual_seed(seed_value)

# hypothesis configuration -----------------------

settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
