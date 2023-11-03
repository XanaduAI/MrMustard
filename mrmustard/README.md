## Contribution guidelines

- Write new [tests](https://github.com/XanaduAI/MrMustard/tree/main/tests) if you add new functionality or fix a bug.
- Type-hint the code (quantum-specific and math-specific type hints (like `Covmat` or `Tensor`) are in `mrmustard.utils.types`).
- Write informative docstrings using the Args/Returns pattern.

You can raise [issues](https://github.com/XanaduAI/MrMustard/issues) to keep track of bugs. We also have a [kanban board](https://github.com/XanaduAI/MrMustard/projects/1) to keep track of projects (you can [make your own](https://github.com/XanaduAI/MrMustard/projects) too).
## Architecture of Mr Mustard
Mr Mustard is split into 3 components:

### 1. [lab](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/lab)
The lab module contains states, gates and detectors. The code in these objects uses the functionality provided by the physics module (see below) and does not care how things are actually computed. The code in the lab module should not use the math module directly.

### 2. [physics](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/physics)
The physics module contains the implementation of all of the functionality used in the lab module. The physics module talks to an interface for a math backend, which is defined in [`math_interface.py`](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/physics/math_interface.py).

### 3. [math](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/math)
The math module contains the concrete implementations of the math interface. At the moment we have [Numpy](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/math/backend_numpy.py) and [Tensorflow](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/math/backend_tensorflow.py).
