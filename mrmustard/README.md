## Contribution guidelines

- Always write new [tests](https://github.com/XanaduAI/MrMustard/tree/abstract_backend/mrmustard/tests) if you add new functionality or fix a bug.
- Always type-hint the code (quantum-specific and math-specific type hints (like `Covmat` or `Tensor`) are in `mrmustard._typing`). 
- Always write informative docstrings using the Args/Returns pattern.

You can raise [issues](https://github.com/XanaduAI/MrMustard/issues) to keep track of bugs. We also have a [kanban board](https://github.com/XanaduAI/MrMustard/projects/1) to keep track of projects (you can [make your own](https://github.com/XanaduAI/MrMustard/projects) too).
## Architecture of MrMustard
MrMustard is split into four+1 components:

### 1. Abstract base classes
Except for the `Parametrized` class (which is a custom data class) abstract classes speak quantum mechanics. Abstract classes cannot be instantiated, as they are abstract. They are unaware of how quantum mechanical calculations are performed (that is a job for the plugins).
At the moment the main abstract classes are:

- [`Parametrized`](https://github.com/XanaduAI/MrMustard/mrmustard/abstract/parametrized.py) (functionality for all parametrized objects (Ops, Detectors, etc...))
- [`State`](https://github.com/XanaduAI/MrMustard/mrmustard/abstract/state.py) (abstract parent class for all types of states (Vacuum, Coherent, etc...))
- [`Transformation`](https://github.com/XanaduAI/MrMustard/mrmustard/abstract/transformation.py) (abstract parent class for gates)
- [`Detector`](https://github.com/XanaduAI/MrMustard/mrmustard/abstract/detector.py) (abstract parent class for non-Gaussian detectors (PNRs, Homodyne, etc...))
- [`Optimizer`](https://github.com/XanaduAI/MrMustard/mrmustard/abstract/optimizer.py) (abstract parent class for optimizers)

Unless we need to fix a bug or implement a planned feature, we don't expect to touch these classes very often.

### 2. Concrete classes
Concrete classes are the specific instances of the abstract classes:

- [States](https://github.com/XanaduAI/MrMustard/mrmustard/concrete/states.py) (Vacuum, Coherent, etc...)
- [Transformations](https://github.com/XanaduAI/MrMustard/mrmustard/concrete/ops.py) (like Squeezers, Homodyne detectors, etc...)
- [Detectors](https://github.com/XanaduAI/MrMustard/mrmustard/concrete/measurements.py) (like PNRs, Threshold, etc...)
- [Optimizers](https://github.com/XanaduAI/MrMustard/mrmustard/concrete/optimizers.py) (at the moment only the default optimizer)

To develop additional concrete classes, determine which type of object you are implementing (e.g. a new detector) and add it to the appropriate file, following the conventions you see holding for other similar objects.

To refactor concrete classes, you need to spot a common pattern among all of the siblings and move it to the abstract parent class.

### 3. Plugins
Plugins add functionality to the concrete classes by composition, without committing to a specific numerical library
(which is instead handled by the backend). At the moment the main plugins are:

- [`gaussian`](https://github.com/XanaduAI/MrMustard/mrmustard/plugins/gaussian.py) (phase space functionality)
- [`fock`](https://github.com/XanaduAI/MrMustard/mrmustard/plugins/fockp.py) (Fock space functionality)
- [`train`](https://github.com/XanaduAI/MrMustard/mrmustard/plugins/train.py) (optimization functionality)
- [`graphics`](https://github.com/XanaduAI/MrMustard/mrmustard/plugins/graphics.py) (plots and visualizations)

To develop the existing plugins (or to add new ones), make sure that the backend is used when calling numerical math methods, e.g. `backend.transpose(M)`.

### 4. Backends
The numerical functionality (be it with autodiff or not) is supplied by the backends.

To write a new backend (e.g. at the time of writing we don't have a pytorch backend) 
one needs to create a new directory inside [`backends/`](https://github.com/XanaduAI/MrMustard/mrmustard/backends) for the new backend and implement
a concrete backend according to `BackendInterface` (implemented
in [`backends/__init__.py`](https://github.com/XanaduAI/MrMustard/mrmustard/backends/__init__.py)).

To refactor backends, build functionality using methods in the same backend, then it can be moved in the `BackendInterface`.

New plugins can be created if necessary.

### +1 typing
The typing namespace contains the types specific to Mr Mustard to use when type-annotating, as well as common types from the python typing module.
