# Structure of MrMustard
MrMustard is split into four components: the concrete classes, the abstract classes, the plugins and the backends.
We also have a utils module, which we plan to deprecate.

Important: always write new tests if you are adding new functionality.

## 1. Abstract base classes
Abstract base classes cannot be instantiated, as they are abstract.
At the moment the main abstract classes are:

- [`Parametrized`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/abstract/parametrized.py) (functionality for all parametrized objects (Ops, Detectors, etc...))
- [`State`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/abstract/state.py) (abstract parent class for all types of states (Vacuum, Coherent, etc...))
- [`Op`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/abstract/op.py) (abstract parent class for gates and Gaussian detectors)
- [`Detector`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/abstract/detector.py) (abstract parent class for non-Gaussian detectors (PNRs, Threshold, etc...))

Unless it's for a bug or a planned feature, we don't expect to touch these classes very often.

## 2. Concrete classes
Concrete classes are the specific:

- [`State`s](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/concrete/states.py) (Vacuum, Coherent, etc...)
- [`Op`s](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/concrete/ops.py) (like Squeezers, Homodyne detectors, etc...)
- [`Detector`s](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/concrete/measurements.py) (like PNRs, Threshold, etc...)

To develop additional concrete classes, determine which type of object you are implementing 
e.g. a new gate) and add it to the appropriate file, 
following the conventions you see holding for other similar objects.

## 3. Plugins
Plugins add functionality without committing to a specific numerical library
(which is instead handled by the backend). At the moment the main plugins are:

- [`SymplecticPlugin`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/plugins/symplecticplugin.py) (phase space functionality)
- [`FockPlugin`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/plugins/fockplugin.py) (Fock space functionality)
- [`TrainPlugin`](https://github.com/XanaduAI/MrMustard/blob/abstract_backend/mrmustard/plugins/trainplugin.py) (optimization functionality)

To develop the existing plugins (or to add new ones) one needs to make sure that only the 
math backend is used when calling numerical math methods.

## 4. Backends
The numerical functionality (be it with autodiff or not) is supplied by the backends.

To write a new backend (e.g. at the time of writing we don't have a pytorch backend) 
ne needs to create a new directory inside `backends/` for the new backend and implement
a concrete interface for low-level math according to the `BackendInterface`.
