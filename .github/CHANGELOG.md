# Release 0.3.0 (development release)

### New features
* Can switch progress bar on and off (default is on) from the settings via `settings.PROGRESSBAR = True/False`.
  [(#128)](https://github.com/XanaduAI/MrMustard/issues/128)

### Breaking changes

### Improvements

### Bug fixes

### Documentation

* The centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  is now used to style the Sphinx documentation.
  [(#126)](https://github.com/XanaduAI/MrMustard/pull/126)

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Filippo Miatto](https://github.com/ziofil)


---

# Release 0.2.0 (current release)

### New features since last release

* Fidelity can now be calculated between two mixed states.
  [(#115)](https://github.com/XanaduAI/MrMustard/pull/115)

* A configurable logger module is added.
  [(#107)](https://github.com/XanaduAI/MrMustard/pull/107)

  ```python
  from mrmustard.logger import create_logger

  logger = create_logger(__name__)
  logger.warning("Warning message")
  ```

### Improvements

* The tensorflow and torch backend adhere to `MathInterface`.
  [(#103)](https://github.com/XanaduAI/MrMustard/pull/103)

### Bug fixes

* Setting the modes on which detectors and state acts using `modes` kwarg or `__getitem__` give consistent results.
  [(#114)](https://github.com/XanaduAI/MrMustard/pull/114)

* Lists are used instead of generators for indices in fidelity calculations.
  [(#117)](https://github.com/XanaduAI/MrMustard/pull/117)

* A raised `KeyboardInterrupt` while on a optimization loop now stops the execution of the program.
  [#105](https://github.com/XanaduAI/MrMustard/pull/105)

### Documentation

* [Basic API reference](https://mrmustard.readthedocs.io/en/latest/introduction/basic_reference.html)
  is updated to use the latest Mr Mustard API.
  [(#119)](https://github.com/XanaduAI/MrMustard/pull/119)
### Contributors

This release contains contributions from (in alphabetical order):

[Sebastián Duque](https://github.com/sduquemesa), [Theodor Isacsson](https://github.com/thisac/), [Filippo Miatto](https://github.com/ziofil)


# Release 0.1.1

### New features since last release

* `physics.normalize` and `physics.norm` are now available.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* `State` now has a norm property.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Can now override autocutoff in `State` by setting the `cutoffs` argument.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

### Improvements since last release

* Renamed `amplification` argument of `Amplifier` to `gain`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Improved `__repr__` for `State`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Added numba section in about().
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

### Bug fixes

* Renamed "pytorch" to "torch" in `mrmustard.__init__()` so that torch can be imported correctly.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Fixed typos in `State.primal()`, `State.__rmul__()`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Fixed a multimode bug in `PNRDetector.__init__()`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Fixed a bug in normalization of `Fock`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

* Fixed a bug in `physics.fidelity()`.
  [(#97)](https://github.com/XanaduAI/MrMustard/pull/97)

### Contributors

This release contains contributions from (in alphabetical order):

[Sebastián Duque](https://github.com/sduquemesa), [Filippo Miatto](https://github.com/ziofil)


# Release 0.1.0

### New features since last release

* This is the initial public release.

### Contributors

This release contains contributions from (in alphabetical order):

[Sebastián Duque](https://github.com/sduquemesa), [Zhi Han](https://github.com/hanzhihua1),
[Theodor Isacsson](https://github.com/thisac/), [Josh Izaac](https://github.com/josh146),
[Filippo Miatto](https://github.com/ziofil), [Nicolas Quesada](https://github.com/nquesada)
