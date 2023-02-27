# Release 0.4.0 (development release)

### New features

### Breaking changes

### Improvements

### Bug fixes
* The unitary group optimization of the interferometer have been added. The symplectic matrix that describes an interferometer belongs to the intersection of the orthogonal group and the symplectic group, which is a unitary group. It fixed the issue that the optimization of the interferometer was orthogonal group optimization. [(#173)](https://github.com/XanaduAI/MrMustard/pull/173)

### Documentation

### Contributors

This release contains contributions from (in alphabetical order):

[Yuan Yao](https://github.com/sylviemonet)

---

# Release 0.3.0 (current release)

### New features
* Can switch progress bar on and off (default is on) from the settings via `settings.PROGRESSBAR = True/False`.
  [(#128)](https://github.com/XanaduAI/MrMustard/issues/128)

* States in Gaussian and Fock representation now can be concatenated.
  ```python
  from mrmustard.lab.states import Gaussian, Fock'
  from mrmustard.lab.gates import Attenuator

  # concatenate pure states
  fock_state = Fock(4)
  gaussian_state = Gaussian(1)
  pure_state = fock_state & gaussian_state

  # also can concatenate mixed states
  mixed1 = fock_state >> Attenuator(0.8)
  mixed2 = gaussian_state >> Attenuator(0.5)
  mixed_state = mixed1 & mixed2

  mixed_state.dm()
  ```
  [(#130)](https://github.com/XanaduAI/MrMustard/pull/130)

* Parameter passthrough allows one to use custom variables and/or functions as parameters. For example we can use parameters of other gates:
    ```python
    from mrmustard.lab.gates import Sgate, BSgate

    BS = BSgate(theta=np.pi/4, theta_trainable=True)[0,1]
    S0 = Sgate(r=BS.theta)[0]
    S1 = Sgate(r=-BS.theta)[1]

    circ = S0 >> S1 >> BS
    ```
  Another possibility is with functions:
  ```python

  def my_r(x):
      return x**2

  x = math.new_variable(0.5, bounds = (None, None), name="x")

  def cost_fn():
    # note that my_r needs to be in the cost function
    # in order to track the gradient
    S = Sgate(r=my_r(x), theta_trainable=True)[0,1]
    return # some function of S

  opt.Optimize(cost_fn, by_optimizing=[x])
  ```
  [(#131)](https://github.com/XanaduAI/MrMustard/pull/131)

* Adds the new trainable gate `RealInterferometer`: an interferometer that doesn't mix the q and p quadratures.
  [(#132)](https://github.com/XanaduAI/MrMustard/pull/132)

* Now marginals can be iterated over:
  ```python
  for mode in state:
    print(mode.purity)
  ```
  [(#140)](https://github.com/XanaduAI/MrMustard/pull/140)

### Breaking changes

* The Parametrized and Training classes have been refactored: now trainable tensors are wrapped in an instance of the `Parameter` class. To define a set of parameters do
  ```python
  from mrmustard.training import Parametrized

  params = Parametrized(
      magnitude=10, magnitude_trainable=False, magnitude_bounds=None,
      angle=0.1, angle_trainable=True, angle_bounds=(-0.1,0.1)
  )
  ```
  which will automatically define the properties `magnitude` and `angle` on the `params` object.
  To access the backend tensor defining the values of such parameters use the `value` property
  ```python
  params.angle.value
  params.angle.bounds

  params.magnitude.value
  ```

  Gates will automatically be an instance of the `Parametrized` class, for example
  ```python
  from mrmustard.lab import BSgate

  bs = BSgate(theta = 0.3, phi = 0.0, theta_trainable: True)

  # access params
  bs.theta.value
  bs.theta.bounds
  bs.phi.value
  ```
  [(#133)](https://github.com/XanaduAI/MrMustard/pull/133),
  patch [(#144)](https://github.com/XanaduAI/MrMustard/pull/144)
  and [(#158)](https://github.com/XanaduAI/MrMustard/pull/158).

### Improvements

* The Parametrized and Training classes have been refactored. The new training module has been added
  and with it the new `Parameter` class: now trainable tensors are being wrapped in an instance of `Parameter`.
  [(#133)](https://github.com/XanaduAI/MrMustard/pull/133),
  patch [(#144)](https://github.com/XanaduAI/MrMustard/pull/144)

* The string representations of the `Circuit` and `Transformation` objects have been improved:
  the `Circuit.__repr__` method now produces a string that can be used to generate a circuit in
  an identical state (same gates and parameters), the `Transformation.__str__` and objects
  inheriting from it now prints the name, memory location of the object as well as the modes
  of the circuit in which the transformation is acting on. The `_markdown_repr_` has been implemented
  and on a jupyter notebook produces a table with valuable information of the Transformation objects.
  [(#141)](https://github.com/XanaduAI/MrMustard/pull/141)

* Add the argument 'modes' to the `Interferometer` operation to indicate which modes the Interferometer is
  applied to.
  [(#121)](https://github.com/XanaduAI/MrMustard/pull/121)

### Bug fixes
* Fixed a bug in the `State.ket()` method. An attribute was called with a typo in its name.
  [(#135)](https://github.com/XanaduAI/MrMustard/pull/135)

* The `math.dagger` function applying the hermitian conjugate to an operator was incorrectly
transposing the indices of the input tensor. Now `math.dagger` appropriately calculates the
Hermitian conjugate of an operator.
  [(#156)](https://github.com/XanaduAI/MrMustard/pull/156)

### Documentation

* The centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  is now used to style the Sphinx documentation.
  [(#126)](https://github.com/XanaduAI/MrMustard/pull/126)

* The documentation now contains the `mm.training` section. The optimization examples on the README
  and Basic API Reference section have been updated to use the latest API.
  [(#133)](https://github.com/XanaduAI/MrMustard/pull/133)

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Sebastian Duque Mesa](https://github.com/sduquemesa), [Filippo Miatto](https://github.com/ziofil), [Yuan Yao](https://github.com/sylviemonet)



---

# Release 0.2.0

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
