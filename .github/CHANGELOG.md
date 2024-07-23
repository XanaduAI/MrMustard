# Current develop

### New features
* Added a new Abc triple for mapping the quadrature representation into Bargmann representation.
  [(#368)](https://github.com/XanaduAI/MrMustard/pull/368)

* Added `sort` function to math backends.

### Breaking changes

### Improvements
* Switch from the `julia` Python package to `juliacall` for easier installation and usage.
  [(#394)](https://github.com/XanaduAI/MrMustard/pull/394)

* Save pytest timings to an S3 bucket for regression analysis. Also add a script to help
  visualize the timing results quickly.
  [(#404)](https://github.com/XanaduAI/MrMustard/pull/404)
  [(#421)](https://github.com/XanaduAI/MrMustard/pull/421)

### Bug fixes
* Fix the bug in the order of indices of the triples for DsMap CircuitComponent.
  [(#385)](https://github.com/XanaduAI/MrMustard/pull/385)

* Ensure all symplectic eigenvalues are returned by the `symplectic_eigenvals` function.

* Ensured support for TensorFlow 2.16+, which would be chosen when installing with `pip`.
  [(#406)](https://github.com/XanaduAI/MrMustard/pull/406)


### Documentation

### Tests

### Contributors
[Samuele Ferracin](https://github.com/SamFerracin)
[Yuan Yao](https://github.com/sylviemonet)
[Filippo Miatto](https://github.com/ziofil)
[Austin Lund](https://github.com/aplund)
[Kasper Nielsen](https://github.com/kaspernielsen96)
[Matthew Silverman](https://github.com/timmysilv)


---

# Release 0.7.3 (current release)

### New features
* Added a function ``to_fock`` to map different representations into Fock representation.
  [(#355)](https://github.com/XanaduAI/MrMustard/pull/355)

* Added a new Abc triple for s-parametrized displacement gate.
  [(#368)](https://github.com/XanaduAI/MrMustard/pull/368)

* Added a function ``real_gaussian_integral`` as helper function to map between different representations.
  [(#371)](https://github.com/XanaduAI/MrMustard/pull/371)

### Breaking changes

### Improvements

### Bug fixes

### Documentation

### Tests

### Contributors
[Samuele Ferracin](https://github.com/SamFerracin),
[Yuan Yao](https://github.com/sylviemonet)
[Filippo Miatto](https://github.com/ziofil)


---

# Release 0.7.1

### New features
* Added functions to generate the ``(A, b, c)`` triples for the Fock-Bargmann representation of
  several states and gates. [(#338)](https://github.com/XanaduAI/MrMustard/pull/338)

* Added support for python 3.11. [(#354)](https://github.com/XanaduAI/MrMustard/pull/354)

### Breaking changes

### Improvements

### Bug fixes
* Fixing a bug in `_transform_gaussian` in transformation.py that modifies the input state's cov and means.
[(#349)](https://github.com/XanaduAI/MrMustard/pull/349)
* Fixing a bug in `general_dyne` in physics/gaussian.py that returns the wrong probability and outcomes with given projection.
[(#349)](https://github.com/XanaduAI/MrMustard/pull/349)

### Documentation

### Tests

### Contributors
[Samuele Ferracin](https://github.com/SamFerracin),
[Yuan Yao](https://github.com/sylviemonet)
[Filippo Miatto](https://github.com/ziofil)


---

# Release 0.7.0

### New features
* Added a new interface for backends, as well as a `numpy` backend (which is now default). Users can run
  all the functions in the `utils`, `math`, `physics`, and `lab` with both backends, while `training`
  requires using `tensorflow`. The `numpy` backend provides significant improvements both in import
  time and runtime. [(#301)](https://github.com/XanaduAI/MrMustard/pull/301)

* Added the classes and methods to create, contract, and draw tensor networks with `mrmustard.math`.
  [(#284)](https://github.com/XanaduAI/MrMustard/pull/284)

* Added functions in physics.bargmann to join and contract (A,b,c) triples.
  [(#295)](https://github.com/XanaduAI/MrMustard/pull/295)

* Added an Ansatz abstract class and PolyExpAnsatz concrete implementation. This is used in the Bargmann representation.
  [(#295)](https://github.com/XanaduAI/MrMustard/pull/295)

* Added `complex_gaussian_integral` method.
  [(#295)](https://github.com/XanaduAI/MrMustard/pull/295)

* Added `Bargmann` representation (parametrized by Abc). Supports all algebraic operations and CV (exact) inner product.
  [(#296)](https://github.com/XanaduAI/MrMustard/pull/296)

### Breaking changes
* Removed circular dependencies by:
  * Removing `graphics.py`--moved `ProgressBar` to `training` and `mikkel_plot` to `lab`.
  * Moving `circuit_drawer` and `wigner` to `physics`.
  * Moving `xptensor` to `math`.
  [(#289)](https://github.com/XanaduAI/MrMustard/pull/289)

* Created `settings.py` file to host `Settings`.
  [(#289)](https://github.com/XanaduAI/MrMustard/pull/289)

* Moved `settings.py`, `logger.py`, and `typing.py` to `utils`.
  [(#289)](https://github.com/XanaduAI/MrMustard/pull/289)

* Removed the `Math` class. To use the mathematical backend, replace
  `from mrmustard.math import Math ; math = Math()` with `import mrmustard.math as math`
  in your scripts.
  [(#301)](https://github.com/XanaduAI/MrMustard/pull/301)

* The `numpy` backend is now default. To switch to the `tensorflow`
  backend, add the line `math.change_backend("tensorflow")` to your scripts.
  [(#301)](https://github.com/XanaduAI/MrMustard/pull/301)

### Improvements

* Calculating Fock representations and their gradients is now more numerically stable (i.e. numerical blowups that
result from repeatedly applying the recurrence relation are postponed to higher cutoff values).
This holds for both the "vanilla strategy" [(#274)](https://github.com/XanaduAI/MrMustard/pull/274) and for the
"diagonal strategy" and "single leftover mode strategy" [(#288)](https://github.com/XanaduAI/MrMustard/pull/288/).
This is done by representing Fock amplitudes with a higher precision than complex128 (countering floating-point errors).
We run Julia code via PyJulia (where Numba was used before) to keep the code fast.
The precision is controlled by `setting settings.PRECISION_BITS_HERMITE_POLY`. The default value is ``128``,
which uses the old Numba code. When setting to a higher value, the new Julia code is run.

* Replaced parameters in `training` with `Constant` and `Variable` classes.
  [(#298)](https://github.com/XanaduAI/MrMustard/pull/298)

* Improved how states, transformations, and detectors deal with parameters by replacing the `Parametrized` class with `ParameterSet`.
  [(#298)](https://github.com/XanaduAI/MrMustard/pull/298)

* Includes julia dependencies into the python packaging for downstream installation reproducibility.
  Removes dependency on tomli to load pyproject.toml for version info, uses importlib.metadata instead.
  [(#303)](https://github.com/XanaduAI/MrMustard/pull/303)
  [(#304)](https://github.com/XanaduAI/MrMustard/pull/304)

* Improves the algorithms implemented in `vanilla` and `vanilla_vjp` to achieve a speedup.
  Specifically, the improved algorithms work on flattened arrays (which are reshaped before being returned) as opposed to multi-dimensional array.
  [(#312)](https://github.com/XanaduAI/MrMustard/pull/312)
  [(#318)](https://github.com/XanaduAI/MrMustard/pull/318)

* Adds functions `hermite_renormalized_batch` and `hermite_renormalized_diagonal_batch` to speed up calculating
  Hermite polynomials over a batch of B vectors.
  [(#308)](https://github.com/XanaduAI/MrMustard/pull/308)

* Added suite to filter undesired warnings, and used it to filter tensorflow's ``ComplexWarning``s.
  [(#332)](https://github.com/XanaduAI/MrMustard/pull/332)

* When re-assigning an immutable setting with the same value, no more error is raised.
  [(#316)](https://github.com/XanaduAI/MrMustard/pull/316)

### Bug fixes

* Added the missing `shape` input parameters to all methods `U` in the `gates.py` file.
[(#291)](https://github.com/XanaduAI/MrMustard/pull/291)
* Fixed inconsistent use of `atol` in purity evaluation for Gaussian states.
[(#294)](https://github.com/XanaduAI/MrMustard/pull/294)
* Fixed the documentations for loss_XYd and amp_XYd functions for Gaussian channels.
[(#305)](https://github.com/XanaduAI/MrMustard/pull/305)
* Replaced all instances of `np.empty` with `np.zeros` to fix instabilities.
[(#309)](https://github.com/XanaduAI/MrMustard/pull/309)
* Fixing a bug where `scipy.linalg.sqrtm` returns an unsupported type.
[(#337)](https://github.com/XanaduAI/MrMustard/pull/337)

### Documentation

### Tests
* Added tests for calculating Fock amplitudes with a higher precision than `complex128`.

### Contributors
[Eli Bourassa](https://github.com/elib20),
[Robbe De Prins](https://github.com/rdprins),
[Samuele Ferracin](https://github.com/SamFerracin),
[Jan Provaznik](https://github.com/jan-provaznik),
[Yuan Yao](https://github.com/sylviemonet)
[Filippo Miatto](https://github.com/ziofil)


---

# Release 0.6.1-post1

### Improvements

* Relaxes dependency versions in pyproject.toml. More specifically, this is to unpin scipy.
  [(#300)](https://github.com/XanaduAI/MrMustard/pull/300)

### Contributors
[Filippo Miatto](https://github.com/ziofil), [Samuele Ferracin](https://github.com/SamFerracin), [Yuan Yao](https://github.com/sylviemonet), [Zeyue Niu](https://github.com/zeyueN)


---
# Release 0.6.0

### New features

* Added a new method to discretize Wigner functions that revolves Clenshaw summations. This method is expected to be fast and
reliable for systems with high number of excitations, for which the pre-existing iterative method is known to be unstable. Users
can select their preferred methods by setting the value of `Settings.DISCRETIZATION_METHOD` to either `interactive` (default) or
`clenshaw`.
  [(#280)](https://github.com/XanaduAI/MrMustard/pull/280)

* Added the `PhaseNoise(phase_stdev)` gate (non-Gaussian). Output is a mixed state in Fock representation. It is not based on a choi operator, but on a nonlinear transformation of the density matrix.
  [(#275)](https://github.com/XanaduAI/MrMustard/pull/275)

### Breaking changes

* The value of `hbar` can no longer be specified outside of `Settings`. All the classes and
  methods that allowed specifying its value as an input now retrieve it directly from `Settings`.
  [(#273)](https://github.com/XanaduAI/MrMustard/pull/273)

* Certain attributes of `Settings` can no longer be changed after their value is queried for the first time.
  [(#273)](https://github.com/XanaduAI/MrMustard/pull/273)

### Improvements

* Calculating Fock representations using the "vanilla strategy" is now more numerically stable (i.e. numerical blowups
that result from repeatedly applying the recurrence relation are now postponed to higher cutoff values).
This is done by representing Fock amplitudes with a higher precision than complex128
(which counters the accumulation of floating-point errors).
We run Julia code via PyJulia (where Numba was used before) to keep the code fast.
[(#274)](https://github.com/XanaduAI/MrMustard/pull/274)

* Tensorflow bumped to v2.14 with poetry installation working out of the box on Linux and Mac.
  [(#281)](https://github.com/XanaduAI/MrMustard/pull/281)

* Incorporated `Tensor` into `Transformation` in order to deal with modes more robustly.
  [(#287)](https://github.com/XanaduAI/MrMustard/pull/287)

* Created the classes `Unitary` and `Channel` to simplify the logic in `Transformation`.
  [(#287)](https://github.com/XanaduAI/MrMustard/pull/287)

### Bug fixes

* Fixed a bug about the variable names in functions (apply_kraus_to_ket, apply_kraus_to_dm, apply_choi_to_ket, apply_choi_to_dm).
  [(#271)](https://github.com/XanaduAI/MrMustard/pull/271)

* Fixed a bug that was leading to an error when computing the Choi representation of a unitary transformation.
  [(#283)](https://github.com/XanaduAI/MrMustard/pull/283)

### Documentation

### Contributors
[Filippo Miatto](https://github.com/ziofil),
[Yuan Yao](https://github.com/sylviemonet),
[Robbe De Prins](https://github.com/rdprins),
[Samuele Ferracin](https://github.com/SamFerracin)
[Zeyue Niu](https://github.com/zeyueN)


---

# Release 0.5.0

### New features

* Optimization callback functionalities has been improved. A dedicated `Callback` class is added which
  is able to access the optimizer, the cost function, the parameters as well as gradients, during the
  optimization. In addition, multiple callbacks can be specified. This opens up the endless possiblities
  of customizing the the optimization progress with schedulers, trackers, heuristics, tricks, etc.
  [(#219)](https://github.com/XanaduAI/MrMustard/pull/219)

* Tensorboard-based optimization tracking is added as a builtin `Callback` class: `TensorboardCallback`.
  It can automatically track costs as well as all trainable parameters during optimization in realtime.
  Tensorboard can be most conveniently viewed from VScode.
  [(#219)](https://github.com/XanaduAI/MrMustard/pull/219)

  ```python
  import numpy as np
  from mrmustard.training import Optimizer, TensorboardCallback

  def cost_fn():
      ...
  def as_dB(cost):
      delta = np.sqrt(np.log(1 / (abs(cost) ** 2)) / (2 * np.pi))
      cost_dB = -10 * np.log10(delta**2)
      return cost_dB

  tb_cb = TensorboardCallback(cost_converter=as_dB, track_grads=True)

  opt = Optimizer(euclidean_lr = 0.001);
  opt.minimize(cost_fn, max_steps=200, by_optimizing=[...], callbacks=tb_cb)

  # Logs will be stored in `tb_cb.logdir` which defaults to `./tb_logdir/...` but can be customized.
  # VScode can be used to open the Tensorboard frontend for live monitoring.
  # Or, in command line: `tensorboard --logdir={tb_cb.logdir}` and open link in browser.
  ```

* Gaussian states support a `bargmann` method for returning the bargmann representation.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

* The `ket` method of `State` now supports new keyword arguments `max_prob` and `max_photons`.
  Use them to speed-up the filling of a ket array up to a certain probability or *total* photon number.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

  ```python
  from mrmustard.lab import Gaussian

  # Fills the ket array up to 99% probability or up to the |0,3>, |1,2>, |2,1>, |3,0> subspace, whichever is reached first.
  # The array has the autocutoff shape, unless the cutoffs are specified explicitly.
  ket = Gaussian(2).ket(max_prob=0.99, max_photons=3)
  ```

* Gaussian transformations support a `bargmann` method for returning the bargmann representation.
  [(#239)](https://github.com/XanaduAI/MrMustard/pull/239)

* BSGate.U now supports method='vanilla' (default) and 'schwinger' (slower, but stable to any cutoff)
  [(#248)](https://github.com/XanaduAI/MrMustard/pull/248)

### Breaking Changes

* The previous `callback` argument to `Optimizer.minimize` is now `callbacks` since we can now pass
  multiple callbacks to it.
  [(#219)](https://github.com/XanaduAI/MrMustard/pull/219)

* The `opt_history` attribute of `Optimizer` does not have the placeholder at the beginning anymore.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

### Improvements

* The math module now has a submodule `lattice` for constructing recurrence relation strategies in the Fock lattice.
  There are a few predefined strategies in `mrmustard.math.lattice.strategies`.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

* Gradients in the Fock lattice are now computed using the vector-jacobian product.
  This saves a lot of memory and speeds up the optimization process by roughly 4x.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

* Tests of the compact_fock module now use hypothesis.
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

* Faster implementation of the fock representation of `BSgate`, `Sgate` and `SqueezedVacuum`, ranging from 5x to 50x.
  [(#239)](https://github.com/XanaduAI/MrMustard/pull/239)

* More robust implementation of cutoffs for States.
  [(#239)](https://github.com/XanaduAI/MrMustard/pull/239)

* Dependencies and versioning are now managed using Poetry.
  [(#257)](https://github.com/XanaduAI/MrMustard/pull/257)

### Bug fixes

* Fixed a bug that would make two progress bars appear during an optimization
  [(#235)](https://github.com/XanaduAI/MrMustard/pull/235)

* The displacement of the dual of an operation had the wrong sign
  [(#239)](https://github.com/XanaduAI/MrMustard/pull/239)

* When projecting a Gaussian state onto a Fock state, the upper limit of the autocutoff now respect the Fock projection.
  [(#246)](https://github.com/XanaduAI/MrMustard/pull/246)

* Fixed a bug for the algorithms that allow faster PNR sampling from Gaussian circuits using density matrices. When the
cutoff of the first detector is equal to 1, the resulting density matrix is now correct.

### Documentation

### Contributors
[Filippo Miatto](https://github.com/ziofil), [Zeyue Niu](https://github.com/zeyueN),
[Robbe De Prins](https://github.com/rdprins), [Gabriele Gullì](https://github.com/ggulli),
[Richard A. Wolf](https://github.com/ryk-wolf)


---

# Release 0.4.1

### New features

### Breaking changes

### Improvements

* Fixed flaky optimization tests and removed tf dependency.
  [(#224)](https://github.com/XanaduAI/MrMustard/pull/224)
  [(#233)](https://github.com/XanaduAI/MrMustard/pull/233)

### Bug fixes

* Unpins package versions in setup.py that got mistakenly pinned in 0.4.0.
  [(#223)](https://github.com/XanaduAI/MrMustard/pull/223)

* fixing a bug with the `Dgate` optimization
  [(#232)](https://github.com/XanaduAI/MrMustard/pull/232)

### Documentation

### Contributors
[Filippo Miatto](https://github.com/ziofil), [Sebastian Duque Mesa](https://github.com/sduquemesa)

---

# Release 0.4.0

### New features

* Ray-based distributed trainer is now added to `training.trainer`. It acts as a replacement
  for `for` loops and enables the parallelization of running many circuits as well as their
  optimizations. To install the extra dependencies: `pip install .[ray]`.
  [(#194)](https://github.com/XanaduAI/MrMustard/pull/194)

  ```python
  from mrmustard.lab import Vacuum, Dgate, Ggate
  from mrmustard.physics import fidelity
  from mrmustard.training.trainer import map_trainer

  def make_circ(x=0.):
      return Ggate(num_modes=1, symplectic_trainable=True) >> Dgate(x=x, x_trainable=True, y_trainable=True)

  def cost_fn(circ=make_circ(0.1), y_targ=0.):
      target = Gaussian(1) >> Dgate(-1.5, y_targ)
      s = Vacuum(1) >> circ
      return -fidelity(s, target)

  # Use case 0: Calculate the cost of a randomly initialized circuit 5 times without optimizing it.
  results_0 = map_trainer(
      cost_fn=cost_fn,
      tasks=5,
  )

  # Use case 1: Run circuit optimization 5 times on randomly initialized circuits.
  results_1 = map_trainer(
      cost_fn=cost_fn,
      device_factory=make_circ,
      tasks=5,
      max_steps=50,
      symplectic_lr=0.05,
  )

  # Use case 2: Run circuit optimization 2 times on randomly initialized circuits with custom parameters.
  results_2 = map_trainer(
      cost_fn=cost_fn,
      device_factory=make_circ,
      tasks=[
          {'x': 0.1, 'euclidean_lr': 0.005, 'max_steps': 50, 'HBAR': 1.},
          {'x': -0.7, 'euclidean_lr': 0.1, 'max_steps': 2, 'HBAR': 2.},
      ],
      y_targ=0.35,
      symplectic_lr=0.05,
      AUTOCUTOFF_MAX_CUTOFF=7,
  )
  ```

* Sampling for homodyne measurements is now integrated in Mr Mustard: when no measurement outcome
  value is specified by the user, a value is sampled from the reduced state probability distribution
  and the conditional state on the remaining modes is generated.
  [(#143)](https://github.com/XanaduAI/MrMustard/pull/143)

  ```python
  import numpy as np
  from mrmustard.lab import Homodyne, TMSV, SqueezedVacuum

  # conditional state from measurement
  conditional_state = TMSV(r=0.5, phi=np.pi)[0, 1] >> Homodyne(quadrature_angle=np.pi/2)[1]

  # measurement outcome
  measurement_outcome = SqueezedVacuum(r=0.5) >> Homodyne()
  ```

* The optimizer `minimize` method now accepts an optional callback function, which will be called
  at each step of the optimization and it will be passed the step number, the cost value,
  and the value of the trainable parameters. The result is added to the `callback_history`
  attribute of the optimizer.
  [(#175)](https://github.com/XanaduAI/MrMustard/pull/175)

* the Math interface now supports linear system solving via `math.solve`.
  [(#185)](https://github.com/XanaduAI/MrMustard/pull/185)

* We introduce the tensor wrapper `MMTensor` (available in `math.mmtensor`) that allows for
  a very easy handling of tensor contractions. Internally MrMustard performs lots of tensor
  contractions and this wrapper allows one to label each index of a tensor and perform
  contractions using the `@` symbol as if it were a simple matrix multiplication (the indices
  with the same name get contracted).
  [(#185)](https://github.com/XanaduAI/MrMustard/pull/185)<br>
  [(#195)](https://github.com/XanaduAI/MrMustard/pull/195)

  ```python
  from mrmustard.math.mmtensor import MMTensor

  # define two tensors
  A = MMTensor(np.random.rand(2, 3, 4), axis_labels=["foo", "bar", "contract"])
  B = MMTensor(np.random.rand(4, 5, 6), axis_labels=["contract", "baz", "qux"])

  # perform a tensor contraction
  C = A @ B
  C.axis_labels  # ["foo", "bar", "baz", "qux"]
  C.shape # (2, 3, 5, 6)
  C.tensor # extract actual result
  ```

* MrMustard's settings object (accessible via `from mrmustard import settings`) now supports
  `SEED` (an int). This will give reproducible results whenever randomness is involved.
  The seed is assigned randomly by default, and it can be reassigned again by setting it to None:
  `settings.SEED = None`. If one desires, the seeded random number generator is accessible directly
  via `settings.rng` (e.g. `settings.rng.normal()`).
  [(#183)](https://github.com/XanaduAI/MrMustard/pull/183)

* The `Circuit` class now has an ascii representation, which can be accessed via the repr method.
  It looks great in Jupyter notebooks! There is a new option at `settings.CIRCUIT_DECIMALS`
  which controls the number of decimals shown in the ascii representation of the gate parameters.
  If `None`, only the name of the gate is shown.
  [(#196)](https://github.com/XanaduAI/MrMustard/pull/196)

* PNR sampling from Gaussian circuits using density matrices can now be performed faster.
  When all modes are detected, this is done by replacing `math.hermite_renormalized` by `math.hermite_renormalized_diagonal`. If all but the first mode are detected,
  `math.hermite_renormalized_1leftoverMode` can be used.
  The complexity of these new methods is equal to performing a pure state simulation.
  The methods are differentiable, so that they can be used for defining a cost function.
  [(#154)](https://github.com/XanaduAI/MrMustard/pull/154)

* MrMustard repo now provides a fully furnished vscode development container and a Dockerfile. To
  find out how to use dev containers for development check the documentation
  [here](https://code.visualstudio.com/docs/devcontainers/containers).
  [(#214)](https://github.com/XanaduAI/MrMustard/pull/214)

### Breaking changes

### Improvements

* The `Dgate` is now implemented directly in MrMustard (instead of on The Walrus) to calculate the
  unitary and gradients of the displacement gate in Fock representation, providing better numerical
  stability for larger cutoff and displacement values.
  [(#147)](https://github.com/XanaduAI/MrMustard/pull/147)
  [(#211)](https://github.com/XanaduAI/MrMustard/pull/211)

* Now the Wigner function is implemented in its own module and uses numba for speed.
  [(#171)](https://github.com/XanaduAI/MrMustard/pull/171)

  ```python
    from mrmustard.utils.wigner import wigner_discretized
    W, Q, P = wigner_discretized(dm, q, p) # dm is a density matrix
  ```

* Calculate marginals independently from the Wigner function thus ensuring that the marginals are
  physical even though the Wigner function might not contain all the features of the state
  within the defined window. Also, expose some plot parameters and return the figure and axes.
  [(#179)](https://github.com/XanaduAI/MrMustard/pull/179)

* Allows for full cutoff specification (index-wise rather than mode-wise) for subclasses
  of `Transformation`. This allows for a more compact Fock representation where needed.
  [(#181)](https://github.com/XanaduAI/MrMustard/pull/181)

* The `mrmustard.physics.fock` module now provides convenience functions for applying kraus
  operators and choi operators to kets and density matrices.
  [(#180)](https://github.com/XanaduAI/MrMustard/pull/180)

  ```python
  from mrmustard.physics.fock import apply_kraus_to_ket, apply_kraus_to_dm, apply_choi_to_ket, apply_choi_to_dm
  ket_out = apply_kraus_to_ket(kraus, ket_in, indices)
  dm_out = apply_choi_to_dm(choi, dm_in, indices)
  dm_out = apply_kraus_to_dm(kraus, dm_in, indices)
  dm_out = apply_choi_to_ket(choi, ket_in, indices)
  ```

* Replaced norm with probability in the repr of `State`. This improves consistency over the
  old behaviour (norm was the sqrt of prob if the state was pure and prob if the state was mixed).
  [(#182)](https://github.com/XanaduAI/MrMustard/pull/182)

* Added two new modules (`physics.bargmann` and `physics.husimi`) to host the functions related
  to those representations, which have been refactored and moved out of `physics.fock`.
  [(#185)](https://github.com/XanaduAI/MrMustard/pull/185)

* The internal type system in MrMustard has been beefed up with much clearer types, like ComplexVector,
  RealMatrix, etc... as well as a generic type `Batch`, which can be parametrized using the other types,
  like `Batch[ComplexTensor]`. This will allow for better type checking and better error messages.
  [(#199)](https://github.com/XanaduAI/MrMustard/pull/199)

* Added multiple tests and improved the use of Hypothesis.
  [(#191)](https://github.com/XanaduAI/MrMustard/pull/191)

* The `fock.autocutoff` function now uses the new diagonal methods for calculating a
  probability-based cutoff. Use `settings.AUTOCUTOFF_PROBABILITY` to set the probability threshold.
  [(#203)](https://github.com/XanaduAI/MrMustard/pull/203)

* The unitary group optimization (for the interferometer) and the orthogonal group optimization
  (for the real interferometer) have been added. The symplectic matrix that describes an
  interferometer belongs to the intersection of the orthogonal group and the symplectic group,
  which is a unitary group, so we needed both.
  [(#208)](https://github.com/XanaduAI/MrMustard/pull/208)

### Bug fixes

* The `Dgate` and the `Rgate` now correctly parse the case when a single scalar is intended
  as the same parameter of a number of gates in parallel.
  [(#180)](https://github.com/XanaduAI/MrMustard/pull/180)

* The trace function in the fock module was giving incorrect results when called with certain
  choices of modes. This is now fixed.
  [(#180)](https://github.com/XanaduAI/MrMustard/pull/180)

* The purity function for fock states no longer normalizes the density matrix before computing
  the purity.
  [(#180)](https://github.com/XanaduAI/MrMustard/pull/180)

* The function `dm_to_ket` no longer normalizes the density matrix before diagonalizing it.
  [(#180)](https://github.com/XanaduAI/MrMustard/pull/180)

* The internal fock representation of states returns the correct cutoffs in all cases
  (solves an issue when a pure dm was converted to ket).
  [(#184)](https://github.com/XanaduAI/MrMustard/pull/184)

* The ray related tests were hanging in github action causing tests to halt and fail.
  Now ray is forced to init with 1 cpu when running tests preventing the issue.
  [(#201)](https://github.com/XanaduAI/MrMustard/pull/201)

* Various minor bug fixes.
  [(#202)](https://github.com/XanaduAI/MrMustard/pull/202)

* Fixed the issue that the optimization of the interferometer was using orthogonal group
  optimization rather than unitary.
  [(#208)](https://github.com/XanaduAI/MrMustard/pull/208)

* Fixes a slicing issue that arises when we compute the fidelity between gaussian and fock states.
  [(#210)](https://github.com/XanaduAI/MrMustard/pull/210)

* The sign of parameters in the circuit drawer are now displayed correctly.
  [(#209)](https://github.com/XanaduAI/MrMustard/pull/209)

* Fixed a bug in the Gaussian state which caused its covariance matrix to be multiplied
  by hbar/2 twice. Adds the argument `modes` to `Ggate`.
  [(#212)](https://github.com/XanaduAI/MrMustard/pull/212)

* Fixes a bug in the cutoffs of the choi operator.
  [(#216)](https://github.com/XanaduAI/MrMustard/pull/216)


### Documentation

### Contributors

This release contains contributions from (in alphabetical order):
[Robbe De Prins](https://github.com/rdprins), [Sebastian Duque Mesa](https://github.com/sduquemesa),
[Filippo Miatto](https://github.com/ziofil), [Zeyue Niu](https://github.com/zeyueN),
[Yuan Yao](https://github.com/sylviemonet)


---

# Release 0.3.0

### New features
* Can switch progress bar on and off (default is on) from the settings via
  `settings.PROGRESSBAR = True/False`.
  [(#128)](https://github.com/XanaduAI/MrMustard/issues/128)

* States in Gaussian and Fock representation now can be concatenated.
  ```python
  from mrmustard.lab.states import Gaussian, Fock
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

* Parameter passthrough allows one to use custom variables and/or functions as parameters.
  For example we can use parameters of other gates:
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

* Adds the new trainable gate `RealInterferometer`: an interferometer that doesn't mix
  the q and p quadratures.
  [(#132)](https://github.com/XanaduAI/MrMustard/pull/132)

* Now marginals can be iterated over:
  ```python
  for mode in state:
    print(mode.purity)
  ```
  [(#140)](https://github.com/XanaduAI/MrMustard/pull/140)

### Breaking changes

* The Parametrized and Training classes have been refactored: now trainable tensors are wrapped
  in an instance of the `Parameter` class. To define a set of parameters do
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

* The Parametrized and Training classes have been refactored. The new training module has been
  added and with it the new `Parameter` class: now trainable tensors are being wrapped
  in an instance of `Parameter`.
  [(#133)](https://github.com/XanaduAI/MrMustard/pull/133),
  patch [(#144)](https://github.com/XanaduAI/MrMustard/pull/144)

* The string representations of the `Circuit` and `Transformation` objects have been improved:
  the `Circuit.__repr__` method now produces a string that can be used to generate a circuit in
  an identical state (same gates and parameters), the `Transformation.__str__` and objects
  inheriting from it now prints the name, memory location of the object as well as the modes
  of the circuit in which the transformation is acting on. The `_markdown_repr_` has been implemented
  and on a jupyter notebook produces a table with valuable information of the Transformation objects.
  [(#141)](https://github.com/XanaduAI/MrMustard/pull/141)

* Add the argument 'modes' to the `Interferometer` operation to indicate which modes
  the Interferometer is applied to.
  [(#121)](https://github.com/XanaduAI/MrMustard/pull/121)

### Bug fixes

* Fixed a bug in the `State.ket()` method. An attribute was called with a typo in its name.
  [(#135)](https://github.com/XanaduAI/MrMustard/pull/135)

* The `math.dagger` function applying the hermitian conjugate to an operator was incorrectly
transposing the indices of the input tensor. Now `math.dagger` appropriately calculates the
Hermitian conjugate of an operator.
  [(#156)](https://github.com/XanaduAI/MrMustard/pull/156)

* The application of a Choi operator to a density matrix was resulting in a transposed dm. Now
the order of the indices in the application of a choi operator to dm and ket is correct.
  [(#188)](https://github.com/XanaduAI/MrMustard/pull/188)

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

* Setting the modes on which detectors and state acts using `modes` kwarg or `__getitem__`
  give consistent results.
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

[Sebastián Duque](https://github.com/sduquemesa), [Theodor Isacsson](https://github.com/thisac/),
[Filippo Miatto](https://github.com/ziofil)


---

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


---

# Release 0.1.0

### New features since last release

* This is the initial public release.

### Contributors

This release contains contributions from (in alphabetical order):

[Sebastián Duque](https://github.com/sduquemesa), [Zhi Han](https://github.com/hanzhihua1),
[Theodor Isacsson](https://github.com/thisac/), [Josh Izaac](https://github.com/josh146),
[Filippo Miatto](https://github.com/ziofil), [Nicolas Quesada](https://github.com/nquesada)
