![Logo](https://github.com/XanaduAI/MrMustard/blob/main/mm_white.png#gh-light-mode-only)
![Logo](https://github.com/XanaduAI/MrMustard/blob/main/mm_dark.png#gh-dark-mode-only)

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Actions Status](https://github.com/XanaduAI/MrMustard/workflows/Tests/badge.svg)](https://github.com/XanaduAI/MrMustard/actions)
[![Python version](https://img.shields.io/pypi/pyversions/mrmustard.svg?style=popout-square)](https://pypi.org/project/MrMustard/)

Mr Mustard is a differentiable simulator with a sophisticated built-in optimizer, that operates seamlessly across phase space and Fock space. It is built on top of an agnostic autodiff interface, to allow for plug-and-play backends (`numpy` by default).

Mr Mustard supports:
- Bargmann representation of all states and channels on arbitrary number of modes with the ability to transform between other representations including:
    - Phase space representation of Gaussian states and Gaussian channels on an arbitrary number of modes
    - Exact Fock representation of any Gaussian circuit and any Gaussian state up to an arbitrary cutoff
- Riemannian optimization on the symplectic group (for Gaussian transformations) and on the unitary group (for interferometers)
- Adam optimizer for euclidean parameters
- Single-mode gates (parallelizable):
    - squeezing, displacement, phase rotation, attenuator, amplifier, additive noise, phase noise
- Two-mode gates:
    - beam splitter, Mach-Zehnder interferometer, two-mode squeezing, CX, CZ, CPHASE
- N-mode gates (with dedicated Riemannian optimization):
    - Interferometer (unitary), RealInterferometer (orthogonal), Ggate (symplectic)
- Single-mode states:
    - Vacuum, Coherent, SqueezedVacuum, Thermal, Number
- Two-mode states:
    - TMSV (two-mode squeezed vacuum)
- N-mode states:
    - GKet
- Photon number moments and entropic measures
- PNR detectors and Threshold detectors with trainable quantum efficiency and dark counts
- Homodyne and PNR measurements
- Composable circuits
- Plug-and-play backends (`numpy` as default)

# The lab module

The lab module contains components you'd find in a lab: states, transformations, measurements, circuits, etc.

States can be used at the beginning of a circuit as well as at the end, in which case a state is interpreted as a measurement (a projection onto that state). Transformations are usually parametrized and map states to states. The action on states is differentiable with respect to the state and to the gate parameters.

## 1. States and Transformations

Here are a few examples of built-in states and transformations:

```python
import numpy as np
from mrmustard.lab import *

vac = Vacuum(modes=(0,1))                      # 2-mode vacuum state
coh = Coherent(mode=0, x=0.1, y=-0.4)          # coh state |alpha> with alpha = 0.1 - 0.4j
sq  = SqueezedVacuum(mode=0, r=0.5)            # squeezed vacuum state
gket  = GKet(modes=(0,1))                      # 2-mode Gaussian state
num_4 = Number(mode=0, n=4)                    # number state |4>

D  = Dgate(mode=0, x=1.0, y=-0.4)              # Displacement by 1.0 along x and -0.4 along y
S  = Sgate(mode=0, r=0.5)                      # Squeezer with r=0.5
R  = Rgate(mode=0, angle=0.3)                  # Phase rotation by 0.3
A  = Amplifier(mode=0, gain=2.0)               # noisy amplifier with 200% gain
L  = Attenuator(mode=0, transmissivity=0.5)    # pure loss channel with 50% transmissivity

BS = BSgate(modes=(0,1), theta=np.pi/4)        # 50/50 beam splitter
S2 = S2gate(modes=(0,1), r=0.5)                # two-mode squeezer
MZ = MZgate(modes=(0,1), phi_a=0.3, phi_b=0.1) # Mach-Zehnder interferometer
I  = Interferometer(modes = (0,1,2,3))         # 4-mode interferometer
```

States can be in linear superpositions of eachother via the addition operator:

```python
cat = (Coherent(mode=0, x=2) + Coherent(mode=0, x=-2)).normalize() # normalized cat state
```

The `repr` of single-mode states shows the Wigner function:
<img width="571" alt="Screen Shot 2021-12-06 at 1 31 17 PM" src="https://user-images.githubusercontent.com/8944955/144902008-8d26d59c-8600-4391-9144-ffcc1b2215c2.png">


States can be contracted with transformations via the right shift operator:

```python
cat >> Sgate(mode=0, r=0.5) >> Dgate(mode=0, x=0.01, y=-0.1)  # squeezed displaced cat
```
<img width="479" alt="Screen Shot 2021-12-07 at 2 03 14 PM" src="https://user-images.githubusercontent.com/8944955/145090219-298ca2ab-92e9-4ac2-beab-33ee33770fb2.png">


States can be measured via contraction with a dual state.

```python
cat >> Number(mode=0, n=3).dual # equivalent to a PNR measurement of 4 photons
```


States can also generate samples via the built-in sampler.

```python
pnr = PNRSampler(cutoff=100)
pnr_samples = pnr.sample(state=cat, n_samples=100, seed=None)

homodyne = HomodyneSampler(phi=0, bounds=(-10,10), num=1000)
homodyne_samples = homodyne.sample(state=cat, n_samples=100, seed=None)
```


## 2. Circuits

More advanced use cases can make use of the ``Circuit`` class. The ``Circuit`` class acts as a wrapper around uncontracted components. Since these components are uncontracted we can optimize over the path of contraction and the Fock shapes with `.optimize`.

```python
circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Coherent(0, x=1.0).dual])
circ.optimize(n_init=100, with_BF_heuristic=True, verbose=True)
assert circ.path == [(1, 2), (0, 1)]
```

## 3. CircuitComponents, Ansatz and Wires

Under the hood, all built-in components inherit from the ``CircuitComponent`` class. A ``CircuitComponent`` ...



# The physics module

The physics module contains all the functionality related to the quantum optics of Mr Mustard. This includes the ``Anstaz`` class which is responsible for handling the numerics of a ``CircuitComponent``.


# The math module

The math module is the backbone of Mr Mustard. Mr Mustard comes with a plug-and-play backends through a math interface. You can use it as a drop-in replacement for tensorflow, numpy or jax and your code will be plug-and-play too!

Here's an example where the ``numpy`` backend is used.

```python
import mrmustard.math as math

math.cos(0.1)  # numpy
```


In a different session, we can change the backend to ``tensorflow``.

```python
import mrmustard.math as math
math.change_backend("tensorflow")

math.cos(0.1)  # tensorflow
```


And to ``jax`` as well.

```python
import mrmustard.math as math
math.change_backend("jax")

math.cos(0.1)  # jax
```


### Optimization

The `mrmustard.training.Optimizer` class (supported by the `tensorflow` backend) uses Adam underneath the hood for the optimization of Euclidean parameters, a custom symplectic optimizer for Gaussian gates and states and a unitary/orthogonal optimizer for interferometers.

The `mrmustard.training.OptimizerJax` class (supported by the `jax` backend) operates similarly but only supports Euclidean parameters. The advantage `mrmustard.training.OptimizerJax` has is making use of JIT compilation to speed up optimizations tenfold.

We can turn any simulation in Mr Mustard into an optimization by marking which parameters we wish to be trainable. Let's take a simple example: synthesizing a displaced squeezed state.

```python
from mrmustard import math
from mrmustard.lab import Dgate, Ggate, Attenuator, Vacuum, Coherent, DisplacedSqueezed
from mrmustard.physics import fidelity
from mrmustard.training import Optimizer

math.change_backend("tensorflow")

D = Dgate(mode=0, x=0.1, y=-0.5, x_trainable=True, y_trainable=True)
L = Attenuator(mode=0, transmissivity=0.5)

# we write a function that takes no arguments and returns the cost
def cost_fn_eucl():
    state_out = Vacuum(modes=0) >> D >> L
    return 1 - state_out.fidelity(Coherent(mode=0, x=0.1, y=0.5))

G = Ggate(modes=0, symplectic_trainable=True)
def cost_fn_sympl():
    state_out = Vacuum(modes=0) >> G >> D >> L
    return 1 - state_out.fidelity(DisplacedSqueezed(mode=0, r=0.3, phi=1.1, x=0.4, y=-0.2))

# For illustration, here the Euclidean optimization doesn't include squeezing
opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.01)
opt.minimize(cost_fn_eucl, by_optimizing=[D])  # using Adam for D

# But the symplectic optimization always does
opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.01)
opt.minimize(cost_fn_sympl, by_optimizing=[G,D])  # uses Adam for D and the symplectic opt for G
```
