![Logo](https://github.com/XanaduAI/MrMustard/blob/main/mm_white.png#gh-light-mode-only)
![Logo](https://github.com/XanaduAI/MrMustard/blob/main/mm_dark.png#gh-dark-mode-only)

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Actions Status](https://github.com/XanaduAI/MrMustard/workflows/Tests/badge.svg)](https://github.com/XanaduAI/MrMustard/actions)
[![Python version](<https://img.shields.io/badge/python-3.8 | 3.9-blue>)](https://pypi.org/project/MrMustard/)

Alpha release (v0.1.0).

Mr Mustard is a differentiable simulator with a sophisticated built-in optimizer, that operates across phase space and Fock space.
It is built on top of an agnostic autodiff interface, to allow for plug-and-play backends (TensorFlow by default, PyTorch coming soon).

Mr Mustard supports (differentiably):
- Phase space representation of Gaussian states and Gaussian channels on an arbitrary number of modes
- Exact Fock representation of any Gaussian circuit and any Gaussian state up to an arbitrary cutoff
- single-mode gates (parallelizable):
    - squeezing, displacement, phase rotation, attenuator, amplifier, additive noise
- two-mode gates:
    - beam splitter, Mach-Zehnder interferometer, two-mode squeezing
- N-mode gates (with dedicated Rimannian optimization):
    - interferometer (orthogonal), Gaussian transformation (symplectic)
- single-mode states (parallelizable):
    - vacuum, coherent, squeezed, displaced-squeezed, thermal
- two-mode states:
    - two-mode squeezed vacuum
- N-mode states:
    - Gaussian state
- Photon number moments and entropic measures
- PNR detectors and Threshold detectors with trainable quantum efficiency and dark counts
- Homodyne, Heterodyne and Generaldyne measurements
- Composable circuits
- Plug-and-play backends (TensorFlow as default)
- An abstraction layer `XPTensor` for seamless symplectic algebra (experimental)


# Basic API Reference

## 1. The lab
The lab module contains things you'd find in the lab: states, gates, measurements, circuits.

### States and Gates
Here are a few examples of states and gates:
```python
import numpy as np
from mrmustard.lab import *

vac = Vacuum(num_modes=2)        # 2-mode vacuum state
coh = Coherent(x=0.1, y=-0.4)    # coh state |alpha> with alpha = 0.1 - 0.4j
sq  = SqueezedVacuum(r=0.5)      # squeezed vacuum state
g   = Gaussian(num_modes=2)      # 2-mode Gaussian state with zero means
fock4 = Fock(4)                  # fock state |4>

D  = Dgate(x=1.0, y=-0.4)         # Displacement by 1.0 along x and -0.4 along y
S  = Sgate(r=0.5)                 # Squeezer with r=0.5<img width="542" alt="Screen Shot 2021-12-06 at 1 30 47 PM" src="https://user-images.githubusercontent.com/8944955/144901934-cacd0665-3cc0-4975-99f7-ade39931a0c5.png">

BS = BSgate(theta=np.pi/4)        # 50/50 beam splitter
I  = Interferometer(8)            # 8-mode interferomter
L  = Attenuator(0.5)              # pure lossy channel with 50% transmissivity
A  = Amplifier(2.0, nbar=1.0)     # noisy amplifier with 200% amplification
```

The `repr` of single-mode states shows the Wigner function:
<img width="571" alt="Screen Shot 2021-12-06 at 1 31 17 PM" src="https://user-images.githubusercontent.com/8944955/144902008-8d26d59c-8600-4391-9144-ffcc1b2215c2.png">

### Gates

Gates are applied to states using python's right-shift operator, e.g. `coh >> S`.
Describing circuits then becomes quite natural:
```python
displaced_squeezed = Vacuum(1) >> Sgate(r=0.5) >> D(x=1.0)
```

If you want to apply a gate to specific modes, use the `getitem` format. Here are a few examples:
```python
D = Dgate(y=-0.4)
S = Sgate(r=0.1, phi=0.5)
state = Vacuum(2) >> D[1] >> S[0] # displacement on mode 1 and squeezing on mode 0

BS = BSgate(theta=1.1)
state = Vacuum(3) >> BS[0,2] # applying a beamsplitter to modes 0 and 2
state = Vacuum(4) >> S[0,1,2] # applying the same single-mode squeezing gate in parallel to modes 0, 1 and 2 but not to mode 3
```

### Circuit
When combining only gates with the right-shift `>>` operator, we create a circuit:
```python
# Ideal X8
X8 = Sgate(r=0.9, phi=np.random.uniform(0.0, 2*np.pi)) >> Interferometer(num_modes=4)
x8_out = Vacuum(num_modes=4) >> X8

# More realistic X8
noise = lambda: np.random.uniform(size=4)
X8_realistic = (Sgate(r=0.9 + 0.1*noise, phi=0.1*noise)
                >> Attenuator(0.89 + 0.01*noise)
                >> Interferometer(4)
                >> Attenuator(0.95 + 0.01*noise, nbar=0.01)
               )

# Bloch Messiah decomposition
bloch_messiah = BSgate(0.2, 1.9) >> Sgate(r=[0.1,0.2]) >> BSgate(-0.1, 2.1) >> Dgate(0.1 -0.4)
my_state = Vacuum(2) >> bloch_messiah
```

### Measurements
In order to perform a measurement, we use the left-shift operator, e.g. `coh << sq` (think of the left-shift on a state as "closing" the circuit).
```python
leftover = x8_out << SqueezedVacuum(r=10.0, phi=np.pi)[2]  # a homodyne measurement of p=0.0 on mode 2
```

Transformations can also be applied in reverse (i.e. dually) by beginning with a state on the **right** and proceeding toward the left:
```python
lossy_pnr_4 = Attenuator(0.8) << fock4
leftover = x8_out << lossy_pnr_4[0]  # measuring 4 photons in mode 0 with a lossy pnr detector
```
This has the advantage of modelling lossy detectors without applying the loss channel to the state going into the detector, which can be overall faster e.g. if the state is kept pure by doing so.

### Detectors
FILL IN

### Equality check
States, Gates and Circuits support equality checking:
```python
bunched = (Coherent(1.0) & Coherent(1.0)) >> BSgate(np.pi/4)
bunched.get_modes(1) == Coherent(np.sqrt(2.0))
>>> True
```

### State operations and properties
States can be joined using the `&` (and) operator:
```python
Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0). # A separable two-mode coherent state

s = SqueezedVacuum(r=1.0)
s4 = s & s & s & s   # four squeezed states
```

Subsystems can be accessed via `get_modes`:
```python
joint = Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0)
joint.get_mode(0)  # first mode
joint.get_mode(1)  # second mode

swapped = joint.get_modes(1,0)
```

### Fock representation
The Fock representation of a State is obtained via `.ket(cutoffs)` or `.dm(cutoffs)`. For circuits and gates it's `.U(cutoffs)` or `.choi(cutoffs)`. The Fock representation is exact (with miron caveats) and it doesn't break differentiability. This means that one can define cost functions on the Fock representation and backpropagate back to the phase space representation.

```python
# Fock representation of a coherent state
coh.ket(cutoffs=[5])   # ket
coh.dm(cutoffs=[5])    # density matrix

Dgate(x=1.0).U(cutoffs=[15])  # truncated unitary op
Dgate(x=1.0).choi(cutoffs=[15])  # truncated choi op
```

States can be initialized in Fock representation and used as any other state:
```python
my_amplitudes = np.array([0.5, 0.25, -0.5, 0.25, 0.25, 0.5, -0.25] + [0.0]*23)  # notice the buffer
my_state = State(ket=my_amplitudes)
my_state >> Sgate(r=0.5)  # just works
```
<img width="542" alt="Screen Shot 2021-12-06 at 1 44 38 PM" src="https://user-images.githubusercontent.com/8944955/144903799-5b6c1524-4357-4be0-9778-e1f0de6943c1.png">


## 2. Physics
FILL IN


## 3. Math
Mr Mustard comes with a plug-and-play backend. You can use it as a drop-in replacement for tensorflow or pytorch and your code will be plug-and-play too!
```python
from mrmustard import settings
from mrmustard.math import Math
math = Math()

math.cos(0.1)  # tensorflow

settings.BACKEND = 'torch'

math.cos(0.1)  # pytorch
```

### 4. Optimization
The utils module is currently only offering the Optimizer (in the future we will implement a Compiler).
The `Optimizer` uses Adam underneath the hood for Euclidean parameters and a custom symplectic optimizer for Gaussian gates and states and an orthogonal optimizer  for interferometers.

```python
from mrmustard.lab import Dgate, Attenuator, Vacuum
from mrmustard.physics import fidelity

D = Dgate(x = 0.1, y = -0.5, x_trainable=True, y_trainable=True)
L = Attenuator(transmissivity=0.5)

# we write a function that takes no arguments and returns the cost
def cost_fn_eucl():
    state_out = Vacuum(1) >> D >> L
    return 1 - fidelity(state_out, Coherent(0.1, 0.5))

G = Ggate(symplectic_trainable=True)
def cost_fn_sympl():
    state_out = Vacuum(1) >> G >> D >> L
    return 1 - fidelity(state_out, DisplacedSqueezed(r=0.3, phi=1.1, x=-0.1, y=0.1))

opt = Optimizer()
opt.minimize(cost_fn_eucl, by_optimizing=[D])  # using Adam for D and the symplectic opt for G

opt = Optimizer()
opt.minimize(cost_fn_sympl, by_optimizing=[G,D])  # using Adam for D and the symplectic opt for G
```
