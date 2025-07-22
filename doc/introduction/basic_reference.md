## Basic API Reference

### 1. States and Gates

Here are a few examples of states:

```python
from mrmustard.lab.states import Coherent, GKet, Number, SqueezedVacuum, Vacuum

vac = Vacuum(modes=(0, 1))             # 2-mode vacuum state
coh = Coherent(mode=0, x=0.1, y=-0.4)  # coh state |alpha> with alpha = 0.1 - 0.4j
sq = SqueezedVacuum(mode=0, r=0.5)     # squeezed vacuum state
g = GKet(modes=(0, 1))                 # 2-mode Gaussian state with zero means
fock4 = Number(mode=0, n=4)            # fock state |4>
```

and gates:

```python
import numpy as np
from mrmustard.lab.transformations import (
    Amplifier,
    Attenuator,
    BSgate,
    Dgate,
    Interferometer,
    MZgate,
    Sgate,
    S2gate,
)

D = Dgate(mode=0, x=1.0, y=-0.4)                   # Displacement by 1.0 along x and -0.4 along y
S = Sgate(mode=0, r=0.5)                           # Squeezer with r=0.5
BS = BSgate(modes=(0, 1), theta=np.pi/4)           # 50/50 beam splitter
S2 = S2gate(modes=(0, 1), r=0.5)                   # two-mode squeezer
MZ = MZgate(modes=(0, 1), phi_a=0.3, phi_b=0.1)    # Mach-Zehnder interferometer
I = Interferometer(modes=(0, 1, 2, 3, 4, 5, 6, 7)) # 8-mode interferometer
L = Attenuator(mode=0, transmissivity=0.5)         # pure lossy channel with 50% transmissivity
A = Amplifier(mode=0, gain=2.0)                    # amplifier with 200% gain
```

The `repr` of single-mode states shows the Wigner function:

```python
import numpy as np
from mrmustard.lab.states import Coherent, Number
from mrmustard.lab.transformations import BSgate

# Create cat states
cat_horizontal = (Coherent(mode=0, x=2) + Coherent(mode=0, x=-2)).normalize()
cat_vertical = (Coherent(mode=1, y=2) + Coherent(mode=1, y=-2)).normalize()

# merge with beamsplitter
both_modes = cat_vertical >> cat_horizontal >> BSgate(modes=(0, 1), theta=np.pi/4)

# Wigner function of the marginal
both_modes[0]
```

<img width="571" alt="Wigner function of the marginal" src="https://github.com/user-attachments/assets/85477eef-abd3-4fe3-a00f-c0a6b1dc0260" />

```python
# Wigner function of the projected state
both_modes >> Number(mode=0, n=3).dual
```

<img width="571" alt="Wigner function of the projected state" src="https://github.com/user-attachments/assets/1f4367d2-1dbd-4088-baed-d66e294b554b" />

```python
# Fock amplitudes of the projected state (exact down to machine precision)
both_modes.fock_array(shape=(100, 4))[:,3]
```

### 2. Gates and the right shift operator `>>`

Applying gates to states looks natural, thanks to python's right-shift operator `>>`:

```python
import numpy as np
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import Dgate, Sgate

displaced_squeezed = Vacuum(modes=0) >> Sgate(mode=0, r=0.5) >> Dgate(mode=0, x=1.0)
```

They can be applied on specific modes with the `mode` or `modes` argument. Here are a few examples:

```python
import numpy as np
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import BSgate, Dgate, Sgate

D = Dgate(mode=1, y=-0.4)
S = Sgate(mode=0, r=0.1, phi=0.5)
state = Vacuum(modes=(0, 1)) >> D >> S   # displacement on mode 1 and squeezing on mode 0

BS = BSgate(modes=(0, 2), theta=1.1)
state = Vacuum(modes=(0, 1, 2)) >> BS    # applying a beamsplitter to modes 0 and 2

S0 = Sgate(mode=0, r=0.1, phi=0.5)
S1 = Sgate(mode=1, r=0.1, phi=0.5)
S3 = Sgate(mode=2, r=0.1, phi=0.5)
V = Vacuum(modes=(0, 1, 2, 3))
state = V >> S0 >> S1 >> S3              # applying an Sgate to modes 0, 1 and 3 but not to mode 2
```

### 3. Circuit

When chaining just gates with the right-shift `>>` operator, we create a circuit:

```python
import numpy as np
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import Attenuator, Interferometer, Sgate

X8 = (
    Sgate(mode=0, r=1.0)
    >> Sgate(mode=1, r=1.0)
    >> Sgate(mode=2, r=1.0)
    >> Sgate(mode=3, r=1.0)
    >> Interferometer(modes=(0, 1, 2, 3))
)
output = Vacuum(modes=(0, 1, 2, 3)) >> X8

# lossy X8
noise = lambda: np.random.uniform()
X8_realistic = (
    Sgate(mode=0, r=0.9 + 0.1*noise(), phi=0.1*noise())
    >> Sgate(mode=1, r=0.9 + 0.1*noise(), phi=0.1*noise())
    >> Sgate(mode=2, r=0.9 + 0.1*noise(), phi=0.1*noise())
    >> Sgate(mode=3, r=0.9 + 0.1*noise(), phi=0.1*noise())
    >> Attenuator(mode=0, transmissivity=0.89 + 0.01*noise())
    >> Attenuator(mode=1, transmissivity=0.89 + 0.01*noise())
    >> Attenuator(mode=2, transmissivity=0.89 + 0.01*noise())
    >> Attenuator(mode=3, transmissivity=0.89 + 0.01*noise())
    >> Interferometer(modes=(0, 1, 2, 3))
    >> Attenuator(mode=0, transmissivity=0.95 + 0.01*noise())
    >> Attenuator(mode=1, transmissivity=0.95 + 0.01*noise())
    >> Attenuator(mode=2, transmissivity=0.95 + 0.01*noise())
    >> Attenuator(mode=3, transmissivity=0.95 + 0.01*noise())
)

# 2-mode Bloch Messiah decomposition
bloch_messiah = Sgate(r=[0.1,0.2]) >> BSgate(-0.1, 2.1) >> Dgate(x=[0.1, -0.4])
my_state = Vacuum(2) >> bloch_messiah
```

### 4. Measurements

In order to perform a measurement, we use the left-shift operator, e.g. `coh << sq` (think of the left-shift on a state as "closing" the circuit).

```python
leftover = Vacuum(4) >> X8 << SqueezedVacuum(r=10.0, phi=np.pi)[2]  # a homodyne measurement of p=0.0 on mode 2
```

Transformations can also be applied in the dual sense by using the left-shift operator `<<`:

```python
Attenuator(0.5) << Coherent(0.1, 0.2) == Coherent(0.1, 0.2) >> Amplifier(2.0)
```

This has the advantage of modelling lossy detectors without applying the loss channel to the state going into the detector, which can be overall faster e.g. if the state is kept pure by doing so.

### 5. Detectors

There are two types of detectors in Mr Mustard. Fock detectors (PNRDetector and ThresholdDetector) and Gaussian detectors (Homodyne, Heterodyne). However, Gaussian detectors are a thin wrapper over just Gaussian states, as Gaussian states can be used as projectors (i.e. `state << DisplacedSqueezed(...)` is how Homodyne performs a measurement).

The PNR and Threshold detectors return an array of unnormalized measurement results, meaning that the elements of the array are the density matrices of the leftover systems, conditioned on the outcomes:

```python
results = Gaussian(2) << PNRDetector(efficiency = 0.9, modes = [0])
results[0]  # unnormalized dm of mode 1 conditioned on measuring 0 in mode 0
results[1]  # unnormalized dm of mode 1 conditioned on measuring 1 in mode 0
results[2]  # unnormalized dm of mode 1 conditioned on measuring 2 in mode 0
# etc...
```

The trace of the leftover density matrices will yield the success probability. If multiple modes are measured then there is a corresponding number of indices:

```python
results = Gaussian(3) << PNRDetector(efficiency = [0.9, 0.8], modes = [0,1])
results[2,3]  # unnormalized dm of mode 2 conditioned on measuring 2 in mode 0 and 3 in mode 1
# etc...
```

Set a lower `settings.PNR_INTERNAL_CUTOFF` (default 50) to speed-up computations of the PNR output.

### 6. Comparison operator `==`

States support the comparison operator:

```python
>>> bunched = (Coherent(1.0) & Coherent(1.0)) >> BSgate(np.pi/4)
>>> bunched.get_modes(1) == Coherent(np.sqrt(2.0))
True
```

As well as transformations (gates and circuits):

```python
>>> Dgate(np.sqrt(2)) >> Attenuator(0.5) == Attenuator(0.5) >> Dgate(1.0)
True
```

### 7. State operations and properties

States can be joined using the `&` (and) operator:

```python
Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0)  # A separable two-mode coherent state

s = SqueezedVacuum(r=1.0)
s4 = s & s & s & s   # four squeezed states
```

Subsystems can be accessed via `get_modes`:

```python
joint = Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0)
joint.get_modes(0)  # first mode
joint.get_modes(1)  # second mode

swapped = joint.get_modes([1,0])
```

### 8. Fock representation

The Fock representation of a State is obtained via `.ket(cutoffs)` or `.dm(cutoffs)`. For circuits and gates it's `.U(cutoffs)` or `.choi(cutoffs)`. The Fock representation is exact (with minor caveats) and it doesn't break differentiability. This means that one can define cost functions on the Fock representation and backpropagate back to the phase space representation.

```python
# Fock representation of a coherent state
Coherent(0.5).ket(cutoffs=[5])   # ket
Coherent(0.5).dm(cutoffs=[5])    # density matrix

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

Alternatively,

```python
my_amplitudes = np.array([0.5, 0.25, -0.5, 0.25, 0.25, 0.5, -0.25])  # no buffer
my_state = State(ket=my_amplitudes)
my_state._cutoffs = [42]  # force the cutoff
my_state >> Sgate(r=0.5)  # works too
```

### The physics module

The physics module contains a growing number of functions that we can apply to states directly. These are made out of the functions that operate on the _representation_ of the state:

- If the state is in Gaussian representation, then internally the physics functions utilize the [physics.gaussian](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/physics/gaussian.py) module.
- If the state is in Fock representation, then internally the physics functions utilize the [physics.fock](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/physics/fock.py) module.

### The math module

The math module is the backbone of Mr Mustard, which consists in the [Math](https://github.com/XanaduAI/MrMustard/blob/main/mrmustard/math/math_interface.py) interface
Mr Mustard comes with a plug-and-play backends through a math interface. You can use it as a drop-in replacement for tensorflow or numpy and your code will be plug-and-play too!
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

### Optimization

The `mrmustard.training.Optimizer` uses Adam underneath the hood for the optimization of Euclidean parameters, a custom symplectic optimizer for Gaussian gates and states and a unitary/orthogonal optimizer for interferometers.

We can turn any simulation in Mr Mustard into an optimization by marking which parameters we wish to be trainable. Let's take a simple example: synthesizing a
displaced squeezed state.

```python
from mrmustard import math
from mrmustard.lab import Dgate, Ggate, Attenuator, Vacuum, Coherent, DisplacedSqueezed
from mrmustard.physics import fidelity
from mrmustard.training import Optimizer

math.change_backend("tensorflow")

D = Dgate(x = 0.1, y = -0.5, x_trainable=True, y_trainable=True)
L = Attenuator(transmissivity=0.5)

# we write a function that takes no arguments and returns the cost
def cost_fn_eucl():
    state_out = Vacuum(1) >> D >> L
    return 1 - fidelity(state_out, Coherent(0.1, 0.5))

G = Ggate(num_modes=1, symplectic_trainable=True)
def cost_fn_sympl():
    state_out = Vacuum(1) >> G >> D >> L
    return 1 - fidelity(state_out, DisplacedSqueezed(r=0.3, phi=1.1, x=0.4, y=-0.2))

opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.01)
opt.minimize(cost_fn_eucl, by_optimizing=[D])  # using Adam for D

opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.01)
opt.minimize(cost_fn_sympl, by_optimizing=[G,D])  # using Adam for D and the symplectic opt for G
```
