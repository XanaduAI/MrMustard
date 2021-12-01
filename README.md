[!img](https://github.com/XanaduAI/MrMustard/blob/mm-manteinance/mmlogo.png)

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Actions Status](https://github.com/XanaduAI/MrMustard/workflows/Tests/badge.svg)](https://github.com/XanaduAI/MrMustard/actions)
![Python version](https://img.shields.io/badge/python-3.8-blue)

MrMustard is a differentiable bridge between phase space and Fock space with rich functionality in both representations.

MrMustard supports in fully differentiable way:
- Phase space representation of Gaussian states and Gaussian channels on an arbitrary number of modes
- Exact Fock representation of any Gaussian circuit and any Gaussian state up to an arbitrary cutoff
- Beam splitter, MZ interferometer, squeezer, displacement, phase rotation, bosonic lossy channel, thermal channel, [more to come...]
- General Gaussian N-mode gate and general N-mode Interferometer with dedicated symplectic and orthogonal optimization routines
- Photon number moments
- PNR detectors, Threshold detectors with trainable quantum efficiency and dark counts
- Homodyne, Heterodyne and Generaldyne Gaussian measurements
- An optimizer with a spiffy progress bar
- A composable Circuit object
- Plug-and-play backends (TF and Pytorch [Upcoming])
- An abstraction layer `XPTensor` for seamless symplectic algebra


## Basic API Reference

### 1: States
States in MrMustard are very powerful objects. States, like gates, are trainable.
They have differentiable methods to return a ket or density matrix in Fock space, covariance matrix and means vector in phase space, as well as photon number moments:

```python
from mrmustard.lab import Vacuum, Coherent, SqueezedVacuum, DisplacedSqueezed, TMSV, Thermal, Gaussian

vac  = Vacuum(num_modes=2)
coh  = Coherent(x=0.1, y=-0.4)  # e.g. 2-mode coherent state
sq   = SqueezedVacuum(r = 0.5, phi = 0.3)
dsq  = DisplacedSqueezed(r = 0.5, phi = 0.3, x = 0.3, y = 0.9)
tmsv = TMSV(r = 0.5, phi = 0.3, x = 0.3, y = 0.9)
th   = Thermal(nbar = 2.0)
g    = Gaussian(num_modes = 2)

# Fock representation of a coherent state
coh.ket(cutoffs=[5])   # ket
coh.dm(cutoffs=[5])    # density matrix

coh.cov   # phase space covariance matrix
coh.means # phase space means

coh.number_cov    # photon number covariance matrix
coh.number_means  # photon number means
```
The `repr` of single-mode states shows the Wigner function as well:
[!img](https://github.com/XanaduAI/MrMustard/blob/main/images/repr.png)


States can be joined using the `&` (and) operator:
```python
Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0). # A separable two-mode state

s = SqueezedVacuum(r=1.0)
s4 = s & s & s & s   # four squeezed states
```

Subsystems can be accessed via indices:
```python
joint = Coherent(x=1.0, y=1.0) & Coherent(x=2.0, y=2.0)
joint[0]  # first mode
joint[1]  # second mode

rearranged = joint[1,0]
```

### 2. Gates
We have a variety of unitary Gaussian gates and non-unitary Gaussian channels.
Note that if a parameter of a single-mode gate is a float or a list of length 1, its value is shared across all the modes the gate is applied to.

```python
from mrmustard.lab import Vacuum
from mrmustard.lab import Dgate, Sgate, Rgate, LossChannel  # 1-mode gates ; parallelizable
from mrmustard.lab import BSgate, MZgate, S2gate  # 2-mode gates
from mrmustard.lab import Ggate, Interferometer  # N-mode gates

# a single-mode squeezer with bounded squeezing magnitude
S = Sgate(r = 0.1, phi = 0.9, r_bounds=(0.0, 1.0))

# two single-mode displacements in parallel, with independent parameters:
D = Dgate(x = [0.1, 0.2], y = [-0.5, 0.4])

# two single-mode displacements in parallel, with shared parameters:
D = Dgate(modes = [0,1], x = 0.1, y = -0.5)

# a mix of shared and independent parameters is also allowed
D = Dgate(x=0.2, y=[0.9, 1.9]))

# a lossy bosonic channel with fixed transmissivity
L = LossChannel(transmissivity = 0.9, transmissivity_trainable = False)

# a generic gaussian transformation on 4 modes
G = Ggate(num_modes=4)

state = G(Vacuum(4))  # output of Gaussian transformation

S = Sgate(r = 0.1, phi = 0.9)
D = Dgate(x = 0.3, y = 1.9)
disp_sq = D(S(Vacuum(1)))  # displaced squeezed vacuum
```

Single-mode gates created with a single parameter parallelize automatically over all modes:
```python
D = Dgate(x=0.5)
D(Vacuum(5)) # D is applied to all modes
```

If you want to apply a gate to specific modes, use the `getitem` format. Here are a few examples:
```python
state = D[1](S[0](Vacuum(2))) # squeezing on mode 0 and displacement on mode 1

BS = BSgate(theta=0.1)
state = BS[0,2](Vacuum(3)) # applying a beamsplitter to modes 0 and 2

state = S[0,1,2](Vacuum(4)) # applying the squeezing gate in parallel to modes 0, 1 and 2 but not to mode 3
```

### 3: Circuits

Circuits are a way of getting more functionality out of a collection of gates.
In order to build a circuit we create an empty circuit object `circ = Circuit()` and append gates to it, or we pass a list of gates.
Circuits are callable and trainable.

```python
from mrmustard.lab import Circuit, Vacuum, Sgate, Interferometer, LossChannel

modes = [0,1,2,3]

X4 = Circuit()
X4.append(Sgate(r = 1.0, phi = np.random.uniform(0.0, 2*np.pi, size=4)), r_bounds=(0.0, 1.0))
X4.append(LossChannel(transmissivity=0.8))  # automatically parallelized over all modes
X4.append(Interferometer(len(modes)))
L = LossChannel(transmissivity=0.9, transmissivity_trainable=False)
X4.append(L[modes])    # shared over all modes

output = X4(Vacuum(4))  # differentiable output state
```

Circuits are great for modelling realistic components:

``` python
from mrmustard.lab import Circuit, MZgate, LossChannel

lossy_MZ = circuit().append([
    LossChannel([0,1], transmissivity=[0.4, 0.45], transmissivity_trainable=False)  # in-couplings
    MZgate([0,1], phi_a = 0.1, phi_b = 0.4, external=False),  # both phases in the MZ
    LossChannel([0,1], transmissivity=[0.44, 0.52], transmissivity_trainable=False)  # out-couplings
    ])

state = lossy_MZ(state)  # differentiable
```

### 4. Detectors
MrMustard supports a variety of differentiable detectors, which fall into two categories: Gaussian and Fock detectors.

Gaussian detectors are Homodyne, Heterodyne and Generaldyne detectors. They measure a Gaussian state and return a post-measurement state (in the unmeasured modes) that is Gaussian. Fock detectors are the PNRDetector and the ThresholdDetector. They measure the Fock representation of a state and return a post-measurement state (in the unmeasured modes) in the Fock representation, as well as the probability of the outcome.
Note that measurements require the outcome to be specified.


```python
from mrmustard.lab import Circuit, Vacuum, Sgate, BSgate, LossChannel, PNRDetector, ThresholdDetector

circ = Circuit()
circ.append(Sgate(r=0.2, phi=[0.9,1.9]))  # a mix of shared and independent parameters is allowed
circ.append(BSgate(theta=1.4, phi=-0.1))

detector = PNRDetector(modes = [0,1], efficiency=0.9, dark_counts=0.01)

state_out = circ(Vacuum(num_modes=2))
detection_probs = detector(state_out, cutoffs=[2,3])

```

### 5. Optimization
MrMustard implements a dedicated optimizer that can perform symplectic and orthogonal optimization, on top of the usual Euclidean optimization.

Here we could use a default TensorFlow optimizer (no `Ggate`s or `Interferometer`s)):
```python
import tensorflow as tf
from mrmustard.lab import Dgate, LossChannel, Vacuum
from mrmustard.physics.gaussian import fidelity

D = Dgate(x = 0.1, y = -0.5, x_bounds=(0.0, 1.0), x_trainable=True)
L = LossChannel(transmissivity=0.5)

# we write a function that takes no arguments and returns the cost
def cost_fn():
    state_out = L(D(Vacuum(1)))
    return 1 - fidelity(state_out, Coherent(0.1, 0.5))

adam = tf.optimizers.Adam(learning_rate=0.001)

from tqdm import trange
for i in trange(100):
    adam.minimize(cost_fn, displacement.trainable_parameters['euclidean'])
```

But we can also always use MrMustard's optimizer (which calls Adam if needed):
```python
import tensorflow as tf
from mrmustard.lab import Circuit, Ggate, LossChannel, Vacuum, DisplacedSqueezed
from mrmustard.utils import Optimizer
from mrmustard.physics.gaussian import fidelity

circ = Circuit()

G = Ggate(modes = [0], displacement_trainable=False)
L = LossChannel(modes=[0], transmissivity=0.5, transmissivity_trainable=False)
circ.extend([G,L])

def cost_fn():
    state_out = circ(Vacuum(num_modes=1))
    return 1 - fidelity(state_out, DisplacedSqueezed(r=0.3, phi=1.1, x=-0.1, y=0.1))

opt = Optimizer()
opt.minimize(cost_fn, by_optimizing=circ, max_steps=500) # the optimizer stops earlier if the loss is stable
```
