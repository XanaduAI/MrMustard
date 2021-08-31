# MrMustard

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)

MrMustard is a differentiable bridge between phase space and Fock space with rich functionality in both representations.

MrMustard supports (fully differentiable):
- Phase space representation on an arbitrary number of modes and Fock representation with mode-wise dimensionality cutoffs.
- Beam splitter, MZ interferometer, squeezer, displacement, phase rotation, bosonic lossy channel, thermal channel, [more to come...]
- General Gaussian N-mode gate and general Interferometer with dedicated symplectic and orthogonal optimization routines
- Photon number moments
- PNR detectors, threshold detectors with trainable quantum efficiency and dark counts
- Homodyne, Heterodyne and Generaldyne Gaussian measurements [New!]
- An optimizer with a spiffy progress bar
- A circuit compiler [New!]
- Plug-and-play backends (TF and Pytorch [Upcoming], even more to come...)
- The abstraction layer `XPTensor` for seamless symplectic algebra


## Basic API Reference

### 1: States
States in MrMustard are very powerful objects. They have differentiable methods to return a ket or density matrix in Fock space, covariance matrix and means vector in phase space, as well as photon number moments:
```python
from mrmustard import Vacuum, Coherent, SqueezedVacuum, DisplacedSqueezed, Thermal

vac = Vacuum(num_modes=2, hbar = 2.0)
coh = Coherent(x=[0.1, 0.2], y=[-0.4, 0.9], hbar=2.0)  # 2-mode coherent state
sq  = SqueezedVacuum(r = 0.5, phi = 0.3)
dsq = DisplacedSqueezed(r = 0.5, phi = 0.3, x = 0.3, y = 0.9, hbar=2.0)
tmsv = TwoModeSqueezedVacuum(r = 0.5, phi = 0.3, x = 0.3, y = 0.9, hbar=2.0)
thr = Thermal(num_modes=2, hbar=2.0)

# e.g. fock representation of coherent state
coh.ket(cutoffs=[4,5])
coh.dm(cutoffs=[4,5])

coh.cov   # phase space covariance matrix
coh.means # phase space means

coh.number_cov    # photon number covariance matrix
coh.number_means  # photon number means
```


### 2. Gates
Gates are callable objects. We have a variety of unitary Gaussian gates and non-unitary Gaussian channels.
Note that if a parameter of a single-mode gate is a float or a list of length 1 its value is shared across all the modes the gate is operating on.

```python
from mrmustard import Vacuum
from mrmustard import Dgate, Sgate, Rgate, LossChannel  # 1-mode gates ; parallelizable
from mrmustard import BSgate, MZgate, S2gate  # 2-mode gates
from mrmustard import Ggate, Interferometer  # N-mode gates

# a single-mode squeezer with bounded squeezing magnitude
S = Sgate(modes = [0], r = 0.1, phi = 0.9, r_bounds=(0.0, 1.0))

# two single-mode displacements in parallel, with independent parameters:
D = Dgate(modes = [0, 1], x = [0.1, 0.2], y = [-0.5, 0.4])

# two single-mode displacements in parallel, with shared parameters:
D = Dgate(modes = [0, 1], x = 0.1, y = -0.5)

# a mix of shared and independent parameters is also allowed
D = Dgate(modes = [0,1], x=0.2, y=[0.9, 1.9]))

# a lossy bosonic channel with fixed transmissivity
L = LossChannel(modes = [0], transmissivity = 0.9, transmissivity_trainable = False)

# a generic gaussian transformation on 4 modes
G = Ggate(modes = [0,1,2,3])

state = G(Vacuum(4))  # output of Gaussian transformation

S = Sgate(modes = [0], r = 0.1, phi = 0.9)
D = Dgate(modes = [0], x = 0.3, y = 1.9)
disp_sq = D(S(Vacuum(1)))  # displaced squeezed vacuum
```


### 3: Circuits

Circuits are a way of getting more functionality out of a collection of gates.
In order to build a circuit we create an empty circuit object `circ = Circuit()` and append gates to it. 
Circuits are mutable sequences, which means they support all of the `list` methods (e.g. `circ.append(gate)`, `for gate in circ`, `some_gates = circ[1:4]`, `circ[6] = this_gate`, `circ.pop()`, etc...)

The circuit is callable as well: it takes a state object representing the input and it returns the output state:

```python
from mrmustard import Circuit, Vacuum, Sgate, Interferometer, LossChannel

modes = [0,1,2,3,4,5,6,7]

X8 = Circuit()
X8.append(Sgate(modes, r = 0.1, phi = np.random.uniform(0.0, 2*np.pi, size=8)), r_bounds=(0.0, 1.0))
X8.append(LossChannel(modes, transmissivity=0.8, transmissivity_trainable=False))  # shared over all modes
X8.append(Interferometer(modes))
X8.append(LossChannel(modes, transmissivity=0.9, transmissivity_trainable=False))  # shared over all modes
state = X8(Vacuum(8))  # differentiable output state

X8c = X8.compile()  # compile the circuit and reduce it to a single quantum channel (differentiable with caveats)
```

Circuits are great for modelling realistic components:

``` python
from mrmustard import Circuit, MZgate, LossChannel

lossy_MZ = circuit().append([
    LossChannel([0,1], transmissivity=[0.4, 0.45], transmissivity_trainable=False)  # in-couplings
    MZgate([0,1], phi_a = 0.1, phi_b = 0.4, external=False),  # both phases in the MZ
    LossChannel([0,1], transmissivity=[0.44, 0.52], transmissivity_trainable=False)  # out-couplings
    ])

state = lossy_MZ(state)  # differentiable
```

### 4. Detectors
MrMustard supports a variety of differentiable detectors, but they all fall into two categories: Gaussian and Fock detectors.

Gaussian detectors are Homodyne, Heterodyne and Generaldyne detectors. They measure a Gaussian state and return a post-measurement state (in the unmeasured modes) that is Gaussian. Fock detectors are the PNRDetector and the ThresholdDetector. They measure the Fock representation of a state and return a post-measurement state (in the unmeasured modes) in the Fock representation, as well as the probability of the outcome.
Note that measurements require the outcome to be specified.


```python
from mrmustard.tools import Circuit
from mrmustard.states import Vacuum
from mrmustard.gates import Sgate, BSgate, LossChannel
from mrmustard.measurements import PNRDetector, ThresholdDetector, Homodyne


circ = Circuit()
circ.append(Sgate(modes = [0,1], r=0.2, phi=[0.9,1.9]))  # a mix of shared and independent parameters is allowed
circ.append(BSgate(modes = [0,1], theta=1.4, phi=-0.1))
circ.append(LossChannel(modes=[0,1], transmissivity=0.5))

detector = PNRDetector(modes = [0,1], efficiency=0.9, dark_counts=0.01)

state_out = circ(Vacuum(num_modes=2))
detection_probs = detector(state_out, cutoffs=[2,3])

# TODO Teleportation of a single-mode Gaussian state

```

### 5. Optimization
MrMustard implements a dedicated optimizer that can perform symplectic and orthogonal optimization, on top of the usual Euclidean optimization.

Here we use a default TensorFlow optimizer (no `Ggate`s or `Interferometer`s)):
```python
import tensorflow as tf
from mrmustard import Dgate, LossChannel, Vacuum, fidelity

D = Dgate(modes = [0], x = 0.1, y = -0.5, x_bounds=(0.0, 1.0), x_trainable=True, y_trainable=False)
L = LossChannel(modes=[0], transmissivity=0.5, transmissivity_trainable=False)

# we write a function that takes no arguments and returns the cost
def cost_fn():
    state_out = L(D(Vacuum(num_modes=1)))
    return 1 - fidelity(state_out, Coherent(0.1, 0.5))

adam = tf.optimizers.Adam(learning_rate=0.001)

from tqdm import trange
for i in trange(100):
    adam.minimize(cost_fn, displacement.trainable_parameters['euclidean'])
```

Here we use MrMustard's optimizer (which calls Adam when needed):
```python
import tensorflow as tf
from mrmustard import Circuit, Optimizer, Ggate, LossChannel, Vacuum, DisplacedSqueezed

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
