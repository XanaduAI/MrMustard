# MrMustard

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)

MrMustard is a differentiable bridge between phase space and Fock space. In MrMustard _every operation is differentiable_.

MrMustard supports:
- Gaussian channels and transformations (beam splitter, squeezer, displacement, phase rotation, bosonic lossy channel, thermal channel, general Gaussian gates)
- Phase space representation, Fock representation, photon number moments
- Custom fock-cutoff per mode
- PNR detectors and threshold detectors with quantum efficiency and dark counts
- symplectic optimization (with a spiffy progress bar)
- plugin-based backends for easy customization and contributions


## API Reference

### 1: Circuits

In order to build a circuit we create an empty circuit object `circ = Circuit()` and append gates to it. 
Circuits are mutable sequences, which means they support all of the `list` methods (e.g. `circ.append(gate)`, `for gate in circ`, `some_gates = circ[1:4]`, `circ[6] = this_gate`, etc...)

The circuit is also callable: it takes a state object representing the input and it returns the output state:

```python
from mrmustard.tf import Circuit, Ggate, Vacuum

circ = Circuit()

circ.append(Ggate(modes=[0,1], displacement_trainable=False))

state_in = Vacuum(num_modes=2, hbar = 2.0)
state_out = circ(state_in)
```

### 2. Gates
It's not necessary to set up a whole circuit if you just want to apply a few gates. Just like the circuit, gates are callable too (calling the circuit actually calls all of the gates in sequence):

```python
from mrmustard.tf import Vacuum, Dgate, LossChannel

displacement = Dgate(modes = [0], x = 0.1, y = -0.5)
loss = LossChannel(modes=[0], transmissivity=0.5)

state_in = Vacuum(num_modes=1, hbar=2.0)
state_out = loss(displacement(state_in))
```

### 3. Detectors
MrMustard supports detectors, and even though the output of a detector is a probability distribution over the outcomes, even this operation is differentiable.

```python
from mrmustard.tf import Circuit, BSgate, LossChannel, Vacuum, PNR

circ = Circuit()
circ.append(Sgate(modes = [0,1], r=0.2, phi=[0.9,1.9])) # if a parameter is not a list, its value is the same on all modes
circ.append(BSgate(modes = [0,1], theta=1.4, phi=-0.1))
circ.append(LossChannel(modes=[0,1], transmissivity=0.5)) # same here

detector = PNR(modes = [0,1], quantum_efficiency=0.9, dark_count_prob=0.01)

state_out = circ(Vacuum(num_modes=2))
detection_probs = detector(state_out, cutoffs=[2,3])
```

### 4: States
States in MrMustard are very powerful objects. They have differentiable methods to return a ket or density matrix in Fock space, covariance matrix and means vector in phase space, as well as photon number covariance and photon number means vector:
```python
from mrmustard.tf import Circuit, Ggate, LossChannel, Vacuum

circ = Circuit()
circ.append(Ggate(modes=[0,1], displacement_trainable=False))
circ.append(LossChannel(modes=[0,1], transmissivity=0.5))
state_in = Vacuum(num_modes=2, hbar = 2.0)
state_out = circ(state_in)

state_out.ket(cutoffs=[4,5])  # this will be None as the state is mixed
state_out.dm(cutoffs=[4,5])   # each mode can have a custom cutoff

state_out.cov   # phase space covariance
state_out.means # phase space means

state_out.number_cov    # photon number covariance
state_out.number_means  # photon number covariance
```

### 5. Optimization
The optimizer in MrMustard is a convenience class, which means that other optimizers can be used, as all the transformations are differentiable with respect to the parameters of the gates. The only reason where you may want to use the optimizer is becasue it supports symplectic optimization for generic Gaussian transformations (`Ggate`), and it applies it automatically if there are `Ggate`s in the circuit.

Here we use a default TensorFlow optimizer (no `Ggate`s):
```python
from mrmustard.tf import Dgate, LossChannel, Vacuum

displacement = Dgate(modes = [0], x = 0.1, y = -0.5, x_bounds=(0.0, 1.0), x_trainable=True, y_trainable=False)
loss = LossChannel(modes=[0], transmissivity=0.5, transmissivity_trainable=False)

def cost_fn():
    state_out = loss(displacement(Vacuum(num_modes=1)))
    return tf.abs(state_out.means[0] - 0.2)**2

adam = tf.optimizers.Adam(learning_rate=0.001)

from tqdm import trange
for i in trange(100):
    adam.minimize(cost_fn, displacement.euclidean_parameters)
```

Here we use MrMustard's optimizer:
```python
from mrmustard.tf import Ggate, LossChannel, Vacuum

circ = Circuit()

displacement = Ggate(modes = [0], displacement_trainable=False)
loss = LossChannel(modes=[0], transmissivity=0.5, transmissivity_trainable=False)
circ.append(displacement)
circ.append(loss)

def cost_fn():
    state_out = circ(Vacuum(num_modes=1))
    return tf.abs(state_out.means[0] - 0.2)**2

opt = Optimizer()
opt.minimize(cost_fn, by_optimizing=[circ], max_steps=100) # the optimizer stops earlier if the loss is stable
```
