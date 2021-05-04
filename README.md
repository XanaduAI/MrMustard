# MrMustard

[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)

MrMustard is an easily-extendible simulation and optimization library for Gaussian circuits.
Create circuits, add gates, compute the output. It's all differentiable. 

MrMustard features:
- Beam splitter, squeezer, displacement, phase rotation, bosonic lossy channel, general Gaussian gates
- Pure and mixed states
- Gaussian representation (covariance matrix and means vector)
- Fock space representation (with custom per-mode cutoffs)
- Optimization on the symplectic group
- Plugin architecture for adding new backends (currently running on TensorFlow)
- [coming soon] Complex Gaussian mixture representation 


## API Reference

#### Create a circuit
```python
from mrmustard.tf import Circuit

circ = Circuit(num_modes = 2)
```

#### Add gates
```python
from mrmustard.tf import BSgate, Sgate

circ.add_gate(Sgate(modes = [0]))
circ.add_gate(Sgate(modes = [1]))
circ.add_gate(BSgate(modes = [0,1]))
```

#### Fock output state
```python
fock = circ.fock_output(cutoffs = [8,6,20]) # custom per-mode cutoffs
```

#### Symplectic output state
```python
gaussian = circ.gaussian_output()
symplectic.cov # covariance matrix
symplectic.means # covariance matrix
```

#### Fock optimization
```python
def loss_fn():
    probs = circ.fock_probabilities(cutoffs=[5,2,2])
    return probs[0,0,0] + probs[1,1,1] # I made this up

opt = Optimizer()
opt.minimize(circ, loss_fn)
```

#### Gaussian optimization
```python
import tensorflow as tf
def loss_fn():
    cov = circ.gaussian_output().cov
    return tf.abs(cov[0,0] + cov[1,1] - 0.5)**2 # I made this up

opt = Optimizer()
opt.minimize(circ, loss_fn)
```