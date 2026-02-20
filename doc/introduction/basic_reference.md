# Basic Reference

## Examples

### State Visualization

```python
import numpy as np
from mrmustard.lab.states import Coherent, Number
from mrmustard.lab.transformations import BSgate

# Create cat states
cat_horizontal = (Coherent(mode=0, alpha=2.0) + Coherent(mode=0, alpha=-2.0)).normalize()
cat_vertical = (Coherent(mode=1, alpha=2.0j) + Coherent(mode=1, alpha=-2.0j)).normalize()

# merge with beamsplitter
both_modes = cat_vertical >> cat_horizontal >> BSgate(modes=(0, 1), theta=np.pi/4)

# Wigner function of the marginal
both_modes.get_modes(0)
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

### Circuit Simulation

```python
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import BSgate, Dgate, Sgate
from mrmustard.lab.samplers import HomodyneSampler

# Create and apply a circuit
input_state = Vacuum(modes=(0, 1))
output_state = input_state >> BSgate(modes=(0, 1)) >> Sgate(mode=0, r=0.5) >> Dgate(mode=1, alpha=0.5)

# Measure the result
homodyne = HomodyneSampler()
samples = homodyne.sample(state=output_state, n_samples=100)
```

### Optimization

Transform any simulation into an optimization by marking parameters as trainable:

```python
from mrmustard import math
from mrmustard.lab.states import DisplacedSqueezed
from mrmustard.lab.transformations import Dgate, Ggate
from mrmustard.parameters import Variable
from mrmustard.training import Optimizer

math.change_backend("jax")

# Create trainable parameters
alpha = Variable(0.1 - 0.5j, name="alpha")
symplectic = Variable(math.random_symplectic(1), name="symplectic")

# Define cost function which accepts trainable parameters
def cost_fn(alpha, symplectic):
    D = Dgate(mode=0, alpha=alpha)
    G = Ggate(modes=0, symplectic=symplectic)
    state_out = Vacuum(modes=0) >> G >> D
    target = DisplacedSqueezed(mode=0, alpha=0.4 - 0.2j, r=0.3, phi=1.1)
    return 1 - state_out.fidelity(target)

# Optimize
opt = Optimizer(symplectic_lr=0.1, euclidean_lr=0.01)
(alpha_optimized, symplectic_optimized) = opt.minimize(cost_fn, by_optimizing=[alpha, symplectic])
```

## Backend Flexibility

Switch between numerical backends seamlessly:

```python
import mrmustard.math as math

# Default numpy backend
math.cos(0.1)  # numpy

# Switch to jax
math.change_backend("jax")
math.cos(0.1)  # jax
```

## Architecture

### The lab Module

Contains components you'd find in a quantum optics lab: states, transformations, measurements, and circuits. States can be used as initial conditions or as measurements (projections).

### The physics Module

Contains the core quantum optics functionality, including the `Ansatz` class responsible for handling the numerics of circuit components.

### The math Module

The backbone providing plug-and-play backend support. Acts as a drop-in replacement for `numpy` or `jax`.
