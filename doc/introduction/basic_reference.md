## Examples

### State Visualization

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

### Circuit Simulation

```python
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import BSgate, Dgate, Sgate
from mrmustard.lab.samplers import HomodyneSampler

# Create and apply a circuit
input_state = Vacuum(modes=(0, 1))
output_state = input_state >> BSgate(modes=(0, 1)) >> Sgate(mode=0, r=0.5) >> Dgate(mode=1, x=0.5)

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
from mrmustard.training import OptimizerJax

math.change_backend("jax")

# Create trainable gates
D = Dgate(mode=0, x=0.1, y=-0.5, x_trainable=True, y_trainable=True)
G = Ggate(modes=0, symplectic_trainable=True)

# Define cost function
def cost_fn(G, D):
    state_out = Vacuum(modes=0) >> G >> D
    target = DisplacedSqueezed(mode=0, r=0.3, phi=1.1, x=0.4, y=-0.2)
    return 1 - state_out.fidelity(target)

# Optimize
opt = OptimizerJax(symplectic_lr=0.1, euclidean_lr=0.01)
(G,D) = opt.minimize(cost_fn, by_optimizing=[G, D])
```

### Advanced: Circuit Optimization

```python
from mrmustard.lab.circuits import Circuit
from mrmustard.lab.states import Coherent, Number
from mrmustard.lab.transformations import Sgate

# Optimize contraction path and Fock shapes
circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Coherent(0, x=1.0).dual])
circ.optimize(n_init=100, with_BF_heuristic=True, verbose=True)
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

---
