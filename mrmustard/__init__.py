import importlib
# import rich
# rich.pretty.install()

__version__ = "0.1.0"


def set_backend(name: str):
    from mrmustard.core.baseclasses import State, Gate, Detector
    from mrmustard.tools import Optimizer

    mod = importlib.import_module('mrmustard.core.backends.' + name)

    State._math_backend = mod.MathBackend()
    State._state_backend = mod.StateBackend()
    Gate._math_backend = mod.MathBackend()
    Gate._symplectic_backend = mod.SymplecticBackend()
    Detector._math_backend = mod.MathBackend()
    Optimizer._math_backend = mod.MathBackend()
    Optimizer._opt_backend = mod.OptimizerBackend()


set_backend('tf')
