import importlib
from rich.pretty import install

install()

__version__ = "0.1.0"


def set_env(env_name: str):
    from mrmustard.core.baseclasses import State, Op, Detector
    from mrmustard.tools import Optimizer

    mod = importlib.import_module("mrmustard.core.plugins." + env_name)

    State._math = mod.MathBackend()
    State._state_plugin = mod.StatePlugin()
    Op._math = mod.MathBackend()
    Op._symplectic_plugin = mod.SymplecticPlugin()
    Detector._math = mod.MathBackend()
    Optimizer._math = mod.MathBackend()
    Optimizer._opt_plugin = mod.OptimizerPlugin()



def set_tensorflow():
    set_env("tensorflow")

def set_pytorch():
    set_env("torch")

def set_jax():
    set_env("jax")

def set_numpy():
    set_env("numpy")


set_tensorflow()  # default