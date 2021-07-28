import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

def set_env(env_name: str):
    from mrmustard.plugins import FockPlugin, SymplecticPlugin, TrainPlugin
    from mrmustard.tools import Optimizer

    backend = importlib.import_module("mrmustard.backends." + env_name).Backend()

    FockPlugin._backend = backend
    SymplecticPlugin._backend = backend
    TrainPlugin._backend = backend


def using_tensorflow():
    set_env("tensorflow")

def using_pytorch():
    set_env("torch")

def using_jax():
    set_env("jax")

def using_numpy():
    set_env("numpy")


using_tensorflow()  # default