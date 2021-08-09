import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

from mrmustard.plugins import FockPlugin, GaussianPlugin, TrainPlugin, GraphicsPlugin
from mrmustard.abstract import GaussianMeasurement, FockMeasurement, State, Transformation
from mrmustard.concrete import *

def set_env(env_name: str):
    backend = importlib.import_module("mrmustard.backends." + env_name).Backend()

    FockPlugin.backend = backend
    GaussianPlugin.backend = backend
    TrainPlugin.backend = backend
    GraphicsPlugin.backend = backend

    State._fock = FockPlugin()
    State._gaussian = GaussianPlugin()
    Transformation._gaussian = GaussianPlugin()
    GaussianMeasurement._gaussian = GaussianPlugin()
    FockMeasurement._fock = FockPlugin()

def using_tensorflow():
    return set_env("tensorflow")

def using_pytorch():
    return set_env("torch")

def using_jax():
    return set_env("jax")

def using_numpy():
    return set_env("numpy")

def using_tinygrad():
    return set_env("tinygrad")


using_tensorflow()  # default

