import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

from mrmustard.plugins import FockPlugin, GaussianPlugin, TrainPlugin, GraphicsPlugin

def set_env(env_name: str):
    backend = importlib.import_module("mrmustard.backends." + env_name).Backend()

    FockPlugin._backend = backend
    GaussianPlugin._backend = backend
    TrainPlugin._backend = backend
    GraphicsPlugin._backend = backend

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

from mrmustard.concrete import *