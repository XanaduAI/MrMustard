import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

def get_backend(env_name: str):
    return importlib.import_module("mrmustard.backends." + env_name).Backend

# get backend first
Backend = get_backend("tensorflow")

from mrmustard.concrete import *
from mrmustard.abstract import State

hbar = 2.0  # default