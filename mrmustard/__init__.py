# this is the topmost __init__.py file of the mrmustard package

import importlib
from rich.pretty import install

install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"
hbar = 2.0
tmsv_r = 3.0


def get_backend(backend_name: str):
    return importlib.import_module("mrmustard.backends." + backend_name).Backend


Backend = get_backend("tensorflow")
from mrmustard.concrete import *
from mrmustard.abstract import State
