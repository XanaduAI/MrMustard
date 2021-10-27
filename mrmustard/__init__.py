# this is the topmost __init__.py file of the mrmustard package

import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

def activate_backend(backend_name: str):
    "Activates the backend in the core"
    Backend = importlib.import_module("mrmustard.backends." + backend_name).Backend
    from mrmustard.core import fock;
    fock.backend = Backend()
    
    from mrmustard.core import gaussian;
    gaussian.backend = Backend()

    from mrmustard.core import train;
    train.backend = Backend();
    train.euclidean_opt = train.backend.DefaultEuclideanOptimizer()

    from mrmustard.concrete import circuit
    circuit.backend = Backend()
    
    from mrmustard.experimental import xptensor
    xptensor.backend = Backend()
    
    return 

activate_backend("tensorflow")

from mrmustard.concrete import *
from mrmustard.abstract import State
