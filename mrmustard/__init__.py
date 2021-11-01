# this is the topmost __init__.py file of the mrmustard package

import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

def activate_backend(backend_name: str):
    "Activates the math backend in the physics module"
    Math = importlib.import_module("mrmustard.math." + backend_name).Math
    from mrmustard.physics import fock
    fock.math = Math()
    
    from mrmustard.physics import gaussian
    gaussian.math = Math()

    from mrmustard.utils import train
    train.math = Math()
    train.euclidean_opt = train.math.DefaultEuclideanOptimizer()

    from mrmustard.concrete import circuit
    circuit.math = Math()
    
    from mrmustard.experimental import xptensor
    xptensor.math = Math()

activate_backend("tensorflow")

# from mrmustard.lab import *
#from mrmustard.abstract import State
