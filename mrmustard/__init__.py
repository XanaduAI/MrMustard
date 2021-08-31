from rich.pretty import install

install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"


def get_env(env_name: str):
    import importlib

    return importlib.import_module("mrmustard.backends." + env_name).Backend


Backend = get_env("tensorflow")  # default backend

# TODO
# Backend = get_env("pytorch")
# Backend = get_env("jax")
# Backend = get_env("numpy")
# Backend = get_env("tinygrad?")

from mrmustard.concrete import *
