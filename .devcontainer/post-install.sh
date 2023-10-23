#! /bin/zsh

poetry install --all-extras --with dev,doc
julia --project="julia_pkg" -e "using Pkg; Pkg.instantiate()
