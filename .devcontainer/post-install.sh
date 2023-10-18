#! /bin/zsh

poetry install --all-extras --with dev,doc
julia -e "using Pkg; Pkg.add(\"PyCall\"); Pkg.add(\"MultiFloats\")"