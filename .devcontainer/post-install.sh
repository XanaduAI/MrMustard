#! /bin/zsh

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r doc/requirements.txt
pip install ray
pip install -e .
