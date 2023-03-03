#! /bin/zsh

pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir-r requirements-dev.txt
pip install --no-cache-dir -r doc/requirements.txt
pip install ray
pip install --no-cache-dir -e .
