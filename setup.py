# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import platform

from setuptools import find_packages, setup

with open("mrmustard/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy==1.23.5",
    "scipy==1.8.0",
    "numba==0.56.4",
    "thewalrus==0.19.0",
    "tensorflow_macos==2.10.0" if platform.system() == "Darwin" else "tensorflow==2.10.1",
    "tensorflow-probability==0.18.0",
    "rich==10.15.1",
    "tqdm==4.62.3",
    "matplotlib==3.5.0",
]

extra_requirements = {
    "ray": ["ray[tune]", "scikit-optimize"],
}

info = {
    "name": "mrmustard",
    "version": version,
    "description": "Differentiable quantum Gaussian circuits",
    "url": "https://github.com/XanaduAI/mrmustard",
    "author": "Xanadu",
    "author_email": "filippo@xanadu.ai",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "install_requires": requirements,
    "extras_require": extra_requirements,
    "long_description": open("README.md", encoding="utf-8").read(),
    "long_description_content_type": "text/markdown",
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
