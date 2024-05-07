The Bargmann representation theory
==========================================

The Bargmann representation of the state can be obtained as projecting the state on to the Bargmann basis.

The Bargmann basis :math:`|z\rangle_b` can be defined from the coherent state basis :math:`|z\rangle_c` 

.. math:: 
    |z\rangle_b = e^\frac{|z|^2}{2}|z\rangle_c = \sum_n \frac{z^n}{\sqrt{n!}}|n\rangle_f.

Any Gaussian objects :math:`O` can be written in the Bargmann basis as a Gaussian exponential function, parametrized by a matrix :math:`A`, a vector :math:`b` and a scalar :math:`c`, which is called ``triples`` through all the documentations in MM.

A ``n``-mode pure Gaussian state :math:`|\psi\rangle` can be defined as

..math::
    \langle \alpha | \psi \rangle = 

.. toctree::
    :maxdepth: 1

    utils/triples
    utils/gaussian_integral

.. currentmodule:: mrmustard.physics.bargmann

.. automodapi:: mrmustard.physics.bargmann
    :no-heading:
    :include-all-objects: