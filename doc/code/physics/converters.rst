The converters between different representations
==========================================

Though MrMustard runs internally only with Fock and Bargmann representations, it supports different representations in the initialization and the result part.

The conversions exist:

* From Bargmann representation to Fock representation (for all quantum objects);
* From Bargmann representation to phase space representation (for only quantum states);
* From phase space representation to Bargmann representation (for only quantum states);

From Bargmann representation to Fock representation conversion is realized by using the ``hermite_renormalized`` function, which can be considered as a Map gate as well.

     ---------------
----| BargmannToFock|----
     ---------------
   
If there is a single-mode pure state :math:`|\psi\rangle`, which can be denoted as

  --------------------
 |:math:`|\psi\rangle`|----
  --------------------


.. currentmodule:: mrmustard.physics.converters

.. automodapi:: mrmustard.physics.converters
    :no-heading:
    :include-all-objects: