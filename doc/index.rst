Mr Mustard Documentation
########################

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        p.lead.grey-text {
            margin-bottom: 30px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>

    <div class="container mt-2 mb-2">
        <div class="row container-fluid">
            <div class="col-lg-4 col-12 align-middle mb-2 text-center">
                <img src="_static/StateLearning.gif" class="img-fluid" alt="Responsive image" style="width:100%; max-width: 300px;"></img>
            </div>
            <div class="col-lg-8 col-12 align-middle mb-2">
                <p class="lead grey-text">
                    Mr Mustard: Your Universal Differentiable Toolkit for Quantum Optics
                </p>
        </div>
        <div class="row mt-3">

.. index-card::
    :link: introduction/basic_reference.html
    :name: Key Concepts
    :description: Learn about the main features of Mr Mustard

.. index-card::
    :link: development/development_guide.html
    :name: Developing
    :description: How you can contribute to Mr Mustard

.. index-card::
    :link: code/mm.html
    :name: API
    :description: Explore the Mr Mustard API

.. raw:: html

        </div>
    </div>

Features
========

🔄 **Universal Representation Compatibility**

* Initialize any component from any representation: ``Ket.from_quadrature(...)``, ``Channel.from_bargmann(...)``

..

* Convert between representations seamlessly: ``my_component.to_fock(...)``, ``my_component.to_quadrature(...)``

..

* Supported representations: Bargmann, Phase space, Characteristic functions, Quadrature, Fock

..

⚡ **Fast & Exact**

* State-of-the-art algorithms for Fock amplitudes of Gaussian components

..

* Exact computation up to arbitrary cutoff

..

* Batch processing support

..

🎯 **Built-in Optimization**

* Differentiable with respect to all parameters

..

* Riemannian optimization on symplectic/unitary/orthogonal groups

..

* Cost functions can mix different representations

..

🧩 **Flexible Circuit Construction**

* Contract components in any order

..

* Linear superpositions of compatible objects

..

* Plug-and-play backends (``numpy``, ``tensorflow``, ``jax``)

.. toctree::
   :maxdepth: 1
   :caption: Using Mr Mustard
   :hidden:

   introduction/basic_reference

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/development_guide
   development/research
   development/release_notes.md

.. toctree::
   :maxdepth: 1
   :caption: Mr Mustard API
   :hidden:

   code/mm
   code/lab
   code/physics
   code/math
   code/training
   code/utils
