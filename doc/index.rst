Mr. Mustard's Documentation
=======================================

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

   <div class="row container-fluid">
      <div class="col-lg-4 col-12 align-middle mb-2 text-center">
          <img src="_static/StateLearning.gif" class="img-fluid" alt="Responsive image" style="width:100%; max-width: 300px;"></img>
      </div>
      <div class="col-lg-8 col-12 align-middle mb-2">
        <p class="lead grey-text">
            MrMustard is a differentiable bridge between phase space and Fock space with rich functionality in both representations.
        </p>
   </div>

Features
========

MrMustard supports in fully differentiable way:

* Phase space representation of Gaussian states and Gaussian channels on an arbitrary number of modes

..

* Exact Fock representation of any Gaussian circuit and any Gaussian state up to an arbitrary cutoff

..

* Beam splitter, MZ interferometer, squeezer, displacement, phase rotation, bosonic lossy channel, thermal channel, [more to come...]

..

* General Gaussian N-mode gate and general N-mode Interferometer with dedicated symplectic and orthogonal optimization routines

..

* Photon number moments

..

* PNR detectors, Threshold detectors with trainable quantum efficiency and dark counts

..

* Homodyne, Heterodyne and Generaldyne Gaussian measurements

..

* An optimizer with a spiffy progress bar

..

* A composable Circuit object

..

* Plug-and-play backends (TF and Pytorch [Upcoming])

..

* An abstraction layer `XPTensor` for seamless symplectic algebra

..


.. toctree::
   :maxdepth: 1
   :caption: Using Mr. Mustard
   :hidden:

   introduction/basic_reference

.. toctree::
   :maxdepth: 1
   :caption: Mr. Mustard API
   :hidden:

   code/lab
   code/physics
   code/math
