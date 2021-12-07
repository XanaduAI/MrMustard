Mr Mustard's Documentation
=======================================

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

   <style>
      #right-column.card {
         box-shadow: none!important;
      }
      #right-column.card:hover {
         box-shadow: none!important;
      }
      .breadcrumb {
         display: none;
      }
      h1 {
         text-align: center;
         margin-bottom: 15px;
      }
      .footer-relations {
         border-top: 0px;
      }
   </style>
   <div class="row container-fluid">
      <div class="col-lg-4 col-12 align-middle mb-2 text-center">
          <img src="_static/StateLearning.gif" class="img-fluid" alt="Responsive image" style="width:100%; max-width: 300px;"></img>
      </div>
      <div class="col-lg-8 col-12 align-middle mb-2">
        <p class="lead grey-text">
            MrMustard is a differentiable bridge between phase space and Fock space with rich functionality in both representations.
        </p>
   </div>
   <div style='clear:both'></div>
    <div class="container mt-2 mb-2">
        <div class="row mt-3">
            <div class="col-lg-4 mb-2 adlign-items-stretch">
                <a href="introduction/basic_reference.html">
                    <div class="card rounded-lg" style="height:100%;">
                        <div class="d-flex">
                            <div>
                                <h3 class="card-title pl-3 mt-4">
                                Key concepts
                                </h3>
                                <p class="mb-3 grey-text px-3">
                                    Learn about Mr Mustard's main features <i class="fas fa-angle-double-right"></i>
                                </p>
                            </div>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="development/development_guide.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            Developing
                            </h3>
                            <p class="mb-3 grey-text px-3">How you can contribute to Mr Mustard <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="code/mm.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            API
                            </h3>
                            <p class="mb-3 grey-text px-3">Explore Mr Mustard's API <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
        </div>
    </div>

Features
========

MrMustard supports in fully differentiable way:

* Phase space representation of Gaussian states and Gaussian channels on an arbitrary number of modes

..

* Exact Fock representation of any Gaussian circuit and any Gaussian state up to an arbitrary cutoff

..

* Beam splitter, MZ interferometer, squeezer, displacement, phase rotation, bosonic lossy channel, thermal channel, more to come..

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

* Plug-and-play backends (TF and Pytorch)

..

* An abstraction layer `XPTensor` for seamless symplectic algebra

..


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
