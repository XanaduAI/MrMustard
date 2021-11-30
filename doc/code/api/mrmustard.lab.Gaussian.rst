sf.lab.Gaussian
===============

.. currentmodule:: mrmustard.lab

.. autoclass:: Gaussian
   :show-inheritance:

   .. raw:: html

      <a class="attr-details-header collapse-header" data-toggle="collapse" href="#attrDetails" aria-expanded="false" aria-controls="attrDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Attributes
         </h2>
      </a>
      <div class="collapse" id="attrDetails">

   .. autosummary::
      :nosignatures:

      ~Gaussian.cov
      ~Gaussian.cutoffs
      ~Gaussian.fock
      ~Gaussian.is_gaussian
      ~Gaussian.is_mixed
      ~Gaussian.is_pure
      ~Gaussian.means
      ~Gaussian.modes
      ~Gaussian.num_modes
      ~Gaussian.number_cov
      ~Gaussian.number_means
      ~Gaussian.number_stdev
      ~Gaussian.purity
      ~Gaussian.shape
      ~Gaussian.trainable_parameters

   .. autoattribute:: cov
   .. autoattribute:: cutoffs
   .. autoattribute:: fock
   .. autoattribute:: is_gaussian
   .. autoattribute:: is_mixed
   .. autoattribute:: is_pure
   .. autoattribute:: means
   .. autoattribute:: modes
   .. autoattribute:: num_modes
   .. autoattribute:: number_cov
   .. autoattribute:: number_means
   .. autoattribute:: number_stdev
   .. autoattribute:: purity
   .. autoattribute:: shape
   .. autoattribute:: trainable_parameters

   .. raw:: html

      </div>

   .. raw:: html

      <a class="meth-details-header collapse-header" data-toggle="collapse" href="#methDetails" aria-expanded="false" aria-controls="methDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Methods
         </h2>
      </a>
      <div class="collapse" id="methDetails">

   .. autosummary::

      ~Gaussian.__call__
      ~Gaussian.dm
      ~Gaussian.fock_probabilities
      ~Gaussian.get_modes
      ~Gaussian.ket

   .. automethod:: __call__
   .. automethod:: dm
   .. automethod:: fock_probabilities
   .. automethod:: get_modes
   .. automethod:: ket

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
