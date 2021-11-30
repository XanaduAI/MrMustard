sf.lab.Circuit
==============

.. currentmodule:: mrmustard.lab

.. autoclass:: Circuit
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

      ~Circuit.XYd
      ~Circuit.XYd_dual
      ~Circuit.X_matrix
      ~Circuit.X_matrix_dual
      ~Circuit.Y_matrix
      ~Circuit.Y_matrix_dual
      ~Circuit.bell
      ~Circuit.d_vector
      ~Circuit.d_vector_dual
      ~Circuit.is_gaussian
      ~Circuit.is_unitary
      ~Circuit.modes
      ~Circuit.num_modes
      ~Circuit.trainable_parameters

   .. autoattribute:: XYd
   .. autoattribute:: XYd_dual
   .. autoattribute:: X_matrix
   .. autoattribute:: X_matrix_dual
   .. autoattribute:: Y_matrix
   .. autoattribute:: Y_matrix_dual
   .. autoattribute:: bell
   .. autoattribute:: d_vector
   .. autoattribute:: d_vector_dual
   .. autoattribute:: is_gaussian
   .. autoattribute:: is_unitary
   .. autoattribute:: modes
   .. autoattribute:: num_modes
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

      ~Circuit.U
      ~Circuit.__call__
      ~Circuit.append
      ~Circuit.choi
      ~Circuit.dual
      ~Circuit.extend
      ~Circuit.reset
      ~Circuit.transform_fock
      ~Circuit.transform_gaussian

   .. automethod:: U
   .. automethod:: __call__
   .. automethod:: append
   .. automethod:: choi
   .. automethod:: dual
   .. automethod:: extend
   .. automethod:: reset
   .. automethod:: transform_fock
   .. automethod:: transform_gaussian

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
