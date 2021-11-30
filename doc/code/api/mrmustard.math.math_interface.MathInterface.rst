sf.math.math_interface.MathInterface
====================================

.. currentmodule:: mrmustard.math.math_interface

.. autoclass:: MathInterface
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

      ~MathInterface.euclidean_opt

   .. autoattribute:: euclidean_opt

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

      ~MathInterface.J
      ~MathInterface.Xmat
      ~MathInterface.abs
      ~MathInterface.add_at_modes
      ~MathInterface.all_diagonals
      ~MathInterface.arange
      ~MathInterface.asnumpy
      ~MathInterface.assign
      ~MathInterface.astensor
      ~MathInterface.atleast_1d
      ~MathInterface.binomial_conditional_prob
      ~MathInterface.block
      ~MathInterface.cast
      ~MathInterface.clip
      ~MathInterface.concat
      ~MathInterface.conj
      ~MathInterface.constraint_func
      ~MathInterface.convolution
      ~MathInterface.convolve_probs
      ~MathInterface.convolve_probs_1d
      ~MathInterface.cos
      ~MathInterface.cosh
      ~MathInterface.dagger
      ~MathInterface.det
      ~MathInterface.diag
      ~MathInterface.diag_part
      ~MathInterface.eigvals
      ~MathInterface.einsum
      ~MathInterface.exp
      ~MathInterface.expand_dims
      ~MathInterface.expm
      ~MathInterface.eye
      ~MathInterface.gather
      ~MathInterface.hash_tensor
      ~MathInterface.hermite_renormalized
      ~MathInterface.imag
      ~MathInterface.inv
      ~MathInterface.istensor
      ~MathInterface.istrainable
      ~MathInterface.left_matmul_at_modes
      ~MathInterface.lgamma
      ~MathInterface.log
      ~MathInterface.loss_and_gradients
      ~MathInterface.matmul
      ~MathInterface.matvec
      ~MathInterface.matvec_at_modes
      ~MathInterface.maximum
      ~MathInterface.minimum
      ~MathInterface.new_constant
      ~MathInterface.new_variable
      ~MathInterface.norm
      ~MathInterface.ones
      ~MathInterface.ones_like
      ~MathInterface.outer
      ~MathInterface.pad
      ~MathInterface.pinv
      ~MathInterface.poisson
      ~MathInterface.random_orthogonal
      ~MathInterface.random_symplectic
      ~MathInterface.real
      ~MathInterface.reshape
      ~MathInterface.riemann_to_symplectic
      ~MathInterface.right_matmul_at_modes
      ~MathInterface.rotmat
      ~MathInterface.sin
      ~MathInterface.single_mode_to_multimode_mat
      ~MathInterface.single_mode_to_multimode_vec
      ~MathInterface.sinh
      ~MathInterface.sqrt
      ~MathInterface.sqrtm
      ~MathInterface.sum
      ~MathInterface.tensordot
      ~MathInterface.tile
      ~MathInterface.trace
      ~MathInterface.transpose
      ~MathInterface.unitary_to_orthogonal
      ~MathInterface.update_add_tensor
      ~MathInterface.update_tensor
      ~MathInterface.zeros
      ~MathInterface.zeros_like

   .. automethod:: J
   .. automethod:: Xmat
   .. automethod:: abs
   .. automethod:: add_at_modes
   .. automethod:: all_diagonals
   .. automethod:: arange
   .. automethod:: asnumpy
   .. automethod:: assign
   .. automethod:: astensor
   .. automethod:: atleast_1d
   .. automethod:: binomial_conditional_prob
   .. automethod:: block
   .. automethod:: cast
   .. automethod:: clip
   .. automethod:: concat
   .. automethod:: conj
   .. automethod:: constraint_func
   .. automethod:: convolution
   .. automethod:: convolve_probs
   .. automethod:: convolve_probs_1d
   .. automethod:: cos
   .. automethod:: cosh
   .. automethod:: dagger
   .. automethod:: det
   .. automethod:: diag
   .. automethod:: diag_part
   .. automethod:: eigvals
   .. automethod:: einsum
   .. automethod:: exp
   .. automethod:: expand_dims
   .. automethod:: expm
   .. automethod:: eye
   .. automethod:: gather
   .. automethod:: hash_tensor
   .. automethod:: hermite_renormalized
   .. automethod:: imag
   .. automethod:: inv
   .. automethod:: istensor
   .. automethod:: istrainable
   .. automethod:: left_matmul_at_modes
   .. automethod:: lgamma
   .. automethod:: log
   .. automethod:: loss_and_gradients
   .. automethod:: matmul
   .. automethod:: matvec
   .. automethod:: matvec_at_modes
   .. automethod:: maximum
   .. automethod:: minimum
   .. automethod:: new_constant
   .. automethod:: new_variable
   .. automethod:: norm
   .. automethod:: ones
   .. automethod:: ones_like
   .. automethod:: outer
   .. automethod:: pad
   .. automethod:: pinv
   .. automethod:: poisson
   .. automethod:: random_orthogonal
   .. automethod:: random_symplectic
   .. automethod:: real
   .. automethod:: reshape
   .. automethod:: riemann_to_symplectic
   .. automethod:: right_matmul_at_modes
   .. automethod:: rotmat
   .. automethod:: sin
   .. automethod:: single_mode_to_multimode_mat
   .. automethod:: single_mode_to_multimode_vec
   .. automethod:: sinh
   .. automethod:: sqrt
   .. automethod:: sqrtm
   .. automethod:: sum
   .. automethod:: tensordot
   .. automethod:: tile
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: unitary_to_orthogonal
   .. automethod:: update_add_tensor
   .. automethod:: update_tensor
   .. automethod:: zeros
   .. automethod:: zeros_like

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
