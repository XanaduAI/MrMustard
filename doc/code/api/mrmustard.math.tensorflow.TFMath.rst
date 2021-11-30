sf.math.tensorflow.TFMath
=========================

.. currentmodule:: mrmustard.math.tensorflow

.. autoclass:: TFMath
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

      ~TFMath.complex128
      ~TFMath.complex64
      ~TFMath.euclidean_opt
      ~TFMath.float32
      ~TFMath.float64

   .. autoattribute:: complex128
   .. autoattribute:: complex64
   .. autoattribute:: euclidean_opt
   .. autoattribute:: float32
   .. autoattribute:: float64

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

      ~TFMath.DefaultEuclideanOptimizer
      ~TFMath.J
      ~TFMath.Xmat
      ~TFMath.abs
      ~TFMath.add_at_modes
      ~TFMath.all_diagonals
      ~TFMath.arange
      ~TFMath.asnumpy
      ~TFMath.assign
      ~TFMath.astensor
      ~TFMath.atleast_1d
      ~TFMath.binomial_conditional_prob
      ~TFMath.block
      ~TFMath.boolean_mask
      ~TFMath.cast
      ~TFMath.clip
      ~TFMath.concat
      ~TFMath.conj
      ~TFMath.constraint_func
      ~TFMath.convolution
      ~TFMath.convolve_probs
      ~TFMath.convolve_probs_1d
      ~TFMath.cos
      ~TFMath.cosh
      ~TFMath.dagger
      ~TFMath.det
      ~TFMath.diag
      ~TFMath.diag_part
      ~TFMath.eigh
      ~TFMath.eigvals
      ~TFMath.eigvalsh
      ~TFMath.einsum
      ~TFMath.exp
      ~TFMath.expand_dims
      ~TFMath.expm
      ~TFMath.eye
      ~TFMath.gather
      ~TFMath.getitem
      ~TFMath.hash_tensor
      ~TFMath.hermite_renormalized
      ~TFMath.imag
      ~TFMath.inv
      ~TFMath.istensor
      ~TFMath.istrainable
      ~TFMath.left_matmul_at_modes
      ~TFMath.lgamma
      ~TFMath.log
      ~TFMath.loss_and_gradients
      ~TFMath.matmul
      ~TFMath.matvec
      ~TFMath.matvec_at_modes
      ~TFMath.maximum
      ~TFMath.minimum
      ~TFMath.new_constant
      ~TFMath.new_variable
      ~TFMath.norm
      ~TFMath.ones
      ~TFMath.ones_like
      ~TFMath.outer
      ~TFMath.pad
      ~TFMath.pinv
      ~TFMath.poisson
      ~TFMath.random_orthogonal
      ~TFMath.random_symplectic
      ~TFMath.real
      ~TFMath.reshape
      ~TFMath.riemann_to_symplectic
      ~TFMath.right_matmul_at_modes
      ~TFMath.rotmat
      ~TFMath.setitem
      ~TFMath.sin
      ~TFMath.single_mode_to_multimode_mat
      ~TFMath.single_mode_to_multimode_vec
      ~TFMath.sinh
      ~TFMath.sqrt
      ~TFMath.sqrtm
      ~TFMath.sum
      ~TFMath.svd
      ~TFMath.tensordot
      ~TFMath.tile
      ~TFMath.trace
      ~TFMath.transpose
      ~TFMath.unitary_to_orthogonal
      ~TFMath.update_add_tensor
      ~TFMath.update_tensor
      ~TFMath.xlogy
      ~TFMath.zeros
      ~TFMath.zeros_like

   .. automethod:: DefaultEuclideanOptimizer
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
   .. automethod:: boolean_mask
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
   .. automethod:: eigh
   .. automethod:: eigvals
   .. automethod:: eigvalsh
   .. automethod:: einsum
   .. automethod:: exp
   .. automethod:: expand_dims
   .. automethod:: expm
   .. automethod:: eye
   .. automethod:: gather
   .. automethod:: getitem
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
   .. automethod:: setitem
   .. automethod:: sin
   .. automethod:: single_mode_to_multimode_mat
   .. automethod:: single_mode_to_multimode_vec
   .. automethod:: sinh
   .. automethod:: sqrt
   .. automethod:: sqrtm
   .. automethod:: sum
   .. automethod:: svd
   .. automethod:: tensordot
   .. automethod:: tile
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: unitary_to_orthogonal
   .. automethod:: update_add_tensor
   .. automethod:: update_tensor
   .. automethod:: xlogy
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
