sf.math.torch.TorchMath
=======================

.. currentmodule:: mrmustard.math.torch

.. autoclass:: TorchMath
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

      ~TorchMath.complex128
      ~TorchMath.complex64
      ~TorchMath.euclidean_opt
      ~TorchMath.float32
      ~TorchMath.float64

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

      ~TorchMath.DefaultEuclideanOptimizer
      ~TorchMath.J
      ~TorchMath.Xmat
      ~TorchMath.abs
      ~TorchMath.add_at_modes
      ~TorchMath.all_diagonals
      ~TorchMath.arange
      ~TorchMath.asnumpy
      ~TorchMath.assign
      ~TorchMath.astensor
      ~TorchMath.atleast_1d
      ~TorchMath.binomial_conditional_prob
      ~TorchMath.block
      ~TorchMath.boolean_mask
      ~TorchMath.cast
      ~TorchMath.clip
      ~TorchMath.concat
      ~TorchMath.conj
      ~TorchMath.constraint_func
      ~TorchMath.convolution
      ~TorchMath.convolve_probs
      ~TorchMath.convolve_probs_1d
      ~TorchMath.cos
      ~TorchMath.cosh
      ~TorchMath.dagger
      ~TorchMath.det
      ~TorchMath.diag
      ~TorchMath.diag_part
      ~TorchMath.eigvals
      ~TorchMath.eigvalsh
      ~TorchMath.einsum
      ~TorchMath.exp
      ~TorchMath.expand_dims
      ~TorchMath.expm
      ~TorchMath.eye
      ~TorchMath.gather
      ~TorchMath.hash_tensor
      ~TorchMath.hermite_renormalized
      ~TorchMath.imag
      ~TorchMath.inv
      ~TorchMath.istensor
      ~TorchMath.istrainable
      ~TorchMath.left_matmul_at_modes
      ~TorchMath.lgamma
      ~TorchMath.log
      ~TorchMath.loss_and_gradients
      ~TorchMath.matmul
      ~TorchMath.matvec
      ~TorchMath.matvec_at_modes
      ~TorchMath.maximum
      ~TorchMath.minimum
      ~TorchMath.new_constant
      ~TorchMath.new_variable
      ~TorchMath.norm
      ~TorchMath.ones
      ~TorchMath.ones_like
      ~TorchMath.outer
      ~TorchMath.pad
      ~TorchMath.pinv
      ~TorchMath.poisson
      ~TorchMath.random_orthogonal
      ~TorchMath.random_symplectic
      ~TorchMath.real
      ~TorchMath.reshape
      ~TorchMath.riemann_to_symplectic
      ~TorchMath.right_matmul_at_modes
      ~TorchMath.rotmat
      ~TorchMath.sin
      ~TorchMath.single_mode_to_multimode_mat
      ~TorchMath.single_mode_to_multimode_vec
      ~TorchMath.sinh
      ~TorchMath.sqrt
      ~TorchMath.sqrtm
      ~TorchMath.sum
      ~TorchMath.svd
      ~TorchMath.tensordot
      ~TorchMath.tile
      ~TorchMath.trace
      ~TorchMath.transpose
      ~TorchMath.unitary_to_orthogonal
      ~TorchMath.update_add_tensor
      ~TorchMath.update_tensor
      ~TorchMath.xlogy
      ~TorchMath.zeros
      ~TorchMath.zeros_like

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
   .. automethod:: eigvals
   .. automethod:: eigvalsh
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
