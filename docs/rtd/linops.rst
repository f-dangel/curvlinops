Linear operators
================


Hessian
-------

.. autoclass:: curvlinops.HessianLinearOperator
   :members: __init__

Generalized Gauss-Newton
------------------------

.. autoclass:: curvlinops.GGNLinearOperator
   :members: __init__

Fisher (approximate)
--------------------

.. autoclass:: curvlinops.FisherMCLinearOperator
   :members: __init__

.. autoclass:: curvlinops.KFACLinearOperator
   :members: __init__, to_device, torch_matmat, torch_matvec, trace, det, logdet, frobenius_norm

Uncentered gradient covariance (empirical Fisher)
-------------------------------------------------

.. autoclass:: curvlinops.EFLinearOperator
   :members: __init__

Jacobians
---------

.. autoclass:: curvlinops.JacobianLinearOperator
   :members: __init__

.. autoclass:: curvlinops.TransposedJacobianLinearOperator
   :members: __init__

Inverses
--------

.. autoclass:: curvlinops.CGInverseLinearOperator
   :members: __init__, set_cg_hyperparameters

.. autoclass:: curvlinops.LSMRInverseLinearOperator
   :members: __init__, set_lsmr_hyperparameters, matvec_with_info

.. autoclass:: curvlinops.NeumannInverseLinearOperator
   :members: __init__, set_neumann_hyperparameters

.. autoclass:: curvlinops.KFACInverseLinearOperator
   :members: __init__, torch_matmat, torch_matvec

Sub-matrices
------------

.. autoclass:: curvlinops.SubmatrixLinearOperator
   :members: __init__, set_submatrix

Spectral density approximation
==============================

.. autofunction:: curvlinops.lanczos_approximate_spectrum

.. autofunction:: curvlinops.lanczos_approximate_log_spectrum

.. autoclass:: curvlinops.LanczosApproximateSpectrumCached
   :members: __init__, approximate_spectrum

Trace approximation
===================

.. autoclass:: curvlinops.HutchinsonTraceEstimator
   :members: __init__, sample

.. autoclass:: curvlinops.HutchPPTraceEstimator
   :members: __init__, sample

Diagonal approximation
======================

.. autoclass:: curvlinops.HutchinsonDiagonalEstimator
   :members: __init__, sample

Frobenius norm approximation
============================

.. autoclass:: curvlinops.HutchinsonSquaredFrobeniusNormEstimator
   :members: __init__, sample

Experimental
============

The API of experimental features may be subject to changes, or they might become
deprecated.

.. autoclass:: curvlinops.experimental.ActivationHessianLinearOperator
   :members: __init__
