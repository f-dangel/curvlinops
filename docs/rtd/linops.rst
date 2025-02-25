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

.. autoclass:: curvlinops.FisherType
   :members:
   :member-order: bysource

.. autoclass:: curvlinops.KFACType
   :members:
   :member-order: bysource

.. autoclass:: curvlinops.KFACLinearOperator
   :members: __init__, trace, det, logdet, frobenius_norm, state_dict, load_state_dict, from_state_dict

.. autoclass:: curvlinops.EKFACLinearOperator
   :members: __init__, trace, det, logdet, frobenius_norm, state_dict, load_state_dict, from_state_dict

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
   :members: __init__

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

.. autofunction:: curvlinops.hutchinson_trace

.. autofunction:: curvlinops.hutchpp_trace

.. autofunction:: curvlinops.xtrace

Diagonal approximation
======================

.. autofunction:: curvlinops.hutchinson_diag

.. autofunction:: curvlinops.xdiag

Frobenius norm approximation
============================

.. autoclass:: curvlinops.hutchinson_squared_fro

Experimental
============

The API of experimental features may be subject to changes, or they might become
deprecated.

.. autoclass:: curvlinops.experimental.ActivationHessianLinearOperator
   :members: __init__
