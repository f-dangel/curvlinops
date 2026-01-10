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
   :members: __init__, trace, det, logdet, frobenius_norm, state_dict, load_state_dict, from_state_dict

.. autoclass:: curvlinops.EKFACLinearOperator
   :members: __init__, trace, det, logdet, frobenius_norm, state_dict, load_state_dict, from_state_dict

.. autoclass:: curvlinops.FisherType

.. autoclass:: curvlinops.KFACType

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
   :members: __init__

.. autoclass:: curvlinops.LSMRInverseLinearOperator
   :members: __init__

.. autoclass:: curvlinops.NeumannInverseLinearOperator
   :members: __init__

.. autoclass:: curvlinops.KFACInverseLinearOperator
   :members: __init__

Sub-matrices
------------

.. autoclass:: curvlinops.SubmatrixLinearOperator
   :members: __init__, set_submatrix

Spectral density approximation
==============================

.. note::
   This functionality currently expects SciPy ``LinearOperator`` instances.

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

Experimental features may be subject to changes or become deprecated.

.. autoclass:: curvlinops.experimental.ActivationHessianLinearOperator
   :members: __init__
