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

Uncentered gradient covariance (empirical Fisher)
-------------------------------------------------

.. autoclass:: curvlinops.EFLinearOperator
   :members: __init__

Inverses
--------

.. autoclass:: curvlinops.CGInverseLinearOperator
   :members: __init__, set_cg_hyperparameters

Spectral density approximation
==============================

.. autofunction:: curvlinops.lanczos_approximate_spectrum

.. autofunction:: curvlinops.lanczos_approximate_log_spectrum

.. autoclass:: curvlinops.LanczosApproximateSpectrumCached
   :members: __init__, approximate_spectrum
