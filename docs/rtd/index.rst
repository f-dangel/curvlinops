ViViT
=================================

ViViT is a collection of numerical tricks to efficiently access curvature from
the generalized Gauss-Newton (GGN) matrix based on its low-rank structure.

.. code:: bash

  pip install vivit-for-pytorch@git+https://github.com/f-dangel/vivit.git#egg=vivit-for-pytorch

It is designed to be used with `BackPACK
<http://www.github.com/f-dangel/backpack>`_ and can compute

- GGN eigenvalues (:ref:`basic example <Computing GGN eigenvalues>`)

- GGN eigenpairs (eigenvalues + eigenvector, :ref:`basic example <Computing GGN
  eigenpairs>`)

- 1ˢᵗ- and 2ⁿᵈ-order directional derivatives along GGN eigenvectors (:ref:`basic
  example <Computing directional derivatives along GGN eigenvectors>`)

- Directionally damped Newton steps (:ref:`basic example <Computing
  directionally damped Newton steps>`)

These operations can further approximate the GGN to reduce cost via
sub-sampling, Monte-Carlo approximation, and block-diagonal approximation.

.. toctree::
	:maxdepth: 2
	:caption: Getting started

.. toctree::
	:maxdepth: 2
	:caption: ViViT

	basic_usage/index
