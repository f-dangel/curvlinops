CurvLinOps
=================================

This library implements :code:`scipy`
:class:`LinearOperator <scipy.sparse.linalg.LinearOperator>` s for deep learning
matrices, such as

- the Hessian

- the Fisher/generalized Gauss-Newton (GGN)

Matrix-vector products are carried out in PyTorch, i.e. potentially on a GPU.
The library supports defining these matrices not only on a mini-batch, but on
data sets (looping over batches during a :meth:`matvec
<scipy.sparse.linalg.LinearOperator.matvec>` operation).

You can plug these linear operators into :code:`scipy`, while carrying out the
heavy lifting (matrix-vector multiplies) in PyTorch on GPU. My favorite example
for such a routine is :func:`eigsh <scipy.sparse.linalg.eigsh>` that lets you
compute a subset of eigenpairs.

Installation
------------

.. code:: bash

  pip install curvlinops-for-pytorch


.. toctree::
	:maxdepth: 2
	:caption: Getting started

	usage


.. toctree::
	:maxdepth: 2
	:caption: CurvLinOps

	linops
	basic_usage/index

.. toctree::
	:caption: Internals

	internals
