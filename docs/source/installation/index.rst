.. _installation:

Installation
============

Docker image
----------------------

A GPU-enabled docker image is available at

<insert link to Google Container Repository>

Installing from source
----------------------

We recommend installing CellMincer within a dedicated Python environment created via ``conda`` or ``venv`` to prevent potential dependency conflicts.

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellmincer python=3.10
  $ conda activate cellmincer

Install with ``pip`` from Git repository:

.. code-block:: console

   (cellmincer) $ pip install git+https://github.com/broadinstitute/CellMincer.git

Because CellMincer uses GPU computing for both training on and denoising voltage imaging, you may wish to confirm that your machine has GPUs with appropriate drivers installed and is CUDA-enabled (e.g. check that ``torch.cuda.is_available()`` returns ``True``).
