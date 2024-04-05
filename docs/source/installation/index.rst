.. _installation:

Installation
============

Currently, the only method of accessing CellMincer locally is to install it from source. We will update this guide when a ``pip`` version becomes available.

Installing from source
----------------------

While not strictly required, initializing a dedicated environment for CellMincer via ``conda`` or ``venv`` may prevent dependency conflicts with your existing Python installation.

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellmincer python=3.7
  $ conda activate cellmincer

Clone this repository and install CellMincer (in editable ``-e`` mode):

.. code-block:: console

   (cellmincer) $ git clone https://github.com/broadinstitute/CellMincer.git
   (cellmincer) $ pip install -e CellMincer

Because CellMincer uses GPU computing for its neural networks, you may wish to confirm that your PyTorch installation recognizes your GPUs and the accompanying CUDA drivers.
