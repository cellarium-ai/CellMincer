.. _introduction:

What is CellMincer?
===================

CellMincer is a software package for training and deploying self-supervised denoising models for voltage imaging.

Overview
--------
Voltage imaging, like other forms of fluorescence imaging, uses a fluorescent indicator to report electrophysiological activity. Imaging platforms like Optopatch can stimulate and record a field of neurons in parallel, but shot noise and camera noise limit their reliability in reconstructing small-scale electrophysiological features.

CellMincer leverages strong spatial correlations to denoise voltage imaging using a deep neural network architecture. Using self-supervised learning, CellMincer can train on datasets of interest without needing a source of ground truth before denoising those same datasets. Alternatively, pretrained CellMincer models typically generalize well to new data, and its training scheme can be adapted for fine-tuning existing models. See the :ref:`Usage <usage>` section for examples of these use-cases.

.. figure:: ../_static/graphics/raw_denoised_traces.gif
    :class: with-border

    Sample Optopatch recording before and after CellMincer denoising, with several associated neuron traces.
