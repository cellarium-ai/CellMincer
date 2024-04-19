.. _introduction:

What is CellMincer?
===================

CellMincer is a software package for training and deploying self-supervised denoising models for voltage imaging.

What is voltage imaging?
------------------------
Voltage imaging, like other forms of fluorescence imaging, uses a fluorescent voltage indicator to report electrophysiological activity. Fluorescent voltage indicators are diverse, including voltage sensitive dyes (such as BeRST [#BeRST]_) and genetically-encoded voltage indicators (such as QuasAr [#QuasAr]_). Together with genetically encoded light gated cation channels (e.g. Channelrhodopsins-2 [#Lightgate]_), researchers can optically manipulate and record a large field of neurons in parallel. The true potential of this powerful toolbox, however, is often constrained by a low signal-to-noise ratio (SNR) due to shot noise and camera noise, ultimately limiting their application in reconstructing small-scale EP features.

We introduce CellMincer, a novel self-supervised deep learning method designed specifically for denoising voltage imaging datasets. CellMincer operates on the principle of masking and predicting sparse sets of pixels across short temporal windows and conditions the denoiser on precomputed spatiotemporal auto-correlations to effectively model long-range dependencies without the need for large temporal denoising contexts. CellMincer can jointly train on many datasets across multiple GPUs to produce a highly generalizable model, but satisfactory results can be achieved by training on a single voltage imaging dataset with as few as 5000 frames. Because of the model's self-supervised training scheme, the dataset to be denoised can also serve as the model's only training data. See the :ref:`Quick-start tutorial <tutorial>` section for examples of these use-cases.

.. figure:: ../_static/graphics/raw_denoised_traces.gif
    :class: with-border

    Sample Optopatch recording before and after CellMincer denoising, with several associated neuron traces.

.. rubric:: Footnotes

.. [#BeRST]
   Huang, Yi-Lin, Alison S. Walker, and Evan W. Miller. "A photostable silicon rhodamine platform for optical voltage sensing." Journal of the American Chemical Society 137.33 (2015): 10767-10776.
.. [#QuasAr]
   Hochbaum, Daniel R., et al. "All-optical electrophysiology in mammalian neurons using engineered microbial rhodopsins." Nature Methods 11.8 (2014): 825-833.
.. [#Lightgate]
   Nagel, Georg, et al. "Channelrhodopsin-2, a directly light-gated cation-selective membrane channel." Proceedings of the National Academy of Sciences 100.24 (2003): 13940-13945.