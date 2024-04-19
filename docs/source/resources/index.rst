.. _resources:

Resources
=========

Refer to `downloading from Google Cloud storage <https://cloud.google.com/storage/docs/uploads-downloads>`_

Example configuration YAMLs
---------------------------

| Preprocessing YAML configuration sample file
| ``gs://broad-dsp-cellmincer-data/configs/preprocess/optosynth.yaml``

| Training YAML configuration sample file
| ``gs://broad-dsp-cellmincer-data/configs/train/default.yaml``

Please modify these YAML files as appropriate for your run. The complete preprocessing and training YAML configuration options and their descriptions can be found under :ref:`Resources <resources>`.

Trained models
--------------

| Pretrained CellMincer model on 5 Optosynth datasets simulated with :math:`Q=50` photons per fluorophore, which can be `found here <gs://broad-dsp-cellmincer-data/Optosynth/raw/>`__. See :ref:`our preprint <citation>` for a detailed explanation of the Optosynth data generation process.
| ``gs://broad-dsp-cellmincer-data/models/optosynth.ckpt``

| Pretrained CellMincer model on 10 Optopatch datasets, which can be `found here <gs://broad-dsp-cellmincer-data/FarhiOptopatch/>`__. [#farhi]_
| ``gs://broad-dsp-cellmincer-data/models/optopatch10.ckpt``

| Pretrained CellMincer model on 26 Optopatch datasets using BeRST fluorescence, which can be `found here <gs://broad-dsp-cellmincer-data/PairedBeRST/raw/>`__. [#miller]_
| ``gs://broad-dsp-cellmincer-data/models/pairedberst.ckpt``

.. rubric:: Footnotes

.. [#farhi]
   Data provided by Sami Farhi, Spatial Technology Platform (STP), Broad Institute of MIT and Harvard.
.. [#miller]
   Data provided by Evan Miller, Departments of Molecular \& Cell Biology and Chemistry and Helen Wills Neuroscience Institute, UC Berkeley.
