CellMincer
===========

CellMincer is a self-supervised machine learning framework for voltage imaging denoising models. A visual comparison of voltage imaging data before and after CellMincer below:

![Alt Text](./docs/source/_static/graphics/raw_denoised_traces.gif)

A schmatic of Optopatch data and CellMincer's high-level architecture below:

![Alt Text](./docs/source/_static/graphics/fig1-cellmincer-schem.png)

# Documentation and resources

Coming soon!

# Data availability

Raw and denoised voltage imaging datasets, as well as pretrained models and example configurations, can be found at this Google bucket: `gs://broad-dsp-cellmincer-data` (refer to [downloading from Google Cloud storage](https://cloud.google.com/storage/docs/uploads-downloads)).

# Preprint and citation
The bioRxiv preprint for CellMincer [can be found here](https://www.biorxiv.org/content/10.1101/2024.04.12.589298v1). The BibTeX citation:
```
@article {Wang2024.04.12.589298,
	author = {Brice Wang and Tianle Ma and Theresa Chen and Trinh Nguyen and Ethan Crouse and Stephen J Fleming and Alison S Walker and Vera Valakh and Ralda Nehme and Evan W Miller and Samouil L Farhi and Mehrtash Babadi},
	title = {Robust self-supervised denoising of voltage imaging data using CellMincer},
	elocation-id = {2024.04.12.589298},
	year = {2024},
	doi = {10.1101/2024.04.12.589298},
	URL = {https://www.biorxiv.org/content/early/2024/04/15/2024.04.12.589298},
	eprint = {https://www.biorxiv.org/content/early/2024/04/15/2024.04.12.589298.full.pdf},
	journal = {bioRxiv}
}
```

# Related Github repositories

[CellMincerPaperAnalysis](https://github.com/cellarium-ai/CellMincerPaperAnalysis) contains notebooks for reproducing the analysis and figures in the preprint.

[Optosynth](https://github.com/cellarium-ai/Optosynth) is a voltage imaging simulation framework which generates synthetic data used to optimize and benchmark CellMincer.