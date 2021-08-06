CellMincer
===========

CellMincer is a software package for learning self-supervised denoising models for voltage-imaging movies.

Installation
============

```
git clone https://github.com/broadinstitute/CellMincer.git
pip install -e CellMincer/
```

Modules
=======

CellMincer provides the following tools:

`preprocess` adaptively dejitters, detrends, and estimates the PG-noise in a raw movie.
`cellmincer preprocess -i <path_to_raw_movie> -o <output_directory> --manifest <path_to_raw_movie_manifest> --config <path_to_preprocessing_config>`

`feature` computes a set of global features spanning the xy-dimensions, including cross-correlations between adjacent pixels. Passing the --no-active-range flag disables active range detection. Disabling this is preferable when the movie lacks well-defined periods of activity and dormancy.
`cellmincer feature -i <path_to_detrended_movie> -o <output_directory> [--no-active-range]`

`train` trains a denoising model using noise2self-like loss. Multiple input datasets can be provided with successive `-i` arguments.
`cellmincer train -i <paths_to_dataset_directories> -o <output_directory> --config <path_to_training_config>`

`denoise` uses a trained model to denoise VI-movies. If unspecified, the output directory will be the dataset directory.
`cellmincer denoise -i <path_to_dataset_directory> [-o <output_directory>] <path_to_config>`

Template `.yaml` configurations are provided in `configs/`.
