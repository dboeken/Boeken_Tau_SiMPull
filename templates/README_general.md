
# BOEKEN_Tau-SiMPull

This repository contains the analysis code associated with the Tau SiMPull project, led by Dorothea Böken. This manuscript has been submitted for publication under the title **"#####"**.

This manuscript has been submitted as a preprint via BioRxiv [here](biorxiv/link). A link to the final version will be provided upon publication.

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> 3.10.5). For specific package requirements, see the environment.yml file, or  create a new conda environment containing all packages by running ```conda create -f environment.yml```. 

## Raw data

For convenience, example datasets are provided here under the ```data``` folder. These data may be used to explore the workflows presented here as described below.

In addition, the complete set of raw data have been uploaded as an open-access [Zenodo dataset](https://doi.org/###/zenodo.###). These datasets can be collected automatically using the ```raw_data.py``` script in each of the respective analysis folders.

## Workflow

Raw images were processed using a python-based adaptation of ComDet to identify bright intensity spots in images with a heterogeneous background. Thresholds were optimised for each experiment by comparing the number of particles identified by ComDet in positive and negative control images.

For colocalization experiments, ComDet identification was performed independently on both detection channels. Colabelled spots were identified as those with centroids within a 4-pixel radius according to the Euclidean distance in opposing channels. Finally, the proportion of colocalised spots was calculated for a given channel as the number of spots matched to a corresponding spot divided by the total number of spots detected in that channel.

Super-resolution images were reconstructed using the Picasso package. Localisations were identified and fitted, then corrected for microscope drift, precision and random localisations. Themlocalisations was then subjected to a series of morphological dilation, closing and erosion operations as provided by the scikit-image package to yield single connected regions of interest corresponding to individual aggregates. Each aggregate was then measured for basic region properties and for skeletonised length.

Example preprocessed data (here termed raw data) are provided here within the ```data``` folder to test the included analysis scripts.

Individual analyses are presented within the ```src``` folder. Where processing order is important for individual analyses, scripts have been numbered and should be run in order before unnumbered counterparts.

<!-- 1. [Analysis-name](link/to/folder)

| Script      | Language/Interpreter | Description   |
|-------------|----------------------|---------------|
| script_name | Jython/ImageJ        | Functionality |
| script_name | Python               | Functionality |
 -->
