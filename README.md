[![DOI](https://zenodo.org/badge/612212565.svg)](https://zenodo.org/badge/latestdoi/612212565)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8020036.svg)](https://doi.org/10.5281/zenodo.8020036)


# BOEKEN Tau SiMPull 2023

This repository contains the analysis code associated with the Tau SiMPull project, led by Dorothea Böken. This manuscript has been submitted for publication under the title **"Characterisation of tau aggregates in human samples at super-resolution"**.

This manuscript has been submitted as a preprint via BioRxiv [here](https://doi.org/10.1101/2023.06.12.544575). A link to the final version will be provided upon publication.

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> 3.10.5). For specific package requirements, see the environment.yml file, or create a new conda environment containing all packages by running ```conda create -f environment.yml```. 

## Raw data

Example images for use with the initial [ComDet][1]<sup>[1]</sup> (diffraction-limited) and [Picasso][2]<sup>[2]</sup> (super-resolution) steps have been provided, alongside the complete set of preprocessed data as an open-access [Zenodo dataset](https://doi.org/###/zenodo.###). These datasets can be automatically collected using the ```raw_data.py``` script in the ```preprocessing``` folder, and will be placed in a new directory titled 'data'.

## Workflow

Raw images were originally preprocessed using utility scripts provided by [smma](https://github.com/dezeraecox/smma), which allow for manual quality control steps coupled to automatic processing using either a python implementation of [ComDet][1]<sup>[1]</sup> for diffraction-limited images, or [Picasso][2]<sup>[2]</sup> and [SKAN][3]<sup>[3]</sup> for super-resolved images. Thresholds were optimised for each experiment by manually inspecting the output for various threshold combinations on positive and negative control images before applying the automatic analysis to the entire image dataset for that experiment.

The output of these preprocessing steps are then directly analysed using scripts available in the ```analysis``` folder. Here, each figure has a dedicated (independent) analysis script, which performs various filtering, calculation and statistical operations. The results are then saved to ```.csv``` where relevant before being visualised using scripts provided in the ```plotting``` folder. Again, each figure has a dedicated (independent) script. Thus, the scripts can be run in any order with the exception of requiring the analysis script for a given figure (and supplementary figure) to be run before plotting.

## Acknowledgements

This work relies heavily on the excellent existing functionalities provided by the [ComDet][1]<sup>[1]</sup>, [Picasso][2]<sup>[2]</sup> and [SKAN][3]<sup>[3]</sup> packages. 

## References

[1]: https://github.com/UU-cellbiology/ComDet
1. E. Katrukha, ekatrukha/ComDet: ComDet 0.5.3 (2020), doi:10.5281/ZENODO.4281064. 

[2]: https://github.com/square/picasso
2. J. Schnitzbauer, M. T. Strauss, T. Schlichthaerle, F. Schueder, R. Jungmann, Super-resolution microscopy with DNA-PAINT. Nature Protocols 2017 12:6. 12, 1198–1228 (2017).

[3]: https://github.com/jni/skan
3. J. Nunez-Iglesias, A. J. Blanch, O. Looker, M. W. Dixon, L. Tilley, A new Python library to analyse skeleton images confirms malaria parasite remodelling of the red blood cell membrane skeleton. PeerJ. 2018, e4312 (2018).