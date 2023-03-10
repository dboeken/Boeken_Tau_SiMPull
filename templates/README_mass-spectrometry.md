[![DOI](https://zenodo.org/badge/..../.svg)](https://doi.org/###/zenodo.###)

# AUTHOR_Running-title

This repository contains the analysis code associated with the **###** project, led by **###**. This manuscript has been submitted for publication under the title **"#####"**.

This manuscript has been submitted as a preprint via BioRxiv [here](biorxiv/link). A link to the final version will be provided upon publication.

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> **version**). For specific package requirements, see the environment.yml file, or  create a new conda environment containing all packages by running ```conda create -f environment.yml```. In addition to the analysis contained here, some simple statistical tests were performed using [GraphPad Prism v **8.0**](https://www.graphpad.com/scientific-software/prism/).

## Raw data

The .RAW files have been deposited via the [PRIDE][1]<sup>[1]</sup> partner repository to the [ProteomeXchange Consortium][2]<sup>[2]</sup> under the dataset identifier PXD######. For convenience, the preprocessed identification and quantitation data (hereon termed raw data) have also been uploaded as an open-access [Zenodo dataset](https://doi.org/###/zenodo.###). These data can be collected automatically using the ```raw_data.py``` script in each of the respective ```src``` analysis folders.

Various public databases were also queried as cited in the accompanying manuscript, for which access protocols are provided in the respective ```utilities``` scripts where appropriate.

## Workflow

Initial processing of the novel mass spectrometry spectra files was completed using either [Proteome Discoverer] or [MaxQuant][3]<sup>[3]</sup>.

Individual analyses are presented within the ```src``` folder. Where processing order is important for individual analyses, scripts have been numbered and should be run in order before unnumbered counterparts.

1. [Analysis-name](link/to/folder)

| Script      | Language/Interpreter | Description   |
|-------------|----------------------|---------------|
| script_name | Python               | Functionality |
| script_name | Python               | Functionality |

## References

[1]: https://www.ebi.ac.uk/pride/archive/

1. Perez-Riverol, Yasset, Attila Csordas, Jingwen Bai, Manuel Bernal-Llinares, Suresh Hewapathirana, Deepti J Kundu, Avinash Inuganti, et al. “The PRIDE Database and Related Tools and Resources in 2019: Improving Support for Quantification Data.” Nucleic Acids Research 47, no. D1 (January 8, 2019): D442–50. https://doi.org/10.1093/nar/gky1106.

[2]: http://proteomecentral.proteomexchange.org

2. Deutsch, Eric W., Attila Csordas, Zhi Sun, Andrew Jarnuczak, Yasset Perez-Riverol, Tobias Ternent, David S. Campbell, et al. “The ProteomeXchange Consortium in 2017: Supporting the Cultural Change in Proteomics Public Data Deposition.” Nucleic Acids Research 45, no. Database issue (January 4, 2017): D1100–1106. https://doi.org/10.1093/nar/gkw936.

[Proteome Discoverer]: https://www.thermofisher.com/au/en/home/industrial/mass-spectrometry/liquid-chromatography-mass-spectrometry-lc-ms/lc-ms-software/multi-omics-data-analysis/proteome-discoverer-software.html

[3]: https://www.maxquant.org/

3. Tyanova, Stefka, Tikira Temu, and Juergen Cox. “The MaxQuant Computational Platform for Mass Spectrometry-Based Shotgun Proteomics.” Nature Protocols 11, no. 12 (December 2016): 2301–19. https://doi.org/10.1038/nprot.2016.136.
