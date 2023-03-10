# Public repository template for scientific manuscripts

This repository contains a collection of templates designed to ease the creation of public-facing code and data repositories for scientific manuscripts. The templates provided are geared toward python-directed computational analyses of biological datasets, but could be adapted to suit other purposes at the users discretion.

## Getting started
---

### 1. Clone the template repository

To use this repository, first clone the repository using the ```Use this template``` button. It is recommended to title the repository using the *AUTHOR_Running-title syntax* for ease of use and access. Once your repository is created, make sure to clone a local copy for editing. 

### 2. Add components

Add your codebase under the ```src``` folder and example data to the ```data``` folder. An optional ```utilities``` folder is also provided to house scripts that are accessed via relative imports in the ```src``` files. It is recommended to mirror the structure of the ```data``` and ```src``` folders, such that ```src/analysis-one/``` contains processing scripts for ```data/analysis-one/```. Optional README files are provided in each folder allowing you to provide more detailed information if necessary. Alternatively, these files can be removed.

Add your environment file (an [environment.yml](environment.yml) example is provided), including relevant version constraints. For example, an environment file containing only the components explicitly installed in a conda environment can be generated using the command ```conda env export --from-history```.

Check (or replace) the [license](LICENSE) file, and ensure that it provides appropriate permissions for anyone wishing to repurpose your codebase/dataset. 

Finally, check (or replace) the [.gitignore](.gitignore) file. A standard python version is provided.

### 3. Edit the README

Templates of interest (currently provided are *general*, *image-analysis* and *mass-spectrometry* examples) can be found in the [templates](templates/) folder. Copy the template README you select into the root folder, and remove the remaining templates. Edit the new README template file to reflect your manuscript details, and include a summary of the workflow(s) to be contained within the repository. Once you have completed the repository, you can rename the template file to README.md, which will then replace this file as the front matter for the repository. 

Some commonly-used references are provided in each template, as well as an example table for providing detailed workflow descriptions. Alternatively, users familiar with python may wish to try a specific workflow management method, such as [SnakeMake](https://snakemake.readthedocs.io/en/stable/) or [YAWL](https://yawlfoundation.github.io/).


### 4. Edit the citation file

A template [citation file](citation.cff) is provided, which contains optional metadata fields. This file will be rendered by GitHub alongside your repository giving visitors an option to cite the repository automatically via an APA or BibTex formatted citation. You may choose to use this file for a citation of the manuscript, or for the repository itself (*NB:* if choosing this option, you should include a DOI generated using step 6 below). Alternatively, remove this file to prevent the citation functionality from being rendered.


### 5. *RECOMMENDED*: Zenodo repository for code archive and DOI creation

In addition to providing publically-accessible code via this GitHub repository, services such as [FigShare](https://figshare.com/) and [Zenodo](https://zenodo.org/) can connect with GitHub to generate archived and versioned copies of individual repositories. Using this integration, it is then possible to generate a DOI which can be included in your manuscript to point to the accompanying codebase here.

It is recommended to do this once editing your cloned version of this repository is complete, and space is provided at the top of the README in each template to house the associated repository badges.


### 6. *OPTIONAL*: External repositories for storing raw data collections

Many disciplines maintain technique-specific repositories for all raw data associated with published manuscripts, for example the PRIDE repository housing raw mass spectrometry data. It is recommended to use these repositories in the first instance, and usage instructions given by the provider should be followed. In most cases, these databases will generate a unique identifier or DOI which can then be linked in the accompanying repository here. For example, space is provided for PRIDE identifiers for mass spectrometry data etc.

In the case of no specific repositories being available, general-use examples include [FigShare](https://figshare.com/) and [Zenodo](https://zenodo.org/). Both accept raw dataset submissions, generate a DOI and provide long-term public managed access to your dataset.

Once any external data respository DOI's have been generated, add the associated repository badges to the top of the README.


## Disclaimer
---

*This template repository was designed for personal use and is provided as-is. Whilst I endeavour to keep it up-to-date and respond to issues raised here, I can provide no guarantee of the completeness, accuracy, reliability, suitability or availability of the information, services and software contained here for your use case.*