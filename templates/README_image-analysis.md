[![DOI](https://zenodo.org/badge/..../.svg)](https://doi.org/###/zenodo.###)

# AUTHOR_Running-title

This repository contains the analysis code associated with the **###** project, led by **###**. This manuscript has been submitted for publication under the title **"#####"**.

This manuscript has been submitted as a preprint via BioRxiv [here](biorxiv/link). A link to the final version will be provided upon publication.

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> **version**). For specific package requirements, see the environment.yml file, or  create a new conda environment containing all packages by running ```conda create -f environment.yml```. In addition to the analysis contained here, some simple statistical tests were performed using [GraphPad Prism v **8.0**](https://www.graphpad.com/scientific-software/prism/).

## Raw data

For convenience, example datasets are provided here under the ```data``` folder. These data may be used to explore the workflows presented here as described below.

In addition, the complete set of raw data have been uploaded as an open-access [Zenodo dataset](https://doi.org/###/zenodo.###). These datasets can be collected automatically using the ```raw_data.py``` script in each of the respective analysis folders.

## Workflow

Example raw images and partially processed results are provided here within the ```data``` folder to test the included analysis scripts.

Initial preprocessing of the raw microscopy files to extract TIFF images from proprietary formats was completed in [Fiji][1]<sup>[1]</sup> or [ImageJ][2] <sup>[2]</sup> using the [BioFormats importer][3] <sup>[3]</sup> (available via the drag-and-drop interface). Some additional utility scripts are included covering this in the ```src``` folder. Stacked or individual TIFF files were then exported for further processing where necessary, examples of which are provided within the ```data``` folder.

Individual analyses are presented within the ```src``` folder. Where processing order is important for individual analyses, scripts have been numbered and should be run in order before unnumbered counterparts.

1. [Analysis-name](link/to/folder)

| Script      | Language/Interpreter | Description   |
|-------------|----------------------|---------------|
| script_name | Jython/ImageJ        | Functionality |
| script_name | Python               | Functionality |

## References

[1]: https://imagej.net/ImageJ2

1. Schindelin, Johannes, Ignacio Arganda-Carreras, Erwin Frise, Verena Kaynig, Mark Longair, Tobias Pietzsch, Stephan Preibisch, et al. “Fiji: An Open-Source Platform for Biological-Image Analysis.” Nature Methods 9, no. 7 (July 2012): 676–82. https://doi.org/10.1038/nmeth.2019.

[2]: https://imagej.net/Fiji

2. Rueden, Curtis T., Johannes Schindelin, Mark C. Hiner, Barry E. DeZonia, Alison E. Walter, Ellen T. Arena, and Kevin W. Eliceiri. “ImageJ2: ImageJ for the next Generation of Scientific Image Data.” BMC Bioinformatics 18, no. 1 (November 29, 2017): 529. https://doi.org/10.1186/s12859-017-1934-z.

[3]: https://docs.openmicroscopy.org/bio-formats/5.8.2/users/imagej/installing.html

3. Linkert, Melissa, Curtis T. Rueden, Chris Allan, Jean-Marie Burel, Will Moore, Andrew Patterson, Brian Loranger, et al. “Metadata Matters: Access to Image Data in the Real World.” Journal of Cell Biology 189, no. 5 (May 31, 2010): 777–82. https://doi.org/10.1083/jcb.201004104.
