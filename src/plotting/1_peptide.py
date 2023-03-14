import os
import re
from logging import captureWarnings
from random import sample
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from microfilm.microplot import microshow
from microfilm import colorify


from skimage.io import imread

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open("data/data_path.txt", "r").readlines()[0]
else:
    root_path = ''

# Set paths
input_path = f'{root_path}data/peptide_data/spots_per_fov.csv'
output_folder = 'results/1_peptide/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set default font parameters
font = {'family': 'normal',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

palette = {
    '':'',
    '':'',
    '':'',
}

# =====================Organise data=====================
# Read in summary FOV data
spots = pd.read_csv(f'{input_path}')
spots.drop([col for col in spots.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# Drop extra wells collected by default due to grid pattern
spots.dropna(subset=['sample'], inplace=True)

# expand sample info
spots[['capture', 'sample', 'detect']
      ] = spots['sample'].str.split('_', expand=True)
spots['spots_count'] = spots['spots_count'].fillna(0)

mean_number_spots = spots.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_number_spots[['sample', 'concentration']
                  ] = mean_number_spots['sample'].str.split('-', expand=True)

# =================Read in example images=================

example_BSA = imread(f'{root_path}data/peptide_images/X6Y1R3W3_641.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)
example_monomer = imread(f'{root_path}data/peptide_images/X4Y1R3W2_641.tif')
example_monomer = np.mean(example_monomer[10:, :, :], axis=0)
example_dimer = imread(f'{root_path}data/peptide_images/X1Y1R2W2_641.tif')
example_dimer = np.mean(example_dimer[10:, :, :], axis=0)


# ==================Generate main figure==================

fig, axes = plt.subplots(1, 4, figsize=(23, 5), gridspec_kw={'width_ratios': [6, 6, 6, 5]})

# Panel D: Example BSA image
microim1 = microshow(images=[example_BSA],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axes[0],
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)
microim1 = microshow(images=[example_monomer],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axes[2],
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)
microim1 = microshow(images=[example_dimer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axes[1],
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)

# Panel E: Spot count
df = mean_number_spots[(mean_number_spots['concentration'].isin(['02', '', 'low'])) & (mean_number_spots['capture'] == 'HT7')]
sns.stripplot(
    data=df, 
    x='sample', 
    y='spots_count', 
    color='#36454F',
    s=10
)
sns.barplot(
    data=df,
    x='sample',
    y='spots_count',
    capsize=0.2,
    errwidth=2,
    color='darkgrey'
)
axes[3].set_ylim(0, 400)
axes[3].set_ylabel("Mean spots per FOV")
axes[3].set_xlabel("")

axes[0].annotate('B', xy=(0, 1.05), xycoords='axes fraction', size=24, weight='bold')
axes[1].annotate('C', xy=(0, 1.05), xycoords='axes fraction', size=24, weight='bold')
axes[2].annotate('D', xy=(0, 1.05), xycoords='axes fraction', size=24, weight='bold')
axes[3].annotate('E', xy=(-0.3, 1.05), xycoords='axes fraction', size=24, weight='bold')

plt.tight_layout()
plt.savefig(f'{output_folder}1_peptide.svg')
plt.show()


########plot AT8 peptide data for supplementals #########

sns.set(rc={'figure.figsize': (3, 4)})
sns.set_theme(style="ticks", font_scale=1.4)
for (capture), df in mean_number_spots[mean_number_spots['capture'].isin(['AT8'])].groupby(['capture']):
    sns.stripplot(data=df, x='sample', y='spots_count',)
    # plt.xlim(0, 6000)

    sns.stripplot(
        data=df.groupby(['capture', 'sample', 'detect',
                         ]).mean().reset_index(),
        x='sample',
        y='spots_count',
        #hue='sample',
        dodge=False,
        s=12,
        alpha=0.4
    )

    plt.ylim(0, 500)
    #plt.ylabel("mean spots")

    plt.title(f'{capture}')
    #plt.show()
    plt.savefig(f'{output_folder}mean_number_spots_AT8.svg')
