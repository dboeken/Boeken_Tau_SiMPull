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
        'size': 16}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

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
example_monomer = imread(f'{root_path}data/peptide_images/X4Y1R3W2_641.tif')
example_dimer = imread(f'{root_path}data/peptide_images/X1Y1R2W2_641.tif')

# ==================Generate main figure==================

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Panel D: Example BSA image
example_BSA_mapped, _, _ = colorify.colorify_by_hex(mean_img_A, cmap_hex='#FF00FF', rescale_type='limits',
                                       limits=[0,5000])
axes[0].imshow()

# Panel E: Spot count
df = mean_number_spots[(mean_number_spots['concentration'].isin(['02', '', 'low'])) & (mean_number_spots['capture'] == 'HT7')]
sns.stripplot(data=df, x='sample', y='spots_count', color='darkgrey', s=10)
sns.barplot(
    data=df,
    x='sample',
    y='spots_count',
    capsize=0.2,
    errwidth=2,
)
axes[3].ylim(0, 400)
axes[3].ylabel("Mean number of spots per FOV")

plt.savefig(f'{output_folder}mean_number_spots_HT7.svg')


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
