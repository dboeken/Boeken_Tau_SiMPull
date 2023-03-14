from logging import captureWarnings
from random import sample
from statistics import mean
import matplotlib
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

logger.info('Import OK')

if os.path.exists('data_path.txt'):
    root_path = open("experimental_data/root_path.txt", "r").readlines()[0]
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
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

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

########plot HT7 peptide data #########

sns.set(rc={'figure.figsize': (4, 4)})
sns.set_theme(style="ticks", font_scale=1.4)
for (capture), df in mean_number_spots[mean_number_spots['concentration'].isin(['02', '', 'low'])].groupby(['capture']):
    sns.stripplot(data=df, x='sample', y='spots_count', color='black')
    # plt.xlim(0, 6000)

    sns.barplot(
        data=df,
        x='sample',
        y='spots_count',
        #hue='sample',
        #dodge=False,
        #s=12,
        #alpha=0.4
        capsize=0.2,
        errwidth=2,
    )

    plt.ylim(0, 400)
    #plt.ylabel("mean spots")

    plt.title(f'{capture}')
    #plt.show()
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
