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
#from smma.src import visualise
from loguru import logger

logger.info('Import OK')

input_path = 'results/spot_detection/count_spots/spots_per_fov.csv'
output_folder = 'results/spot_detection/plot_summary/'

input_AT8 = '/Users/dorotheaboeken/Documents/GitHub/230112_serum_AT8/results/spot_detection/count_spots/spots_per_fov.csv'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
# spots['spots_count'] = spots['spots_count'].replace(0, np.nan)


mean_spots = spots.groupby(['sample', 'capture', 'slide_position'])[
    'spots_count'].median().round(0)

# mean_spots_AT8.to_csv(f'{output_folder}AT8_serum.csv')
# mean_spots.to_csv(f'{output_folder}HT7_serum.csv')


# AT8_HT7 = mean_spots_AT8.div(mean_spots)
# AT8_HT7 = mean_spots_AT8['']

sample_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'BSA',
               '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'AD', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD'}
spots['disease_state'] = spots['sample'].astype(
    str).map(sample_dict)


spots = spots.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state']).median().reset_index()


filtered_spots = spots[spots.disease_state != 'CRL']
filtered_spots = filtered_spots[filtered_spots.disease_state != 'discard']

spots.to_csv(f'{output_folder}filtered_HT7_serum.csv')

for (capture), df in filtered_spots.groupby(['capture']):
    sns.barplot(data=df, x='disease_state', y='spots_count',)
    # plt.xlim(0, 6000)
    # plt.ylim(0, 6000)

    plt.title(f'{capture}')
    plt.show()


sns.set_theme(style="ticks", font_scale=1.4)
for (capture), df in filtered_spots.groupby(['capture']):
    sns.stripplot(
        data=df.groupby(['capture', 'sample', 'detect', 'disease_state',
                         ]).mean().reset_index(),
        x='disease_state',
        y='spots_count',
        hue='sample',
        #ax=axes[x],
        #palette=palette4,
        dodge=False,
        #    s=12,
        #    alpha=0.4
    )
    plt.show()
