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
from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms
from microfilm.microplot import microshow
from skimage.io import imread



logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/serum_DL_data/'
output_folder = f'{root_path}results/spot_detection/plot_summary/'

# input_AT8 = '/Users/dorotheaboeken/Documents/GitHub/230112_serum_AT8/results/spot_detection/count_spots/spots_per_fov.csv'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'


sample_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'BSA',
               '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'AD', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD'}


def read_in(input_path, detect):
    spots = pd.read_csv(f'{input_path}')
    spots.drop([col for col in spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)
    # Drop extra wells collected by default due to grid pattern
    spots.dropna(subset=['sample'], inplace=True)

    spots[['capture', 'sample', 'detect']
          ] = spots['sample'].str.split('_', expand=True)
    filtered_spots = spots[spots['detect'] == detect].copy()

    filtered_spots['disease_state'] = filtered_spots['sample'].astype(
        str).map(sample_dict)
    
    filtered_spots = filtered_spots[filtered_spots.disease_state != 'CRL']

    filtered_spots = filtered_spots[filtered_spots.disease_state != 'discard']

    BSA = filtered_spots[filtered_spots['sample']== 'BSA'].copy().reset_index()

    filtered_spots = filtered_spots[filtered_spots.disease_state != 'BSA']

    mean_spots = filtered_spots.groupby(['disease_state', 'channel', 'sample', 'capture', 'detect']).mean().reset_index()

    BSA = BSA.groupby(
        ['disease_state', 'channel', 'sample', 'capture', 'detect', 'slide_position']).mean().reset_index()

    DL_mean_spots = pd.concat([BSA, mean_spots])

    return DL_mean_spots


palette = {
    'CRL': '#345995',
    'AD': '#FB4D3D',
    'BSA': 'lightgrey',
}


palette_repl = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#FB4D3D',
    '55': '#FB4D3D',
    '246': '#FB4D3D',
    'BSA': 'lightgrey',
    'AD Mix': 'darkgrey',

}


def scatbarplot(ycol, ylabel, palette, ax, data):
    order = ['AD', 'BSA']
    sns.barplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=False,
        order=order,
    )
    sns.stripplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=15,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'BSA')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)




DL_spots_AT8 = read_in(f'{input_path}AT8_spots_per_fov.csv', 'AT8')
DL_spots_HT7 = read_in(f'{input_path}HT7_spots_per_fov.csv', 'HT7')
DL_spots_summary = pd.concat([DL_spots_AT8, DL_spots_HT7]).reset_index()

DL_spots_summary.to_csv(f'{output_folder}DL_spots_count_summary.csv')


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.1)


scatbarplot('spots_count', 'Mean number spots',
            palette, axes[0], data=DL_spots_summary[DL_spots_summary['detect'] == 'AT8'])

scatbarplot('spots_count', 'Mean number spots',
            palette, axes[1], data=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'])




# # Read in summary FOV data
# spots = pd.read_csv(f'{input_path}')
# spots.drop([col for col in spots.columns.tolist()
#             if 'Unnamed: ' in col], axis=1, inplace=True)
# # Drop extra wells collected by default due to grid pattern
# spots.dropna(subset=['sample'], inplace=True)

# # expand sample info
# spots[['capture', 'sample', 'detect']
#       ] = spots['sample'].str.split('_', expand=True)
# spots['spots_count'] = spots['spots_count'].fillna(0)
# # spots['spots_count'] = spots['spots_count'].replace(0, np.nan)


# mean_spots = spots.groupby(['sample', 'capture', 'slide_position'])[
#     'spots_count'].median().round(0)

# # mean_spots_AT8.to_csv(f'{output_folder}AT8_serum.csv')
# # mean_spots.to_csv(f'{output_folder}HT7_serum.csv')


# # AT8_HT7 = mean_spots_AT8.div(mean_spots)
# # AT8_HT7 = mean_spots_AT8['']

# sample_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'BSA',
#                '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'AD', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD'}
# spots['disease_state'] = spots['sample'].astype(
#     str).map(sample_dict)


# spots = spots.groupby(
#     ['capture', 'sample', 'slide_position', 'detect', 'disease_state']).median().reset_index()


# filtered_spots = spots[spots.disease_state != 'CRL']
# filtered_spots = filtered_spots[filtered_spots.disease_state != 'discard']

# spots.to_csv(f'{output_folder}filtered_HT7_serum.csv')

# for (capture), df in filtered_spots.groupby(['capture']):
#     sns.barplot(data=df, x='disease_state', y='spots_count',)
#     # plt.xlim(0, 6000)
#     # plt.ylim(0, 6000)

#     plt.title(f'{capture}')
#     plt.show()


# sns.set_theme(style="ticks", font_scale=1.4)
# for (capture), df in filtered_spots.groupby(['capture']):
#     sns.stripplot(
#         data=df.groupby(['capture', 'sample', 'detect', 'disease_state',
#                          ]).mean().reset_index(),
#         x='disease_state',
#         y='spots_count',
#         hue='sample',
#         #ax=axes[x],
#         #palette=palette4,
#         dodge=False,
#         #    s=12,
#         #    alpha=0.4
#     )
#     plt.show()
