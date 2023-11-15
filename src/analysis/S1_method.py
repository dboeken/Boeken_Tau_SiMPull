"""
Preprocesseing for Figure S1
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from loguru import logger
from microfilm.microplot import microshow
from scipy.optimize import leastsq
from skimage.io import imread
from statannotations.Annotator import Annotator

from src.utils import scatbar

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/homogenate_DL_data/'
output_folder = f'{root_path}results/S1_method/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
font = {'family': 'arial',
        'weight': 'normal',
        'size': 7}
cm = 1/2.54


matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300


def read_in(input_path):
    spots = pd.read_csv(f'{input_path}')
    spots.drop([col for col in spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)
    # Drop extra wells collected by default due to grid pattern
    spots.dropna(subset=['sample'], inplace=True)

    spots['fov'] = spots['well_info'].str.split('_').str[0]
    spots['slide_position'] = spots['fov'].str[0:4]

    spots[['capture', 'sample', 'detect']
          ] = spots['sample'].str.split('_', expand=True)

    mean_spots = spots.groupby(['layout', 'channel', 'sample',
                                         'capture', 'detect', 'slide_position']).mean().reset_index()

    return mean_spots


# =========Organise data========
mean_spots_Xreactivity = read_in('data/homogenate_DL_data/Xreactivity_spots_per_fov.csv')
mean_spots_dilution = read_in(
    'data/homogenate_DL_data/dilution_spots_per_fov.csv')

spots_hom_DL = pd.read_csv('results/2_homogenate_DL/spots_count_summary.csv')
spots_hom_DL = spots_hom_DL[spots_hom_DL['sample']!='BSA']
tau_elisa = pd.read_csv('results/S1_method/total_tau_ELISA.csv', sep=';')
mean_tau_elisa = tau_elisa.groupby(['sample']).mean().reset_index()
elisa_dict=dict(zip(mean_tau_elisa['sample'], mean_tau_elisa['concentration']))

elisa_dict= {'9': 11934239.666666666,
 '13': 6791242.333333333,
 '28': 14115627.333333334,
 '55': 11960455.333333334,
 '159': 23236355.0,
 '246': 6395861.666666667}

spots_hom_DL['total_tau']= spots_hom_DL['sample'].astype(str).map(elisa_dict)
spots_hom_DL['norm_tau'] = (spots_hom_DL['spots_count']/ spots_hom_DL['total_tau'])*1000000

mean_spots_dilution = pd.read_csv(
    'data/homogenate_DL_data/dilution_spots_per_fov.csv')
mean_spots_dilution.drop([col for col in mean_spots_dilution.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# fill slides with no spots
mean_spots_dilution['spots_count'] = mean_spots_dilution['spots_count'].fillna(
    0)



mean_spots_dilution[['capture', 'sample', 'detect']
      ] = mean_spots_dilution['sample'].str.split('_', expand=True)

mean_spots_dilution = mean_spots_dilution[mean_spots_dilution['detect'] != 'IgG'].copy(
)


mean_spots_dilution = mean_spots_dilution.groupby(
    ['slide_position', 'capture', 'detect', 'sample', 'layout']).mean().reset_index()

mean_spots_dilution = mean_spots_dilution[
    (mean_spots_dilution['slide_position'] != 'X5Y0') & (
        mean_spots_dilution['layout'] != '1')
].copy()

dilution_dict = {
    'D10': 680000,
    'D100': 68000,
    'D1000': 6800,
    'D10000': 680,
    'BSA': 0
}

mean_spots_dilution['concentration'] = mean_spots_dilution['sample'].map(
    dilution_dict)


df_BSA = mean_spots_dilution[mean_spots_dilution['sample'] == 'BSA'].copy(
)

data = df_BSA.groupby(['capture',
                       'detect', 'layout']).mean().reset_index()

data['key'] = data['capture'] + '_' + data['detect']

dict_BSA = dict(data[['key', 'spots_count']].values)

mean_spots_dilution['key'] = mean_spots_dilution['capture'] + \
    '_' + mean_spots_dilution['detect']

mean_spots_dilution['control'] = mean_spots_dilution['key'].map(
    dict_BSA)

mean_spots_dilution['StoN'] = mean_spots_dilution['spots_count'] / \
    mean_spots_dilution['control']


example_BSA = imread('data/homogenate_DL_data/example_BSA.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)


mean_spots_Xreactivity = mean_spots_Xreactivity[
    (mean_spots_Xreactivity['capture'].isin(['HT7', 'AT8', 'SC211', '6E10'])) &
    (mean_spots_Xreactivity['sample'].isin(['R1E5', 'asyn', 'abeta', 'PBS'])) &
    (mean_spots_Xreactivity['detect'].isin(['HT7', 'AT8', 'SC211', '6E10']))
]

peptide_data = pd.read_csv(f'{root_path}data/peptide_data/spots_per_fov.csv')
peptide_data.drop([col for col in peptide_data.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# Drop extra wells collected by default due to grid pattern
peptide_data.dropna(subset=['sample'], inplace=True)

# expand sample info
peptide_data[['capture', 'sample', 'detect']
      ] = peptide_data['sample'].str.split('_', expand=True)
peptide_data['peptide_data_count'] = peptide_data['spots_count'].fillna(0)

mean_number_peptide_data = peptide_data.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_number_peptide_data[['sample', 'concentration']
                         ] = mean_number_peptide_data['sample'].str.split('-', expand=True)


def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D


def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A, B, C, D = p
    err = y-logistic4(x, A, B, C, D)
    return err


def peval(x, p):
    """Evaluated value at x with current parameters."""
    A, B, C, D = p
    return logistic4(x, A, B, C, D)


palette = {
    'AT8': 'lightgrey',
    'HT7': 'darkgrey',

}


def LOD_cal(data, capture, detect):
    df_D10000 = data[(data['capture'] == capture) & (
        data['detect'] == detect) & (data['sample'] == 'D1000')]

    df_BSA = data[(data['capture'] == capture) & (
        data['detect'] == detect) & (data['sample'] == 'BSA')]

    D10000 = df_D10000['spots_count'].values
    BSA = df_BSA['spots_count'].values

    LOB = np.mean(BSA) + 1.645 * np.std(BSA)

    LOD = LOB + 1.645 * np.std(D10000)

    return LOD, LOB


def LOD_calculation(data, capture, detect, ax):
    df = data[(data['capture'] == capture) & (data['detect'] ==
                                              detect)].copy().dropna(subset=['concentration'])

    x = df['concentration'].values

    y_meas = df['spots_count'].values

    # Initial guess for parameters
    p0 = [0, 0.004, 10, 1000]

    # Fit equation using least squares optimization
    plsq = leastsq(residuals, p0, args=(y_meas, x))

    # Plot results
    ax.plot(
        df.groupby('concentration').mean().reset_index()[
            'concentration'].tolist(),
        df.groupby('concentration').mean().reset_index()[
            'spots_count'].tolist(),
        marker='o', color=palette[detect],
        linewidth=0, markersize=4
    )
    ax.errorbar(
        df.groupby('concentration').mean().reset_index()[
            'concentration'].tolist(),
        df.groupby('concentration').mean().reset_index()[
            'spots_count'].tolist(),
        df.groupby('concentration').std().reset_index()[
            'spots_count'].tolist(),
        df.groupby('concentration').std().reset_index()[
            'spots_count'].tolist(),
        linestyle='', color=palette[detect], capsize=4,
    )

    sns.lineplot(
        x=np.arange(np.min(x), np.max(x), 10), y=peval(np.arange(np.min(x), np.max(x), 10), plsq[0]), color=palette[detect], label=detect, ax=ax
        ),

    a, b, c, d = plsq[0]

    LOD_spots, LOB_spots = LOD_cal(df, capture, detect)

    LOD_conc = c * np.power(((LOD_spots-a)/(d-LOD_spots)), 1/b)
    LOB_conc = c * np.power(((LOB_spots-a)/(d-LOB_spots)), 1/b)

    return LOD_conc, LOB_conc


#LOD val
# LOD_values = []
# for (detect, capture), df in mean_spots_dilution.groupby(['detect', 'capture']):
#     if detect == capture:
#         LOD_conc, LOB_conc = LOD_calculation(
#             mean_spots_dilution, detect, capture)
#         LOD_values.append([detect, capture, LOD_conc, LOB_conc])
# LOD_values = pd.DataFrame(
#     LOD_values, columns=['Detect', 'Capture', 'LOD', 'LOB'])

# plt.legend()
# plt.xlabel('Concentration of tau [pg/mL]')
# plt.ylabel('Number of aggregates per FOV')
# LOD_values


spots_hom_DL.to_csv(
    f'{output_folder}spots_hom_DL.csv')
mean_number_peptide_data.to_csv(
    f'{output_folder}mean_number_peptide_data.csv')
mean_spots_Xreactivity.to_csv(
    f'{output_folder}mean_spots_Xreactivity.csv')
mean_spots_dilution.to_csv(
    f'{output_folder}mean_spots_dilution.csv')

