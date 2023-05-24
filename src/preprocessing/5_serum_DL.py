"""
Preprocessing of the raw data for plotting Figure 5
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path_DL = f'{root_path}data/serum_DL_data/'
input_path_SR = f'{root_path}data/serum_SR_data/'
output_folder = f'results/5_serum/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    

cm = 1 / 2.54
font = {'family': 'arial',

        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi']= 300


sample_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'BSA',
               '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'AD', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD'}

sex_dict = {'1': 'O', '2': 'F', 'BSA': 'BSA',
            '3': 'F', '4': 'F', '5': 'F', '6': 'F', '7': 'M', '8': 'F', '9': 'F', '10': 'O', '11': 'M', '12': 'F', '13': 'F', '15': 'M', '16': 'O', '17': 'O', '18': 'M', '19': 'O', '20': 'F', 'IgG': 'BSA'}


def read_in(input_path, detect,):
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
    
    filtered_spots['sex'] = filtered_spots['sample'].astype(
        str).map(sex_dict)
    
    # filtered_spots = filtered_spots[filtered_spots.disease_state != 'CRL']

    filtered_spots = filtered_spots[filtered_spots.disease_state != 'discard']

    BSA = filtered_spots[filtered_spots['sample']== 'BSA'].copy().reset_index()

    filtered_spots = filtered_spots[filtered_spots.disease_state != 'BSA']

    mean_spots = filtered_spots.groupby(['disease_state', 'channel', 'sample', 'capture', 'detect', 'sex']).mean().reset_index()

    BSA = BSA.groupby(
        ['disease_state', 'channel', 'sample', 'capture', 'detect', 'slide_position', 'sex']).mean().reset_index()

    DL_mean_spots = pd.concat([BSA, mean_spots])

    return DL_mean_spots


palette = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
    'F': '#FB4D3D',
    'M': '#345995', 
    'O': 'lightgrey'
}

pvalues = [0.443, 0.00063, 0.00013]


def perform_lda(for_LDA, value_cols, category_col='key'):
    # Select data columns (X) and category columns (Y)
    X = for_LDA[value_cols].values
    y = for_LDA[category_col].values

    # fit LDA
    lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
    lda_model = lda.fit(X, y)

    # Add LDA fit back to original data for plotting
    for_LDA[['dim1', 'dim2']] = lda_model.transform(X)

    return for_LDA, lda_model


def calculate_eigens(model, labels):
    eigens = model.scalings_
    dist = pd.DataFrame([eigens[:, 0], eigens[:, 1]], index=['xpos', 'ypos']).T
    dist['label'] = labels
    dist['ori_dst'] = [distance.euclidean(
        (xpos, ypos), (0, 0)) for xpos, ypos in dist[['xpos', 'ypos']].values]

    return dist


# Read in diff limited data
DL_spots_AT8 = read_in(f'{input_path_DL}AT8_spots_per_fov.csv', 'AT8')
DL_spots_HT7 = read_in(f'{input_path_DL}HT7_spots_per_fov.csv', 'HT7')
DL_spots_summary = pd.concat([DL_spots_AT8, DL_spots_HT7]).reset_index()
DL_spots_summary.to_csv(f'{output_folder}DL_spots_count_summary.csv')

# Read in super-res data
SR_spots = pd.read_csv(
    f'{input_path_SR}properties_compiled.csv')
SR_spots.drop([col for col in SR_spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)
SR_spots = SR_spots.dropna(subset=['label']).copy()

# Map tissue types for SR data
tissue_dict = {'2022-07-06_21-19-49': 'brain',
               '2023-03-02_14-31-20': 'serum', '2023-03-02_18-25-28': 'serum'}
SR_spots['tissue'] = SR_spots['folder'].astype(
    str).map(tissue_dict)

# Map sample IDs  for SR data
serum_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'discard',
              '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'discard', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD', 'IgG': 'discard'}


brain_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'discard',
              '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD', 'IgG': 'discard'}
SR_spots['sample'] = [str(int(float(sample))) if sample not in ['BSA', 'IgG'] else sample for sample in SR_spots['sample']]
SR_spots['disease_state'] = [serum_dict[sample] if tissue == 'serum' else brain_dict[sample] for sample, tissue in SR_spots[['sample', 'tissue']].values]

SR_spots['sex'] = [sex_dict[sample] if tissue == 'serum' else np.NaN for sample, tissue in SR_spots[['sample', 'tissue']].values]

# Remove IgG, BSA, outlier sample
SR_spots = SR_spots[SR_spots['disease_state']!= 'discard'].copy()

# Assign categories based on thresholds for length, area, ecc
SR_spots['length_cat'] = ['long' if length >
                          150 else 'short' for length in SR_spots['smoothed_length']]
SR_spots['area_cat'] = ['large' if area >
                        4000 else 'small' for area in SR_spots['area']]
SR_spots['ecc_cat'] = ['round' if ecc <
                       0.9 else 'fibrillar' for ecc in SR_spots['eccentricity']]
SR_spots_mean = SR_spots[
    (SR_spots['smoothed_length'] > 50) & 
    (SR_spots['tissue'] == 'serum')
    ].groupby(['sample', 'detect', 'folder', 'well_info', 'tissue', 'disease_state', 'sex']).mean().reset_index().groupby(['sample', 'tissue', 'disease_state', 'sex']).mean().reset_index()

SR_spots_mean['scaled_area'] = SR_spots_mean['area'] * (107/8)**2 / 1000
######## Compile summary data for LDA

# selecting super res data for dim reduction
summary = SR_spots[
    (SR_spots['disease_state'].isin(['AD', 'CRL'])) &
    (SR_spots['capture'] == 'HT7') &
    (SR_spots['prop_type'] == 'smooth')
].copy()
# map localisations from cluster to smoothed
locs_dict = dict(SR_spots[SR_spots['prop_type'] ==
                 'cluster'][['key', '#locs']].values)
summary['#locs'] = summary['key'].map(locs_dict)
summary = summary.groupby(
    ['capture', 'tissue', 'disease_state', 'sample']).mean().copy().reset_index()

summary['key'] = summary['tissue'] + '_' + \
    summary['disease_state'] + '_' + summary['sample']

# Collect diff limited data for brain
brain_DL = pd.read_csv(
    f'{root_path}results/2_homogenate_DL/spots_count_summary.csv')
brain_DL.drop([col for col in brain_DL.columns.tolist()
              if 'Unnamed: ' in col], axis=1, inplace=True)
brain_DL['key'] = 'brain_' + \
    brain_DL['disease_state'] + '_' + brain_DL['sample']

# Collect diff limited data for serum
serum_DL = pd.read_csv(f'{input_path_DL}HT7_spots_per_fov.csv')
serum_DL.drop([col for col in serum_DL.columns.tolist()
              if 'Unnamed: ' in col], axis=1, inplace=True)
serum_DL[['capture', 'sample', 'detect']
         ] = serum_DL['sample'].str.split('_', expand=True)
serum_DL['disease_state'] = serum_DL['sample'].astype(str).map(sample_dict)
serum_DL = serum_DL[(serum_DL['detect'] == 'HT7') & ~(
    serum_DL['disease_state'].isin(['discard', 'IgG', 'BSA']))].copy()
serum_DL = serum_DL.groupby(
    ['disease_state', 'channel', 'sample', 'capture', 'detect']).mean().reset_index()
serum_DL['key'] = 'serum_' + \
    serum_DL['disease_state'] + '_' + serum_DL['sample']

# merge super res and diff limited datasets
DL_spots = pd.concat([brain_DL, serum_DL])
DL_spots = DL_spots[DL_spots['detect'] == 'HT7'].copy()
summary = pd.merge(
    summary, DL_spots[['key', 'spots_count']], on='key', how='left')
summary['tissue'] = summary['key'].str.split('_').str[0]
summary['category'] = summary['tissue'] + '_' + summary['disease_state']


# Select columns for dim reduction
group_cols = ['tissue', 'disease_state', 'sample']
value_cols = ['area', 'eccentricity', 'perimeter',
              'minor_axis_length', 'major_axis_length', 'smoothed_length', '#locs', 'spots_count']
lda, model = perform_lda(summary, value_cols, category_col='category')


eigens_df = calculate_eigens(model, labels=['area', 'eccentricity', 'perimeter',
                                            'minor_axis_length', 'major_axis_length', 'smoothed_length', '#locs', 'spots_count'])

DL_spots_summary.to_csv(
    f'{output_folder}DL_spots_summary.csv')
SR_spots_mean.to_csv(
    f'{output_folder}SR_spots_mean.csv')
lda.to_csv(
    f'{output_folder}lda_summary.csv')
eigens_df.to_csv(
    f'{output_folder}eigens_df.csv')

