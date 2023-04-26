import os
import re
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.transforms as mtransforms
from loguru import logger
from scipy.spatial import distance

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path_DL = f'{root_path}data/serum_DL_data/'
input_path_SR = f'{root_path}data/serum_SR_data/'
output_folder = f'results/serum/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    

# input_AT8 = '/Users/dorotheaboeken/Documents/GitHub/230112_serum_AT8/results/spot_detection/count_spots/spots_per_fov.csv'

cm = 1 / 2.54
font = {'family': 'arial',

        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'


sample_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'BSA',
               '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'AD', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD'}


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
    
    # filtered_spots = filtered_spots[filtered_spots.disease_state != 'CRL']

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


def plot_scatbar_DL(ycol, ylabel, palette, ax, data):
    order = ['AD', 'CRL', 'BSA']
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
        s=10,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'CRL'), ('CRL', 'BSA'), ('AD', 'BSA')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def plot_scatbar(data, ycol, ylabel, palette, order, pairs, hue_order=None, ax=None, s=15):
    if not ax:
        fig, ax = plt.subplots()
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
        hue_order=hue_order
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
        s=s,
        order=order,
        hue_order=hue_order
    )

    ax.set(ylabel=ylabel, xlabel='')
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside', comparisons_correction='bonferroni')
    annotator.apply_and_annotate()

    ax.legend('', frameon=False)


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


def plot_lda(data, palette, hue='disease_state', style='tissue', ax=None, s=300):
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.scatterplot(
        data=data,
        x='dim1',
        y='dim2',
        hue=hue,
        style=style,
        palette=palette,
        s=s,
        ax=ax
    )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    # plt.legend(loc='upper left', ncol=2, columnspacing=0.1, title='')

    return ax


def plot_eigens(model, X, labels=None, num_labels=5):

    fitted = model.transform(X)
    eigens = model.scalings_
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    n = eigens.shape[0]

    dists = distance_eigens(model, labels=labels)

    fig, ax = plt.subplots()
    for i in range(n):
        plt.arrow(0, 0, eigens[i, 0], eigens[i, 1],
                  color='firebrick', linewidth=2)
    if labels is not None:
        for x, y, label in dists.sort_values('ori_dst', ascending=False).iloc[:num_labels][['xpos', 'ypos', 'label']].values:
            plt.text((x) * 1.15, y * 1.15, label,
                     color='firebrick', ha='center', va='center')

    return fig, ax, dists


def calculate_eigens(model, labels):
    eigens = model.scalings_
    dist = pd.DataFrame([eigens[:, 0], eigens[:, 1]], index=['xpos', 'ypos']).T
    dist['label'] = labels
    dist['ori_dst'] = [distance.euclidean(
        (xpos, ypos), (0, 0)) for xpos, ypos in dist[['xpos', 'ypos']].values]

    return dist


def distance_eigens(model, labels):
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
    ].groupby(['sample', 'detect', 'folder', 'well_info', 'tissue', 'disease_state']).mean().reset_index().groupby(['sample', 'tissue', 'disease_state']).mean().reset_index()

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


######## Plot figure 

fig = plt.figure(figsize=(18.4*cm, 6.1*cm))
gs1 = fig.add_gridspec(nrows=1, ncols=6, wspace=1.0, hspace=1.0)
axA = fig.add_subplot(gs1[0:2])
axB1 = fig.add_subplot(gs1[2:3])
axB2 = fig.add_subplot(gs1[3:4])
axC = fig.add_subplot(gs1[4:6])

for ax, label in zip([axA, axB1, axC ], ['A', 'B', 'C']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-35/72, -11/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=16, va='bottom', fontweight='bold')
# DL spot count
plot_scatbar_DL('spots_count', 'Number of spots',
            palette, axA, data=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'])
axA.set_yticks(np.arange(0, 1550, 250))
axA.set_yticklabels(np.arange(0, 1550, 250))
# SR params
plot_scatbar(data=SR_spots_mean, ycol='smoothed_length', ylabel='Length (nm)', palette=palette, order=['AD', 'CRL'], pairs=[('AD', 'CRL')], ax=axB1, s=10)

plot_scatbar(data=SR_spots_mean, ycol='scaled_area', ylabel='Area (×10$^3$ nm$^2$)', palette=palette, order=['AD', 'CRL'], pairs=[('AD', 'CRL')], ax=axB2, s=10)

# Plot LDA
plot_lda(data=lda, palette=palette, hue='disease_state', style='tissue', ax=axC, s=100)
axC.set(ylim=(-8, 8), xlim=(-22, 12))
axC.legend(frameon=False, ncol=2, columnspacing=-0.1, loc='upper right', bbox_to_anchor=(1.05, 1.1), handletextpad=0.2)
leg = axC.get_legend()
new_labels = ['', 'AD', 'CRL', '', 'Homog.', 'Serum']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)

plt.tight_layout()
plt.savefig(f'{output_folder}Figure5_serum.svg')
plt.show()


# plot pseudo-eigens
plot_eigens(model, summary[value_cols].values, labels=value_cols, num_labels=5)
