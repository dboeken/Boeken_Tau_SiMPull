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

from loguru import logger
logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path_DL = f'{root_path}data/serum_DL_data/'
input_path_SR = f'{root_path}data/serum_SR_data/'
output_folder = f'{root_path}results/spot_detection/plot_summary/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# input_AT8 = '/Users/dorotheaboeken/Documents/GitHub/230112_serum_AT8/results/spot_detection/count_spots/spots_per_fov.csv'

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


def scatbarplot_DL(ycol, ylabel, palette, ax, data):
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


def scatbarplot(ycol, ylabel, palette, axes, i, sample_type, detect, data):
    sns.barplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=axes[i],
        dodge=False,
        order=['AD', 'CRL']
    )
    sns.stripplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=axes[i],
        edgecolor='#fff',
        linewidth=1,
        s=15,
        order=['AD', 'CRL']
    )

    axes[i].set(ylabel=ylabel, xlabel='')
    axes[i].set_title(f'{detect}: {sample_type}', y=1.1)
    order = ['AD', 'CRL']
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=axes[i], pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='outside', comparisons_correction='bonferroni')
    annotator.apply_and_annotate()

    if i == 2:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[i].legend(by_label.values(), by_label.keys(),
                       bbox_to_anchor=(1.0, 1.0))
    else:
        axes[i].legend('')


def fit_ecdf(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result


def sample_ecdf(df, value_cols, num_points=100, method='nearest', order=False):
    test_vals = pd.DataFrame(
        np.arange(0, 1.01, (1/num_points)), columns=['ecdf'])
    test_vals['type'] = 'interpolated'
    interpolated = test_vals.copy()
    for col in value_cols:
        col
        test_df = df.dropna(subset=[col])
        ecdf = fit_ecdf(test_df[col])
        test_df['ecdf'] = ecdf(
            test_df.dropna(subset=[col])[col])
        combined = pd.concat([test_df.sort_values(
            'ecdf').dropna(subset=[col]), test_vals])
        combined = combined.set_index('ecdf').interpolate(
            method=method, order=order).reset_index()
        interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[
            col].tolist()
    return interpolated


# interpolate ecdf for combined visualisation
def plot_interpolated_ecdf(ycol, ylabel, properties, palette, for_plotting, sample_ecdf, max_vals=False):
    fitted_ecdfs = []
    for (sample_type, position, detect, sample), df in for_plotting.groupby(['sample_type', 'slide_position', 'detect', 'sample']):
        if max_vals:
            maxval = max_vals[detect]
            filtered_df = df[df[ycol] < maxval].copy()
        else:
            filtered_df = df.copy()
        fitted_ecdf = sample_ecdf(df=filtered_df, value_cols=[
                                  ycol], method='nearest', order=False)
        fitted_ecdf['sample'] = sample
        fitted_ecdf['detect'] = detect
        fitted_ecdf['slide_position'] = position
        fitted_ecdf['sample_type'] = sample_type
        fitted_ecdfs.append(fitted_ecdf)
    fitted_ecdfs = pd.concat(fitted_ecdfs)

    sample_palette = properties[[
        'sample_type', 'sample', 'disease_state']].drop_duplicates().copy()
    sample_palette['color'] = sample_palette['disease_state'].map(palette)
    sample_palette['key'] = sample_palette['sample_type'] + \
        '_' + sample_palette['sample']
    sample_palette = dict(sample_palette[['key', 'color']].values)

    fitted_ecdfs['key'] = fitted_ecdfs['sample_type'] + \
        '_' + fitted_ecdfs['sample']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ((detect, sample_type), df) in enumerate(fitted_ecdfs.groupby(['detect', 'sample_type'])):
        sns.lineplot(
            data=df.reset_index(),
            y=ycol,
            x='ecdf',
            hue='key',
            palette=sample_palette,
            ci='sd',
            ax=axes[i]
        )

        axes[i].set(title=f'{detect} {sample_type}',
                    xlabel='Cumulative distribtion', ylabel=ylabel)
    return fig, axes




DL_spots_AT8 = read_in(f'{input_path_DL}AT8_spots_per_fov.csv', 'AT8')
DL_spots_HT7 = read_in(f'{input_path_DL}HT7_spots_per_fov.csv', 'HT7')
DL_spots_summary = pd.concat([DL_spots_AT8, DL_spots_HT7]).reset_index()

DL_spots_summary.to_csv(f'{output_folder}DL_spots_count_summary.csv')

SR_spots = pd.read_csv(
    f'{input_path_SR}properties_compiled.csv')

SR_spots.drop([col for col in SR_spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)

SR_spots = SR_spots.dropna(subset=['label']).copy()

tissue_dict = {'2022-07-06_21-19-49': 'brain',
               '2023-03-02_14-31-20': 'serum', '2023-03-02_18-25-28': 'serum'}
SR_spots['tissue'] = SR_spots['folder'].astype(
    str).map(tissue_dict)

serum_dict = {'1': 'CRL', '2': 'CRL', 'BSA': 'discard',
              '3': 'AD', '4': 'AD', '5': 'AD', '6': 'discard', '7': 'AD', '8': 'CRL', '9': 'AD', '10': 'AD', '11': 'discard', '12': 'CRL', '13': 'CRL', '15': 'AD', '16': 'CRL', '17': 'CRL', '18': 'CRL', '19': 'CRL', '20': 'AD', 'IgG': 'discard'}
brain_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'discard',
              '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD', 'IgG': 'discard'}



SR_spots['sample'] = [str(int(float(sample))) if sample not in ['BSA', 'IgG'] else sample for sample in SR_spots['sample']]
SR_spots['disease_state'] = [serum_dict[sample] if tissue == 'serum' else brain_dict[sample] for sample, tissue in SR_spots[['sample', 'tissue']].values]

SR_spots = SR_spots[SR_spots['disease_state']!= 'discard'].copy()

SR_spots['length_cat'] = ['long' if length >
                          150 else 'short' for length in SR_spots['smoothed_length']]

SR_spots['area_cat'] = ['large' if area >
                        4000 else 'small' for area in SR_spots['area']]

SR_spots['ecc_cat'] = ['round' if ecc <
                       0.9 else 'fibrillar' for ecc in SR_spots['eccentricity']]



#plot DL


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.1)


scatbarplot_DL('spots_count', 'Mean number spots',
            palette, axes[0], data=DL_spots_summary[DL_spots_summary['detect'] == 'AT8'])

scatbarplot_DL('spots_count', 'Mean number spots',
            palette, axes[1], data=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'])

########


for_plotting = SR_spots[
    (SR_spots['prop_type'] == 'cluster')
].copy()
for_plotting = for_plotting[for_plotting['smoothed_length'] > 50].copy()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, ((tissue, detect), df) in enumerate(for_plotting.groupby(['tissue', 'detect'])):
    data = df.groupby(['folder', 'well_info', 'sample', 'disease_state']).mean(
    ).reset_index().groupby(['sample', 'disease_state']).mean().reset_index()
    #mean of replicate means per sample
    scatbarplot('smoothed_length', 'Average length (nm)',
                palette, axes, i, tissue, detect, data)
plt.tight_layout()



fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, ((tissue, detect), df) in enumerate(for_plotting.groupby(['tissue', 'detect'])):
    data = df.groupby(['folder', 'well_info', 'sample', 'disease_state']).mean(
    ).reset_index().groupby(['sample', 'disease_state']).mean().reset_index()
    #mean of replicate means per sample
    scatbarplot('eccentricity', 'Mean eccentricity',
                palette, axes, i, tissue, detect, data)
plt.tight_layout()
plt.savefig(f'{output_folder}avg-eccentricity.svg')


fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, ((tissue, detect), df) in enumerate(for_plotting.groupby(['tissue', 'detect'])):
    data = df.groupby(['folder', 'well_info', 'sample', 'disease_state']).mean(
    ).reset_index().groupby(['sample', 'disease_state']).mean().reset_index()
    #mean of replicate means per sample
    scatbarplot('area', 'Mean area',
                palette, axes, i, tissue, detect, data)
plt.tight_layout()


for_plotting = SR_spots[
    (~SR_spots['sample'].isin(['BSA', 'IgG'])) &
    (SR_spots['prop_type'] == 'smooth') &
    (SR_spots['smoothed_length'] > 30)
].copy()

# Proportion of > 30 nm are fibrillar
proportion = (for_plotting.groupby(['detect', 'tissue', 'disease_state', 'sample', 'ecc_cat']).count()[
              'smoothed_label'] / for_plotting.groupby(['detect', 'tissue', 'disease_state', 'sample', ]).count()['smoothed_label']).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, ((tissue, detect), df) in enumerate(proportion.groupby(['tissue', 'detect'])):
    #mean of replicate means per sample
    scatbarplot('smoothed_label', 'Proportion of fibrils (%)',
                palette, axes, i, tissue, detect, df[df['ecc_cat'] == 'fibrillar'])
    axes[i].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{output_folder}+30nm_prop-fibrils.svg')


# Proportion of fibrillar > 150 nm
proportion = (for_plotting[for_plotting['ecc_cat'] == 'fibrillar'].groupby(['detect', 'tissue', 'disease_state', 'sample', 'length_cat']).count()[
              'smoothed_label'] / for_plotting[for_plotting['ecc_cat'] == 'fibrillar'].groupby(['detect', 'tissue', 'disease_state', 'sample', ]).count()['smoothed_label']).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, ((tissue, detect), df) in enumerate(proportion.groupby(['tissue', 'detect'])):
    #mean of replicate means per sample
    scatbarplot('smoothed_label', 'Proportion of fibrils > 150 nm (%)',
                palette, axes, i, tissue, detect, df[df['length_cat'] == 'long'])
    axes[i].set_ylim(0, 1)
plt.tight_layout()



########

# selecting super res data for dim reduction
summary = SR_spots[(SR_spots['disease_state'].isin(['AD', 'CRL'])) & (SR_spots['capture'] == 'HT7') & (
    SR_spots['prop_type'] == 'smooth')].groupby(['capture', 'tissue', 'disease_state', 'sample']).mean().copy().reset_index()
summary['key'] = summary['tissue'] + '_' + summary['disease_state']

# map localisations from cluster to smoothed
locs_dict = dict(SR_spots[SR_spots['prop_type'] ==
                 'cluster'][['key', '#locs']].values)

# Collect diff limited data for serum

HT7_DL = pd.read_csv('results/2_homogenate_DL/spots_count_summary.csv')


# Collect diff limited data for brain


# merge super res and diff limited datasets


# Select columns for dim reduction
group_cols = ['tissue', 'disease_state', 'sample']
value_cols = ['area', 'eccentricity', 'perimeter',
              'minor_axis_length', 'major_axis_length','smoothed_length']


# Complete PCA
X = summary.set_index(group_cols)[value_cols].values
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

summary[['dim1', 'dim2']] = pca.transform(X)

fig, ax = plt.subplots(figsize=(6, 5.5))
sns.scatterplot(
    data=summary,
    x='dim1',
    y='dim2',
    hue='disease_state',
    style='tissue',
    palette=palette,
    s=300
)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# Comlete LDA
X = summary[value_cols].values
y = summary['key'].values
lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
lda_model = lda.fit(X, y)

summary[['dim1', 'dim2']] = lda_model.transform(X)


fig, ax = plt.subplots(figsize=(6, 5.5))
sns.scatterplot(
    data=summary,
    x='dim1',
    y='dim2',
    hue='disease_state',
    style='tissue',
    palette=palette,
    s=300
)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# Visualise pairplot for serum vs tissue
sns.pairplot(
    summary[value_cols+['key']], hue='key', palette={'serum_AD': '#9A031E', 'serum_CRL': '#16507E', 'brain_AD': '#5F0F40', 'brain_CRL': '#CBE3F6'})
plt.show()



########

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
