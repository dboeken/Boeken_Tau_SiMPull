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
import matplotlib
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection


##### TO DO
# run code for both capture antibodies, for plotting only plot
# AT8-AT8 and HT7-HT7
#####
# also do number of spots for AT8 AT8 and HT7 HT7
# currently in prism
#######

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

logger.info('Import OK')

input_path = 'data/homogenate_DL_data/HT7_capture_spots_per_fov.csv'
output_folder = 'results/2_homogenate_DL/'

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

slide_params = pd.read_csv(
    f'data/homogenate_DL_data/HT7_capture_slide_parameters.csv')
slide_params.drop([col for col in slide_params.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)
# remove discard images
slide_params.dropna(subset=['keep'], inplace=True)

# expand sample info
slide_params[['capture', 'sample', 'detect']
             ] = slide_params['sample'].str.split('_', expand=True)

spots_intensity = pd.read_csv(
    'data/homogenate_DL_data/HT7_capture_compiled_spots.csv')
spots_intensity['slide_position'] = spots_intensity['well_info'].str[:4]
#spots_intensity['slide_position'] = [f'{layout}_{position}' for layout, position in spots_intensity[['layout', 'slide_position']].values]

spots_intensity = pd.merge(
    slide_params[['slide_position', 'layout', 'sample', 'capture', 'detect']],
    spots_intensity,
    on=['slide_position', 'layout'],
    how='right')

spots_intensity['log_intensity'] = np.log(spots_intensity['mean_intensity'])

#fig, ax = plt.subplots()
palette = {
    '9': 'royalblue',
    '159': 'royalblue',
    '28': 'royalblue',
    '13': 'firebrick',
    '55': 'firebrick',
    '246': 'firebrick',
    'BSA': 'lightgrey',
    'AD Mix': 'darkgrey',

}


#X

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
        test_df = df.dropna().drop_duplicates(subset=[col])
        ecdf = fit_ecdf(test_df[col])
        test_df['ecdf'] = ecdf(
            test_df.dropna().drop_duplicates(subset=[col])[col])
        combined = pd.concat([test_df.sort_values('ecdf').dropna(), test_vals])
        combined = combined.set_index('ecdf').interpolate(
            method=method, order=order).reset_index()
        interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[
            col].tolist()

    return interpolated
# visualise individual ecdfs


# interpolate ecdf for  combined visualisation
fitted_ecdfs = []
for (capture, sample, position, detect), df in spots_intensity.groupby(['capture', 'sample', 'slide_position', 'detect']):
    if detect == 'AT8':
        maxval = 1000
    if detect == 'HT7':
        maxval = 15000
    filtered_df = df[df['mean_intensity'] < maxval]
    fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
                              'mean_intensity'], method='nearest', order=False)
    fitted_ecdf['sample'] = sample
    fitted_ecdf['capture'] = capture
    fitted_ecdf['slide_position'] = position
    fitted_ecdf['detect'] = detect
    fitted_ecdfs.append(fitted_ecdf)
fitted_ecdfs = pd.concat(fitted_ecdfs)


for detect, df in fitted_ecdfs.groupby(['detect']):
    fig, ax = plt.subplots(figsize=(5, 8))
    #sns.ecdfplot(data=df, y='total', color='black')
    sns.lineplot(
        data=df.reset_index(),
        y='mean_intensity',
        x='ecdf',
        hue='sample',
        palette=palette,
        ci='sd',
        ax=ax)

    plt.title(capture)
    plt.xlabel('Cumulative distribtion')
    plt.ylabel('Mean intensity')

    plt.savefig(f'{output_folder}{detect}_hom_cumulative_mean.svg')
    plt.show()


#for (capture, sample, position, detect), df in spots_intensity.groupby(['capture','sample', 'slide_position', 'detect']):
    #mean_intensity
    #spots_intensity['mean_intensity'] = df.mean()
palette2 = {
    '9': 'darkgrey',
    '159': 'darkgrey',
    '28': 'darkgrey',
    '13': 'darkgrey',
    '55': 'darkgrey',
    '246': 'darkgrey',
    'BSA': 'darkgrey',
    'AD Mix': 'darkgrey',

}

palette3 = {
    'AD': 'dimgray',
    'CRL': 'dimgray',
    'BSA': 'dimgray',


}


palette4 = {
    '9': 'royalblue',
    '159': 'darkblue',
    '28': 'b',
    '13': 'red',
    '55': 'darkred',
    '246': 'tomato',
    'BSA': 'lightgrey',
    'AD Mix': 'darkgrey',

}



############ Mean intensity for HT7 capture ################


mean_intensity = spots_intensity.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

sample_dict = {'13': 'AD', '55': 'AD', '246': 'AD',
               '28': 'CRL', '159': 'CRL', '9': 'CRL', 'BSA': 'BSA'}
mean_intensity['disease_state'] = mean_intensity['sample'].map(sample_dict)


mean_intensity = mean_intensity[mean_intensity.detect != 'IgG']

sns.set_theme(style="ticks", font_scale=1.4)

fig, axes = plt.subplots(1, 2, figsize=(5, 4))
fig.tight_layout()
for x, (detect, df) in enumerate(mean_intensity.groupby('detect')):
    # f = sns.barplot(
    #         data = df,
    #         x = 'disease_state',
    #         y = 'mean_intensity',
    #         hue = 'sample',
    #         palette=palette,
    #         ax = axes[x],
    #         capsize=.15,
    #         errwidth=0.7,
    #         saturation= 0.8,
    #         alpha = 0.9
    #     )
    #plt.title(detect)

    sns.stripplot(
        data=df,
        x='disease_state',
        y='mean_intensity',
        hue='sample',
        ax=axes[x],
        palette=palette4,
        dodge=False,
        marker="$\circ$",
        ec="face",
        s=8,
        alpha=0.9

    )

    sns.stripplot(
        data=df.groupby(['capture', 'sample', 'detect',
                        'disease_state']).mean().reset_index(),
        x='disease_state',
        y='mean_intensity',
        hue='sample',
        ax=axes[x],
        palette=palette4,
        dodge=False,
        s=12,
        alpha=0.4

    )

    # sns.stripplot(
    #     data = df.groupby(['capture','detect', 'disease_state']).mean().reset_index(),
    #     x = 'disease_state',
    #     y = 'mean_intensity',
    #     hue = 'disease_state',
    #     ax = axes[x],
    #     palette=palette3,
    #     dodge = False,
    #     s = 10,
    #     alpha = 0.4

    # )

    sns.pointplot(
        data=df.groupby(['capture', 'detect', 'sample',
                        'disease_state']).mean().reset_index(),
        x='disease_state',
        y='mean_intensity',
        hue='disease_state',
        ax=axes[x],
        palette=palette3,
        dodge=False,
        ci='sd',
        s=10,
        alpha=0.5,
        capsize=.1,

    )

    axes[x].set_title(detect)
    axes[x].set(xlabel='Disease State', ylabel='mean intensity')
    #plt.legend(f, ['T181', 'AT8', 'colocalised'])

    #handles, labels = axes[x].get_legend_handles_labels()
    axes[0].legend('')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(1.0, 2.0))

    plt.tight_layout()

    # sns.move_legend(
    #     f, "lower center",
    #     bbox_to_anchor=(-0.3, -0.45), ncol=3, title=None, frameon=False,
    #     )
    # plt.savefig(f'{output_folder}mean_intensity.svg')

mean_intensity.to_csv(f'{output_folder}mean_intensity.csv')
