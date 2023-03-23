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
from statannotations.Annotator import Annotator


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

input_path_HT7 = 'data/homogenate_DL_data/HT7_capture_spots_per_fov.csv'
input_path_AT8 = 'data/homogenate_DL_data/AT8_capture_spots_per_fov.csv'
output_folder = 'results/2_homogenate_DL/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'


sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

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
    
    mean_spots = filtered_spots.groupby(['disease_state', 'channel', 'sample',
                                    'capture', 'detect']).mean().reset_index()

    return mean_spots


spots_AT8 = read_in(input_path_AT8, 'AT8')
spots_HT7 = read_in(input_path_HT7, 'HT7')


palette_DL = {
    'CRL': '#345995',
    'AD': '#FB4D3D',
    'BSA': 'darkgrey',
}


def scatbarplot(ycol, ylabel, palette, ax, data):
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
        s=15,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'CRL'), ('AD', 'BSA'), ('CRL', 'BSA')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='outside', comparisons_correction='bonferroni')
    annotator.apply_and_annotate()

    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(),
    #                 bbox_to_anchor=(1.0, 1.0))
    ax.legend('')


fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes = axes.ravel()
scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[0], spots_HT7)
axes[0].set_title('HT7', y=1.2)

scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[1], spots_AT8)
axes[1].set_title('AT8', y=1.2)

scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[3], spots_HT7)

plt.tight_layout()
#plt.savefig(f'{output_folder}Figure2_Homogenate_DL.svg')



#brightness
slide_params_AT8_path = 'data/homogenate_DL_data/AT8_capture_slide_parameters.csv'
slide_params_HT7_path = 'data/homogenate_DL_data/HT7_capture_slide_parameters.csv'
spots_intensity_HT7 = 'data/homogenate_DL_data/HT7_capture_compiled_spots.csv'
spots_intensity_AT8 = 'data/homogenate_DL_data/AT8_capture_compiled_spots.csv'

def intensity_processing(slide_params, spots_intensity, detect):
    slide_params = pd.read_csv(slide_params)
    slide_params.drop([col for col in slide_params.columns.tolist()
                       if 'Unnamed: ' in col], axis=1, inplace=True)

    # remove discard images
    slide_params.dropna(subset=['keep'], inplace=True)

    # expand sample info
    slide_params[['capture', 'sample', 'detect']
                ] = slide_params['sample'].str.split('_', expand=True)
    
    spots_intensity = pd.read_csv(spots_intensity)
    spots_intensity['slide_position'] = spots_intensity['well_info'].str[:4]


    spots_intensity = pd.merge(
        slide_params[['slide_position', 'layout', 'sample', 'capture', 'detect']],
        spots_intensity,
        on=['slide_position', 'layout'],
        how='right')

    spots_intensity['log_intensity'] = np.log(spots_intensity['mean_intensity'])
    filtered_spots_intensity = spots_intensity[spots_intensity['detect'] == detect].copy(
    )

    return filtered_spots_intensity


AT8_spots_intensity = intensity_processing(
    slide_params_AT8_path, spots_intensity_AT8, 'AT8')

HT7_spots_intensity = intensity_processing(
    slide_params_HT7_path, spots_intensity_HT7, 'HT7')

######brightness

# slide_params = pd.read_csv(
#     f'data/homogenate_DL_data/HT7_capture_slide_parameters.csv')
# slide_params.drop([col for col in slide_params.columns.tolist()
#                   if 'Unnamed: ' in col], axis=1, inplace=True)
# # remove discard images
# slide_params.dropna(subset=['keep'], inplace=True)

# # expand sample info
# slide_params[['capture', 'sample', 'detect']
#              ] = slide_params['sample'].str.split('_', expand=True)

# spots_intensity = pd.read_csv(
#     'data/homogenate_DL_data/HT7_capture_compiled_spots.csv')
# spots_intensity['slide_position'] = spots_intensity['well_info'].str[:4]
# #spots_intensity['slide_position'] = [f'{layout}_{position}' for layout, position in spots_intensity[['layout', 'slide_position']].values]

# spots_intensity = pd.merge(
#     slide_params[['slide_position', 'layout', 'sample', 'capture', 'detect']],
#     spots_intensity,
#     on=['slide_position', 'layout'],
#     how='right')

# spots_intensity['log_intensity'] = np.log(spots_intensity['mean_intensity'])

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
def fitting_ecfd_for_plotting(df_intensity, detect, maxval):
    fitted_ecdfs = []
    for (capture, sample, position), df in df_intensity.groupby(['capture', 'sample', 'slide_position']):
        filtered_df = df[df['mean_intensity'] < maxval]
        fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
                                'mean_intensity'], method='nearest', order=False)
        fitted_ecdf['sample'] = sample
        fitted_ecdf['capture'] = capture
        fitted_ecdf['slide_position'] = position
        fitted_ecdf['detect'] = detect
        fitted_ecdfs.append(fitted_ecdf)

    fitted_ecdfs = pd.concat(fitted_ecdfs)
    return fitted_ecdfs


fitted_ecdf_HT7 = fitting_ecfd_for_plotting(HT7_spots_intensity, 'HT7', 15000)

fitted_ecdf_AT8 = fitting_ecfd_for_plotting(AT8_spots_intensity, 'AT8', 1000)

def ecfd_plot(ycol, ylabel, palette, ax, df):
    sns.lineplot(
        data=df.reset_index(),
        y='mean_intensity',
        x='ecdf',
        hue='sample',
        palette=palette,
        ci='sd',
        ax=ax)
    
    ax.set(ylabel=ylabel, xlabel='')
    ax.legend()
    


fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel()
scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[0], spots_HT7)
axes[0].set_title('HT7', y=1.2)

scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[1], spots_AT8)
axes[1].set_title('AT8', y=1.2)

ecfd_plot('mean_intensity', 'mean intensity',
          palette, axes[6], fitted_ecdf_HT7)

ecfd_plot('mean_intensity', 'mean intensity',
          palette, axes[7], fitted_ecdf_AT8)
handles, labels = axes[7].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'], 'CRL': by_label['9'], 'BSA': by_label['BSA']}

axes[7].legend(simple_legend.values(), simple_legend.keys(),
          loc='upper left')
axes[6].legend(simple_legend.values(), simple_legend.keys(),
          loc='upper left')

plt.tight_layout()



# ######
# ######
# # interpolate ecdf for  combined visualisation
# fitted_ecdfs = []
# for (capture, sample, position), df in HT7_spots_intensity.groupby(['capture', 'sample', 'slide_position']):
#     if detect == 'AT8':
#         maxval = 1000
#     if detect == 'HT7':
#         maxval = 15000
#     filtered_df = df[df['mean_intensity'] < maxval]
#     fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
#                               'mean_intensity'], method='nearest', order=False)
#     fitted_ecdf['sample'] = sample
#     fitted_ecdf['capture'] = capture
#     fitted_ecdf['slide_position'] = position
#     fitted_ecdf['detect'] = detect
#     fitted_ecdfs.append(fitted_ecdf)
# fitted_ecdfs = pd.concat(fitted_ecdfs)




# # interpolate ecdf for  combined visualisation
# fitted_ecdfs = []
# for (capture, sample, position, detect), df in HT7_spots_intensity.groupby(['capture', 'sample', 'slide_position', 'detect']):
#     if detect == 'AT8':
#         maxval = 1000
#     if detect == 'HT7':
#         maxval = 15000
#     filtered_df = df[df['mean_intensity'] < maxval]
#     fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
#                               'mean_intensity'], method='nearest', order=False)
#     fitted_ecdf['sample'] = sample
#     fitted_ecdf['capture'] = capture
#     fitted_ecdf['slide_position'] = position
#     fitted_ecdf['detect'] = detect
#     fitted_ecdfs.append(fitted_ecdf)
# fitted_ecdfs = pd.concat(fitted_ecdfs)


# for detect, df in fitted_ecdfs.groupby(['detect']):
#     fig, ax = plt.subplots(figsize=(5, 8))
#     #sns.ecdfplot(data=df, y='total', color='black')
#     sns.lineplot(
#         data=df.reset_index(),
#         y='mean_intensity',
#         x='ecdf',
#         hue='sample',
#         palette=palette,
#         ci='sd',
#         ax=ax)

#     plt.title(capture)
#     plt.xlabel('Cumulative distribtion')
#     plt.ylabel('Mean intensity')

#     plt.savefig(f'{output_folder}{detect}_hom_cumulative_mean.svg')
#     plt.show()




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


mean_intensity_HT7 = HT7_spots_intensity.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_intensity_AT8 = AT8_spots_intensity.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_intensity = pd.concat([mean_intensity_HT7, mean_intensity_AT8])


sample_dict = {'13': 'AD', '55': 'AD', '246': 'AD',
               '28': 'CRL', '159': 'CRL', '9': 'CRL', 'BSA': 'BSA'}
mean_intensity['disease_state'] = mean_intensity['sample'].map(sample_dict)


mean_intensity = mean_intensity[mean_intensity.detect != 'IgG']

mean_intensity_per_replicate = mean_intensity.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state']).mean().reset_index()

mean_intensity_per_replicate.to_csv(
    f'{output_folder}mean_intensity_per_replicate.csv')

mean_intensity_plotting = mean_intensity_per_replicate.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()







# mean_intensity = HT7_spots_intensity.groupby(
#     ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

# sample_dict = {'13': 'AD', '55': 'AD', '246': 'AD',
#                '28': 'CRL', '159': 'CRL', '9': 'CRL', 'BSA': 'BSA'}
# mean_intensity['disease_state'] = mean_intensity['sample'].map(sample_dict)


# mean_intensity = mean_intensity[mean_intensity.detect != 'IgG']


# mean_intensity_AT8 = AT8_spots_intensity.groupby(
#     ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

# sample_dict = {'13': 'AD', '55': 'AD', '246': 'AD',
#                '28': 'CRL', '159': 'CRL', '9': 'CRL', 'BSA': 'BSA'}
# mean_intensity_AT8['disease_state'] = mean_intensity_AT8['sample'].map(
#     sample_dict)


# mean_intensity_AT8 = mean_intensity_AT8[mean_intensity_AT8.detect != 'IgG']

# mean_intensity = pd.concat([mean_intensity, mean_intensity_AT8])


# sns.set_theme(style="ticks", font_scale=1.4)


def scatbarplot_combined(ycol, ylabel, palette, ax, data, legend=False):
    order = ['AT8', 'HT7']
    sns.barplot(
        data=data,
        x='detect',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=True,
        order=order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x='detect',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=15,
        order=order,
        dodge=True
    )

    ax.set(ylabel=ylabel, xlabel='')


    if legend:#
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor = (1.0, 1.0))
    else:
        ax.legend('')





fig, axes = plt.subplots(figsize=(5, 4))
scatbarplot_combined(ycol='mean_intensity', ylabel='who cares', palette=palette_DL, ax=axes, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'AT8'])
axes.set_ylim(400, 650)
axes2 = axes.twinx()
scatbarplot_combined(ycol='mean_intensity', ylabel='who cares',
            palette=palette_DL, ax=axes2, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'HT7'], legend=True)
axes2.axvline(0.5, linestyle='-', color='black')
axes2.set_ylabel('I care', rotation=270, labelpad = 15)

pairs = [
    (('AT8', 'AD'), ('AT8', 'CRL')),
]
annotator = Annotator(
    ax=axes, pairs=pairs, data=mean_intensity_plotting, x='detect', y='mean_intensity', order=['AT8', 'HT7'], hue='disease_state', hue_order=['AD', 'CRL', 'BSA'])
annotator.configure(test='t-test_ind', text_format='star',
                    loc='inside', comparisons_correction='bonferroni')
annotator.apply_and_annotate()


pairs = [
    (('HT7', 'AD'), ('HT7', 'CRL')),
]
annotator = Annotator(
    ax=axes2, pairs=pairs, data=mean_intensity_plotting, x='detect', y='mean_intensity', order=['AT8', 'HT7'], hue='disease_state', hue_order=['AD', 'CRL', 'BSA'])
annotator.configure(test='t-test_ind', text_format='star',
                    loc='inside', comparisons_correction='bonferroni')
annotator.apply_and_annotate()
fig.tight_layout()

  
mean_intensity_plotting.to_csv(f'{output_folder}mean_intensity.csv')
