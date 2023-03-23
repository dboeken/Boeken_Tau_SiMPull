import matplotlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import matplotlib
from statannotations.Annotator import Annotator

logger.info('Import OK')

##### TO DO
# run code for both capture antibodies, for plotting only plot
# AT8-AT8 and HT7-HT7
#####
# also do number of spots for AT8 AT8 and HT7 HT7
# currently in prism
#######

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    rooth_path =''

input_path = f'{root_path}data/homogenate_DL_data/'
output_folder = f'{root_path}results/2_homogenate_DL/'

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
        slide_params[['slide_position', 'layout',
                      'sample', 'capture', 'detect']],
        spots_intensity,
        on=['slide_position', 'layout'],
        how='right')

    spots_intensity['log_intensity'] = np.log(
        spots_intensity['mean_intensity'])
    filtered_spots_intensity = spots_intensity[spots_intensity['detect'] == detect].copy(
    )

    return filtered_spots_intensity


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


def multipanel_scatbarplot(ycol, ylabel, palette, axes, data, legend=False, left_lims=False, right_lims=False):
    order = ['AT8', 'HT7']
    for i, detect in enumerate(order):
        if i == 0:
            ax = axes
        else:
            ax = axes.twinx()
        sns.barplot(
            data=data[data['detect'] == detect],
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
            data=data[data['detect'] == detect],
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
        if i == 0:
            ax.set_title(detect, loc='left')
            ax.set(ylabel=ylabel, xlabel='')
            ax.set_xticks([])
            if left_lims:
                ax.set_ylim(*left_lims)
        else:
            ax.set_title(detect, loc='right')
            ax.set_ylabel(ylabel, rotation=270, labelpad=15)
            ax.set_xlabel('')
            ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
            ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
            if left_lims:
                ax.set_ylim(*right_lims)

        pairs = [
            ((detect, 'AD'), (detect, 'CRL')),
        ]
        annotator = Annotator(
            ax=ax, pairs=pairs, data=data, x='detect', y='mean_intensity', order=order, hue='disease_state', hue_order=['AD', 'CRL', 'BSA'])
        annotator.configure(test='t-test_ind', text_format='star',
                            loc='inside', comparisons_correction='bonferroni')
        annotator.apply_and_annotate()
        ax.legend('', frameon=False)


spots_AT8 = read_in(f'{input_path}AT8_capture_spots_per_fov.csv', 'AT8')
spots_HT7 = read_in(f'{input_path}HT7_capture_spots_per_fov.csv', 'HT7')


palette_DL = {
    'CRL': '#345995',
    'AD': '#FB4D3D',
    'BSA': 'darkgrey',
}


#brightness
slide_params_AT8_path = f'{input_path}AT8_capture_slide_parameters.csv'
slide_params_HT7_path = f'{input_path}HT7_capture_slide_parameters.csv'
spots_intensity_HT7 = f'{input_path}HT7_capture_compiled_spots.csv'
spots_intensity_AT8 = f'{input_path}AT8_capture_compiled_spots.csv'

AT8_spots_intensity = intensity_processing(
    slide_params_AT8_path, spots_intensity_AT8, 'AT8')

HT7_spots_intensity = intensity_processing(
    slide_params_HT7_path, spots_intensity_HT7, 'HT7')

######brightness
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


fitted_ecdf_HT7 = fitting_ecfd_for_plotting(HT7_spots_intensity, 'HT7', 15000)

fitted_ecdf_AT8 = fitting_ecfd_for_plotting(AT8_spots_intensity, 'AT8', 1000)


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

# sns.set_theme(style="ticks", font_scale=1.4)
mean_intensity_plotting.to_csv(f'{output_folder}mean_intensity.csv')



# ---------------Generate compiled plot---------------


fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel()
scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[0], spots_HT7)
axes[0].set_title('HT7', y=1.2)

scatbarplot('spots_count', 'Number of spots',
            palette_DL, axes[1], spots_AT8)
axes[1].set_title('AT8', y=1.2)

multipanel_scatbarplot(ycol='mean_intensity', ylabel='Mean intensity (AU)', palette=palette_DL,
                       axes=axes[4], data=mean_intensity_plotting, left_lims=False, right_lims=False)
axes[4].axvline(0.5, linestyle='-', color='black')


ecfd_plot('mean_intensity', 'mean intensity',
          palette, axes[6], fitted_ecdf_HT7)

ecfd_plot('mean_intensity', 'mean intensity',
          palette, axes[7], fitted_ecdf_AT8)
handles, labels = axes[7].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9'], 'BSA': by_label['BSA']}

axes[7].legend(simple_legend.values(), simple_legend.keys(),
               loc='upper left')
axes[6].legend(simple_legend.values(), simple_legend.keys(),
               loc='upper left')

plt.tight_layout()
