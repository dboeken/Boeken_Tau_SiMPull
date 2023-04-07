import matplotlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import matplotlib
from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms

from microfilm.microplot import microshow
from skimage.io import imread

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path =''

input_path = f'{root_path}data/homogenate_DL_data/'
output_folder = f'{root_path}results/2_homogenate_DL/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
cm = 1/2.54


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
        s=10,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xticklabels(['AD  ', 'CRL', '    BSA'])
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def scatbarplot_hue(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['AT8', 'HT7']
    hue_order = ['AD', 'CRL', 'BSA']
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
        hue_order=hue_order,
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
        s=10,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    pairs = [(('AT8', 'AD'), ('AT8', 'CRL')), (('HT7', 'AD'), ('HT7', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='detect', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    
    ax.set(ylabel=ylabel)
    
    ax.set_xlabel('')
    ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
    ax.set_xticklabels(['AD  ', 'CRL', '    BSA', 'AD  ', 'CRL', '    BSA'])
    ax.tick_params(axis='x', labelrotation=0)

    ax.annotate('AT8', xy=(0.25, group_label_y), xycoords='axes fraction', ha='center')
    ax.annotate('HT7', xy=(0.75, group_label_y), xycoords='axes fraction', ha='center')
    trans = ax.get_xaxis_transform()
    ax.plot([-0.25,0.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
    ax.plot([0.75,1.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)


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


def fitting_ecfd_for_plotting(df_intensity, detect, maxval, col='mean_intensity'):
    fitted_ecdfs = []
    for (capture, sample, position), df in df_intensity.groupby(['capture', 'sample', 'slide_position']):
        filtered_df = df[df[col] < maxval]
        fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
            col], method='nearest', order=False)
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
        y=ycol,
        x='ecdf',
        hue='sample',
        palette=palette,
        ci='sd',
        ax=ax)

    ax.set(ylabel=ylabel, xlabel='Proportion of spots')
    ax.legend(frameon=False)


def multipanel_scatbarplot(ycol, ylabel, palette, axes, data, left_lims=False, right_lims=False, group_label_y=-0.18, group_line_y=-0.05):
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
            s=1,
            order=order,
            dodge=True
        )
        if i == 0:
            # ax.set_title(detect, loc='left')
            ax.set(ylabel=ylabel, xlabel='')
            ax.set_xticks([])
            if left_lims:
                ax.set_ylim(*left_lims)
        else:
            # ax.set_title(detect, loc='right')
            ax.set_ylabel(ylabel, rotation=270, labelpad=15)
            ax.set_xlabel('')
            ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
            ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
            
            ax.annotate('AT8', xy=(0.25, group_label_y), xycoords='axes fraction', ha='center')
            ax.annotate('HT7', xy=(0.75, group_label_y), xycoords='axes fraction', ha='center')
            trans = ax.get_xaxis_transform()
            ax.plot([-0.25,0.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
            ax.plot([0.75,1.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
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
spots_summary = pd.concat([spots_AT8, spots_HT7])
spots_summary.to_csv(f'{output_folder}spots_count_summary.csv')

#brightness
slide_params_AT8_path = f'{input_path}AT8_capture_slide_parameters.csv'
slide_params_HT7_path = f'{input_path}HT7_capture_slide_parameters.csv'
spots_intensity_HT7 = f'{input_path}HT7_capture_compiled_spots.csv'
spots_intensity_AT8 = f'{input_path}AT8_capture_compiled_spots.csv'

AT8_spots_intensity = intensity_processing(
    slide_params_AT8_path, spots_intensity_AT8, 'AT8')
AT8_spots_intensity['norm_mean_intensity'] = AT8_spots_intensity['mean_intensity'] / 1000
HT7_spots_intensity = intensity_processing(
    slide_params_HT7_path, spots_intensity_HT7, 'HT7')
HT7_spots_intensity['norm_mean_intensity'] = HT7_spots_intensity['mean_intensity'] / 1000
######brightness


fitted_ecdf_HT7 = fitting_ecfd_for_plotting(HT7_spots_intensity, 'HT7', 15, col='norm_mean_intensity')

fitted_ecdf_AT8 = fitting_ecfd_for_plotting(AT8_spots_intensity, 'AT8', 1, col='norm_mean_intensity')


############ Mean intensity for HT7 capture ################
compiled_spots = pd.concat([HT7_spots_intensity, AT8_spots_intensity])

sample_dict = {'13': 'AD', '55': 'AD', '246': 'AD',
               '28': 'CRL', '159': 'CRL', '9': 'CRL', 'BSA': 'BSA'}
compiled_spots['disease_state'] = compiled_spots['sample'].map(sample_dict)

compiled_spots = compiled_spots[compiled_spots.detect != 'IgG']

mean_intensity_per_replicate = compiled_spots.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state']).mean().reset_index()

mean_intensity_per_replicate.to_csv(
    f'{output_folder}mean_intensity_per_replicate.csv')

mean_intensity_plotting = mean_intensity_per_replicate.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()

mean_intensity_plotting.to_csv(f'{output_folder}mean_intensity.csv')

# Calculate proportion of spots > threshold intensity
thresholds = {'AT8': 500, 'HT7': 2000}
compiled_spots['bright_cat'] = ['bright' if val > thresholds[detect] else 'dim' for val, detect in compiled_spots[['mean_intensity', 'detect']].values]

proportion_intensity = (compiled_spots.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout', 'bright_cat']).count()['label'] / compiled_spots.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout']).count()['label']).reset_index()
proportion_intensity['label'] = proportion_intensity['label'] * 100
proportion_intensity = pd.pivot(
    proportion_intensity,
    index=['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout'],
    columns='bright_cat',
    values='label'
).fillna(0).reset_index()

proportion_intensity_plotting = proportion_intensity.groupby(['capture', 'sample', 'detect', 'disease_state']).mean().reset_index().drop('layout', axis=1)

proportion_intensity.to_csv(f'{output_folder}proportion_intensity_per_replicate.csv')


# -----------------Read in example images-----------------

example_AD = imread(f'{root_path}data/homogenate_DL_images/X8Y2R2W2_641.tif')
example_CRL = imread(f'{root_path}data/homogenate_DL_images/X1Y0R4W2_641.tif')
# Max mean projection
example_AD = np.mean(example_AD[10:, :, :], axis=0)
example_CRL = np.mean(example_CRL[10:, :, :], axis=0)


# =====================Generate compiled plot=====================

palette = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#FB4D3D',
    '55': '#FB4D3D',
    '246': '#FB4D3D',
    'BSA': 'lightgrey',
    'AD Mix': 'darkgrey',

}

palette_DL = {
    'CRL': '#345995',
    'AD': '#FB4D3D',
    'BSA': 'darkgrey',
}

fig = plt.figure(figsize=(18.4 * cm, 3 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=3, ncols=6, wspace=0.95, hspace=0.3)
axA = fig.add_subplot(gs1[0, 0:2])
axB = fig.add_subplot(gs1[0, 2:4])
axC = fig.add_subplot(gs1[0, 4:6])
axD = fig.add_subplot(gs1[1, 0:2])
axE1 = fig.add_subplot(gs1[1, 2:3])
axE2 = fig.add_subplot(gs1[1, 3:4])
axF = fig.add_subplot(gs1[1, 4:6])
axG = fig.add_subplot(gs1[2, 0:3])
axH = fig.add_subplot(gs1[2, 3:6])

for ax, label in zip([axA, axB, axC, axD, axE1, axF, axG, axH], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.05, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')
    
# --------Panel A--------
microim1 = microshow(images=[example_AD],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axA,
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
                               rescale_type='limits', limits=[400, 1000])
axA.set_title('AD', fontsize=8)
# --------Panel B--------
microim1 = microshow(images=[example_CRL],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axB,
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
                               rescale_type='limits', limits=[400, 1000])
axB.set_title('CRL', fontsize=8)
    
# --------Panel C--------
scatbarplot_hue('spots_count', 'Number of spots',
                palette_DL, axC, spots_summary, group_line_y=-0.15, group_label_y=-0.22)

# --------Panel D--------
axD.axis('off')

# --------Panel E--------
scatbarplot(ycol='norm_mean_intensity', ylabel='Mean intensity (AU)', palette=palette_DL, ax=axE1, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'AT8'])
axE1.set_title('AT8', fontsize=8)
scatbarplot(ycol='norm_mean_intensity', ylabel='', palette=palette_DL, ax=axE2, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'HT7'])
axE2.set_title('HT7', fontsize=8)


# --------Panel F--------
scatbarplot_hue(ycol='bright', ylabel='Bright spots (%)', palette=palette_DL, ax=axF, data=proportion_intensity_plotting, group_line_y=-0.15, group_label_y=-0.22)

# --------Panel G--------
ecfd_plot('norm_mean_intensity', 'Intensity (AU)',
          palette, axG, fitted_ecdf_HT7)

# --------Panel H--------
ecfd_plot('norm_mean_intensity', 'Intensity (AU)',
          palette, axH, fitted_ecdf_AT8)

# Legend for G,H
handles, labels = axH.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9'], 'BSA': by_label['BSA']}

axG.legend(simple_legend.values(), simple_legend.keys(),
               loc='upper left', frameon=False)
axH.legend(simple_legend.values(), simple_legend.keys(),
               loc='upper left', frameon=False)


plt.tight_layout()
plt.savefig(f'{output_folder}Figure2_homogenate_DL.svg')
plt.show()
