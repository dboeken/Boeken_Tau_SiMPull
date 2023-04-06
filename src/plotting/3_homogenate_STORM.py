###include:

# 1: example images
# 2: cumulative distribution + error bar
# 3: mean length and area
# 4: ratios of large aggregates


from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms

from microfilm.microplot import microshow
from skimage.io import imread


from loguru import logger
logger.info('Import OK')

data_path = 'data/data_path.txt'

if os.path.exists(data_path):
    root_path = open(data_path, 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/homogenate_SR_data/properties_compiled.csv'
#input_samplemap = 'raw_data/sample_map.csv'
output_folder = 'results/super-res/summary/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Read in summary measurements
properties = pd.read_csv(f'{input_path}')
properties.drop([col for col in properties.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)


sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

properties['disease_state'] = properties['sample'].astype(
        str).map(sample_dict)

properties['sample'] = properties['sample'].astype(str)

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
    order = ['AD', 'CRL']
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
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def scatbarplot_hue(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['fibril', 'round']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='ecc_cat',
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
        x='ecc_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=15,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    # add by chance lines by disease state
    # for disease, df2 in data.groupby('disease_state'):
    #     ax.axhline(df2['chance_proportion_coloc'].mean(),
    #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    pairs = [(('fibril', 'AD'), ('round', 'AD')),
             (('fibril', 'CRL'), ('round', 'CRL')), (('fibril', 'AD'), ('fibril', 'CRL')), (('round', 'AD'), ('round', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='ecc_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    # ax.set_xlabel('')
    # ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
    # ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
    # ax.set_yticks([0, 25, 50, 75, 100])
    # ax.set_yticklabels(['0', '25', '50', '75', '100'])
    # ax.annotate('AT8', xy=(0.25, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # ax.annotate('T181', xy=(0.75, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # trans = ax.get_xaxis_transform()
    # ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)
    # ax.plot([0.75, 1.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)




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


def fitting_ecfd_for_plotting(df_intensity, detect, maxval, col):
    fitted_ecdfs = []
    for (sample, disease_state), df in df_intensity.groupby(['sample', 'disease_state']):
        filtered_df = df[df[col] < maxval].copy()
        fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
            col], method='nearest', order=False)
        fitted_ecdf['sample'] = sample
        fitted_ecdf['disease_state'] = disease_state
        # fitted_ecdf['capture'] = capture
        # fitted_ecdf['slide_position'] = position
        fitted_ecdf['detect'] = detect
        fitted_ecdfs.append(fitted_ecdf)

    fitted_ecdfs = pd.concat(fitted_ecdfs)
    return fitted_ecdfs




def ecfd_plot(ycol, ylabel, palette, ax, df):
    sns.lineplot(
        data=df.reset_index(),
        y=ycol,
        x='ecdf',
        hue='disease_state',
        palette=palette,
        ci='sd',
        ax=ax)

    ax.set(ylabel=ylabel, xlabel='Proportion of spots')
    ax.legend(frameon=False)


# ---------------------Visualise mean length ( > 30 nm)---------------------
# remove things outside range of interest
for_plotting = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['smoothed_length'] > 50) &
    (properties['area'] > 2) 
].copy()

for_plotting['scaled_area'] = for_plotting['area'] * (107/8)**2
for_plotting['scaled_perimeter'] = for_plotting['perimeter'] * (107/8)

### Mean length >30 nm
for_plotting_per_replicate = for_plotting.groupby(['disease_state', 'sample', 'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_mean = for_plotting_per_replicate.groupby(['disease_state', 'sample','capture', 'detect']).mean().reset_index()


###
# Calculate proportion of spots > threshold intensity
thresholds = {
    'length': 250,
    'scaled_area': 15000,
    'eccentricity': 0.9, 
    'perimeter': 550
}

for_plotting['length_cat'] = ['long' if val > thresholds['length']
                                   else 'short' for val, detect in for_plotting[['smoothed_length', 'detect']].values]

proportion_length = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_length['label'] = proportion_length['label'] * 100
proportion_length = pd.pivot(
    proportion_length,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='length_cat',
    values='label'
).fillna(0).reset_index()

proportion_length_plotting = proportion_length.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# Calculate proportion of spots > threshold intensity
for_plotting['area_cat'] = ['large' if val > thresholds['scaled_area']
                             else 'small' for val, detect in for_plotting[['scaled_area', 'detect']].values]

proportion_size = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'area_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_size['label'] = proportion_size['label'] * 100
proportion_size = pd.pivot(
    proportion_size,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='area_cat',
    values='label'
).fillna(0).reset_index()

proportion_size_plotting = proportion_size.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# perimeter ratio
for_plotting['perimeter_cat'] = ['long' if val > thresholds['perimeter']
                                 else 'short' for val, detect in for_plotting[['scaled_perimeter', 'detect']].values]

proportion_perimeter = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'perimeter_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_perimeter['label'] = proportion_perimeter['label'] * 100
proportion_perimeter = pd.pivot(
    proportion_perimeter,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='perimeter_cat',
    values='label'
).fillna(0).reset_index()

proportion_perimeter_plotting = proportion_perimeter.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



#### calulcare proportion of fibrils

for_plotting['ecc_cat'] = ['fibril' if val > thresholds['eccentricity']
                              else 'round' for val, detect in for_plotting[['eccentricity', 'detect']].values]

proportion_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_ecc['label'] = proportion_ecc['label'] * 100
proportion_ecc = pd.pivot(
    proportion_ecc,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='ecc_cat',
    values='label'
).fillna(0).reset_index()

proportion_ecc_plotting = proportion_ecc.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



#### ecdf

fitted_ecdf_area = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 30000, col='scaled_area')

fitted_ecdf_smoothed_length = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 1000, col='smoothed_length')

fitted_ecdf_perimeter = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 1000, col='scaled_perimeter')


#### mean length/area of fibrils vs round

whatever_per_replicate = for_plotting.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter']].reset_index()

whatever = for_plotting.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter']].reset_index()


####tar


length_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'length_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count()['label']).reset_index()



length_ecc_plotting = length_ecc.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'length_cat', 'ecc_cat']).mean().reset_index()

length_ecc_plotting = length_ecc_plotting[
    length_ecc_plotting['ecc_cat'] == 'fibril'].copy()
length_ecc_plotting = length_ecc_plotting[
    length_ecc_plotting['length_cat'] == 'long'].copy().reset_index()


# # Make main figure
# fig, axes = plt.subplots(5, 3, figsize=(12, 18))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.1)


# scatbarplot('smoothed_length', 'Average length [nm]',
#             palette, axes[1], for_plotting_mean)

# scatbarplot('scaled_area', 'Mean area [nm$^2$]',
#             palette, axes[4], for_plotting_mean)

# ecfd_plot('smoothed_length', 'Length',
#           palette, axes[0], fitted_ecdf_smoothed_length)

# ecfd_plot('scaled_area', 'Area',
#           palette, axes[3], fitted_ecdf_area)
# #axes[3].set_ylim(0, 50000)

# scatbarplot('long', 'Long [%]',
#             palette, axes[2], proportion_length_plotting)

# scatbarplot('large', 'Large [%]',
#             palette, axes[5], proportion_size_plotting)


# scatbarplot('fibril', 'Fibrils [%]',
#             palette, axes[8], proportion_ecc_plotting)

# scatbarplot('eccentricity', 'Mean eccentricity',
#             palette, axes[7], for_plotting_mean)


# scatbarplot_hue(ycol='scaled_area', ylabel='area',
#                 palette=palette, ax=axes[9], data=whatever)


# scatbarplot_hue(ycol='smoothed_length', ylabel='length',
#                 palette=palette, ax=axes[10], data=whatever)

# scatbarplot_hue(ycol='smoothed_length', ylabel='perimeter',
#                 palette=palette, ax=axes[11], data=whatever)


# scatbarplot('scaled_perimeter', 'Perimeter',
#             palette, axes[13], for_plotting_mean)

# scatbarplot('long', 'perimeter long [%]',
#             palette, axes[14], proportion_perimeter_plotting)

# scatbarplot('label', 'Large [%]',
#             palette, axes[12], length_ecc_plotting)

# plt.tight_layout()


##### plot


fig = plt.figure(figsize=(12, 5))
gs1 = fig.add_gridspec(nrows=4, ncols=6, wspace=0.9, hspace=0.8)
axA1 = fig.add_subplot(gs1[0:1, 0:1])
axA2 = fig.add_subplot(gs1[0:1, 1:2])
axA3 = fig.add_subplot(gs1[1:2, 0:1])
axA4 = fig.add_subplot(gs1[1:2, 1:2])
axB1 = fig.add_subplot(gs1[0:2, 2:3])
axB2 = fig.add_subplot(gs1[0:2, 3:4])
axB3 = fig.add_subplot(gs1[0:2, 4:5])
axB4 = fig.add_subplot(gs1[0:2, 5:6])
axC1 = fig.add_subplot(gs1[2:4, 0:2])
axC2 = fig.add_subplot(gs1[2:3, 0:1])
axD1 = fig.add_subplot(gs1[2:4, 2:3])
axD2 = fig.add_subplot(gs1[2:4, 3:4])
axD3 = fig.add_subplot(gs1[2:4, 4:5])
axD4 = fig.add_subplot(gs1[2:4, 5:6])

# axE = fig.add_subplot(gs1[4:6, 0:2])
# axF = fig.add_subplot(gs1[4:6, 2:4])
# axG = fig.add_subplot(gs1[4:6, 4:6])


for ax, label in zip([axA1, axB1, axC1, axD1 ], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-40/72, -11/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=16, va='bottom', fontweight='bold')

# --------Panel A--------
# microim1 = microshow(images=[example_AD],
#                      cmaps=['Greys'], flip_map=[True],
#                      label_color='black', ax=axA,
#                      unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
#                      rescale_type='limits', limits=[400, 1000])
# axA.set_title('AD')
# --------Panel B--------
scatbarplot('smoothed_length', 'Mean length [nm]',
            palette, axB1, for_plotting_mean)
axB1.set_title('Length')

scatbarplot('scaled_perimeter', 'Mean perimeter [nm]',
            palette, axB2, for_plotting_mean)
axB2.set_title('Perimeter')

scatbarplot('scaled_area', 'Mean area [nm$^2$]',
            palette, axB3, for_plotting_mean)
axB3.set_title('Area')

scatbarplot('eccentricity', 'Mean eccentricity',
            palette, axB4, for_plotting_mean)
axB4.set_title('Eccentricity')

# --------Panel C--------
ecfd_plot('scaled_perimeter', 'Perimeter',
          palette, axC1, fitted_ecdf_perimeter)

ecfd_plot('scaled_perimeter', 'Perimeter',
          palette, axC2, fitted_ecdf_perimeter)
axC2.set_xlim(0.7, 1)
axC2.set(xlabel=None)
axC2.yaxis.set_label_position("right")
axC2.yaxis.tick_right()

# --------Panel D--------
scatbarplot('long', 'Long [%]',
            palette, axD1, proportion_length_plotting)

scatbarplot('long', 'perimeter long [%]',
            palette, axD2, proportion_perimeter_plotting)

scatbarplot('large', 'Large [%]',
            palette, axD3, proportion_size_plotting)


scatbarplot('fibril', 'Fibrils [%]',
            palette, axD4, proportion_ecc_plotting)


# # --------Panel E--------
# scatbarplot_hue(ycol='smoothed_length', ylabel='length',
#                 palette=palette, ax=axE, data=whatever)

# # --------Panel F--------
# scatbarplot_hue(ycol='scaled_perimeter', ylabel='perimeter',
#                 palette=palette, ax=axF, data=whatever)

# # --------Panel G--------
# scatbarplot_hue(ycol='scaled_area', ylabel='area',
#                 palette=palette, ax=axG, data=whatever)



# # Legend for G,H
# handles, labels = axH.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# simple_legend = {'AD': by_label['13'],
#                  'CRL': by_label['9'], 'BSA': by_label['BSA']}

# axG.legend(simple_legend.values(), simple_legend.keys(),
#            loc='upper left', frameon=False)
# axH.legend(simple_legend.values(), simple_legend.keys(),
#            loc='upper left', frameon=False)

plt.tight_layout()
plt.show()


#### supplemental figure

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.1)


scatbarplot_hue(ycol='scaled_area', ylabel='area',
                palette=palette, ax=axes[0], data=whatever)


scatbarplot_hue(ycol='smoothed_length', ylabel='length',
                palette=palette, ax=axes[1], data=whatever)

scatbarplot_hue(ycol='smoothed_length', ylabel='perimeter',
                palette=palette, ax=axes[2], data=whatever)

plt.tight_layout()
