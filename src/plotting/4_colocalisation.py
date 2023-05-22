import functools
import os
import re
from lib2to3.pgen2.pgen import DFAState

from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr
from skimage.io import imread

from microfilm.microplot import microshow

from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms

from microfilm.microplot import microshow
from skimage.io import imread

# from smma.src.utilities import closest_node, find_pairs
# from smma.src.visualise import plot_colocalisation


logger.info('Import OK')

if os.path.exists('data\data_path.txt'):
    root_path = open('data\data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/colocalisation_data/colocalisation_summary.csv'
input_path_spots = f'{root_path}data/colocalisation_data/colocalisation_spots.csv'
output_folder = 'results/4_colocalisation/'
image_path = f'{root_path}data/colocalisation_images/Composite.tif'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


font = {'family': 'normal',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54

# Read in summary FOV data
coloc_spots = pd.read_csv(f'{input_path}')
coloc_spots.drop([col for col in coloc_spots.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)

coloc_spots['position'] = coloc_spots['fov'].str[:4]
#coloc_spots['spots_count'] = coloc_spots['spots_count'].fillna(0)

#calculate mean
meandf = coloc_spots.groupby(
    ['sample', 'position', 'channel', 'detect', 'capture', 'layout']).mean().reset_index()

sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}
antibody_dict = {488: 'T181', 641: 'AT8', 'colocalised': 'colocalised'}

#meandf['disease_state']= meandf['sample'].map(sample_dict)
meandf['disease_state'] = meandf['sample'].astype(str).map(sample_dict)

#only colocalised images
filtered_meandf = meandf[meandf['detect'] == 'AT8 r T181 b'].copy()

#remove one set of channels for coloc spots
filtered_coloc = filtered_meandf[filtered_meandf['channel'] == 488].copy()
filtered_coloc = filtered_coloc[['disease_state', 'channel', 'sample',
                                 'capture', 'coloc_spots']].rename(columns={'coloc_spots': 'total_spots'})
filtered_coloc['channel'] = 'colocalised'


for_plotting = pd.concat([filtered_meandf[[
                         'disease_state', 'channel', 'sample', 'capture', 'total_spots']], filtered_coloc])

for_plotting['antibody_state'] = for_plotting['channel'].map(antibody_dict)

#plotting proportion of colocalised spots

linestyles = {
    'AD': '--',
    'CRL': 'dotted',
    'BSA': 'solid'
}

for_plotting_proportion = filtered_meandf[[
    'disease_state', 'channel', 'sample', 'capture', 'proportion_coloc', 'chance_proportion_coloc']].copy()
# for_plotting_proportion = filtered_meandf[[
#     'disease_state', 'channel', 'sample', 'capture', 'proportion_coloc']].copy()
for_plotting_proportion['antibody_state'] = for_plotting_proportion['channel'].map(
    antibody_dict)

mean_for_plotting_proportion = for_plotting_proportion.groupby(
    ['capture', 'sample', 'channel', 'disease_state']).mean().reset_index()


##################################################################
################### intensity plotting ###########################
##################################################################

for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample',
                                          'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(
    antibody_dict)

melted = pd.melt(for_plotting_intensity, id_vars=['disease_state', 'channel', 'sample', 'capture', 'mean_intensity', 'antibody_state'], value_vars=[
                 'mean_intensity-coloc', 'mean_intensity-noncoloc'], value_name='coloc_intensity', var_name='coloc_status')

melted['coloc_status'] = melted['coloc_status'].str.replace(
    'mean_intensity-', '')
melted['key'] = melted['coloc_status'] + \
    '_' + melted['antibody_state'].astype(str)

#################################################################
##########plot ratio for intensity coloc vs non-coloc############
#################################################################
for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample',
                                          'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(
    antibody_dict)
for_plotting_intensity['intensity_ratio'] = for_plotting_intensity['mean_intensity-coloc'] / \
    for_plotting_intensity['mean_intensity-noncoloc']
for_plotting_intensity['log2_intensity_ratio'] = np.log2(
    for_plotting_intensity['intensity_ratio'])



########### stochiometry
input_path_spots = f'{root_path}data/colocalisation_data/colocalisation_spots.csv'

colocalised_summary = pd.read_csv(f'{input_path_spots}')

colocalised = colocalised_summary[colocalised_summary['coloc?'] == 1].copy()
sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}
colocalised['disease_state'] = colocalised['sample'].astype(
    str).map(sample_dict)

#sample names strings vs integers
# colocalised[['sample', 'capture', 'detect', 'disease_state']].drop_duplicates()['sample'].unique()


#merge table on mean intenisty of colocalised spots in both channels and update labels
for_plotting = pd.pivot(colocalised, index=['fov', 'layout', 'sample', 'capture', 'pair_id', 'disease_state'], columns=[
                        'channel'], values=['mean_intensity']).reset_index()
for_plotting.columns = [f'{x}' if y ==
                        '' else f'{x}_{y}' for x, y in for_plotting.columns]

palette = sns.color_palette(['#ffffff'] +
                            list(sns.color_palette('magma', n_colors=200).as_hex())[66:])



#############


new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)
cmap

filtered_disease = for_plotting[for_plotting['disease_state'] == 'AD'].copy()


##########

##########
##########
palette_DL = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
}

palette_coloc = {
    641: '#F03A47',
    488: '#F03A47',
    'BSA': '#A9A9A9',
}


new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)


def hexbinplotting(ylabel, colour, ax, data, capture):

    df = data[data['capture'] == capture].copy()
    hexs = ax.hexbin(data=df, x='norm_mean_intensity_641',
              y='norm_mean_intensity_488', cmap=colour, vmin=0, vmax=900)
    ax.set(ylabel=ylabel)
    ax.set(xlabel ='mean intensity 638')
    sns.kdeplot(data=df, x='norm_mean_intensity_641', y='norm_mean_intensity_488', color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3)

    return hexs


def scatbarplot_hue(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = [641, 488]
    hue_order = ['AD', 'CRL', 'BSA']
    sns.barplot(
        data=data,
        x='channel',
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
        x='channel',
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

    #add by chance lines by disease state
    for disease, df2 in data.groupby('disease_state'):
        ax.axhline(df2['chance_proportion_coloc'].mean(),
                   linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    pairs = [((641, 'AD'), (641, 'CRL')), ((488, 'AD'), (488, 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='channel', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    ax.set_xlabel('')
    ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
    ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    ax.annotate('AT8', xy=(0.25, group_label_y),
                xycoords='axes fraction', ha='center')
    ax.annotate('T181', xy=(0.75, group_label_y),
                xycoords='axes fraction', ha='center')
    trans = ax.get_xaxis_transform()
    ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)
    ax.plot([0.75, 1.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)


def scatbarplot_hue_intensity(ycol, ylabel, palette, ax, data, stats_df, group_label_y=-0.18, group_line_y=-0.05,):
    order = ['AT8', 'T181']
    hue_order = [641, 488]
    sns.barplot(
        data=data,
        x='capture',
        y=ycol,
        hue='channel',
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
        x='capture',
        y=ycol,
        hue='channel',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=10,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    for i, capture in enumerate(order):
        for detect in hue_order:
            star = stats_df[(stats_df['capture'] == capture) & (
                stats_df['channel'] == detect)]['significance'].tolist()[0]
            j = -0.2 if detect == 641 else 0.2
            ax.annotate(star, xy=(i+j, 2.2),
                        xycoords='data', ha='center')

    ax.set(ylabel=ylabel)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
    ax.set_xlabel('')
    ax.set_xticks([-0.2, 0.2, 0.8, 1.2])
    ax.set_xticklabels(['AT8', 'T181', 'AT8', 'T181'])

    ax.annotate('AT8 capture', xy=(0.25, group_label_y),
                xycoords='axes fraction', ha='center')
    ax.annotate('T181 capture', xy=(0.75, group_label_y),
                xycoords='axes fraction', ha='center')
    trans = ax.get_xaxis_transform()
    ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)
    ax.plot([0.75, 1.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)


AT8_mean_for_plotting_proportion = mean_for_plotting_proportion[mean_for_plotting_proportion['capture']=='AT8'].copy()

T181_mean_for_plotting_proportion = mean_for_plotting_proportion[mean_for_plotting_proportion['capture'] == 'T181'].copy(
)

AD_brightness_per_replicate = for_plotting_intensity[for_plotting_intensity['sample'].isin(['13', '55', '246'])].copy()

AD_brightness_plotting = AD_brightness_per_replicate.groupby(
    ['capture', 'sample', 'channel', 'disease_state']).mean().reset_index()


filtered_disease['norm_mean_intensity_488'] = filtered_disease['mean_intensity_488'] / 1000

filtered_disease['norm_mean_intensity_641'] = filtered_disease['mean_intensity_641'] / 1000



brightness_ratio_pval = []
for (capture, channel), df in AD_brightness_plotting.groupby(['capture', 'channel']):
    _, pval = stats.ttest_1samp(df['intensity_ratio'], popmean=1)
    brightness_ratio_pval.append([capture, channel, pval])
# make new df with channel, capture, p value and star rating
brightness_ratio_stats = pd.DataFrame(brightness_ratio_pval, columns=[
    'capture', 'channel', 'pval'])
brightness_ratio_stats['significance'] = ['****' if val < 0.0001 else ('***' if val < 0.001 else (
    '**' if val < 0.01 else ('*' if val < 0.05 else 'ns')))for val in brightness_ratio_stats['pval']]


AT8_mean_for_plotting_proportion.to_csv(
    f'{output_folder}AT8_mean_for_plotting_proportion.csv')
T181_mean_for_plotting_proportion.to_csv(
    f'{output_folder}T181_mean_for_plotting_proportion.csv')
filtered_disease.to_csv(
    f'{output_folder}filtered_disease.csv')
AD_brightness_plotting.to_csv(
    f'{output_folder}AD_brightness_plotting.csv')

## Read image
example_coloc = imread(image_path)

# Make main figure
fig, axes = plt.subplots(2, 3, figsize=(18.4 * cm, 2 * 6.1 * cm))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                top=None, wspace=-1, hspace=1)


for x, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-0.6, 0., fig.dpi_scale_trans)
    axes[x].text(0.0, 1.0, label, transform=axes[x].transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

microim1 = microshow(
    images=[example_coloc[0, :, :], example_coloc[1, :, :]], 
    cmaps=['pure_magenta', 'pure_green'], #flip_map=[True],
    label_color='black', ax=axes[0], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 7000], [0, 3500]])

axes[0].axis('off')

scatbarplot_hue('proportion_coloc', 'Colocalised [%]',
                palette_DL, axes[1], AT8_mean_for_plotting_proportion, group_line_y=-0.15, group_label_y=-0.22)
axes[1].set_ylim(0, 110)
axes[1].set_title('AT8 capture', fontsize=8)


scatbarplot_hue('proportion_coloc', 'Colocalised [%]',
                palette_DL, axes[2], T181_mean_for_plotting_proportion, group_line_y=-0.15, group_label_y=-0.22)

axes[2].set_ylim(0, 110)
axes[2].set_title('T181 capture', fontsize=8)

scatbarplot_hue_intensity('intensity_ratio', 'Intensity Ratio',
                          palette_coloc, axes[3], AD_brightness_plotting, brightness_ratio_stats, group_line_y=-0.15, group_label_y=-0.22)

axes[3].set_ylim(0, 2.6)
axes[3].axhline(1, linestyle='--', linewidth=1.2, color='#4c4c52')

hexs4 = hexbinplotting('mean intensity 488', cmap,
               axes[4], filtered_disease, 'AT8')
               
cb=plt.colorbar(hexs4, ax=axes[4])
axes[4].set_title('AT8 capture', fontsize=8)


hexs5 = hexbinplotting('mean intensity 488', cmap,
               axes[5], filtered_disease, 'T181')

cb=plt.colorbar(hexs5, ax=axes[5])
cb.set_label('Count', rotation=270, labelpad=15)
axes[5].set_title('T181 capture', fontsize=8)

plt.tight_layout()

plt.savefig(f'{output_folder}Figure4_coloc.svg')


