"""
Generating Figure 4
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import ListedColormap
from microfilm.microplot import microshow

from src.utils import plot_hexbin, scatbar

logger.info('Import OK')

# =================Set paths=================
input_folder = 'results/4_colocalisation/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

palette = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
    641: '#F03A47',
    488: '#F03A47',
    'AT8': '#F03A47',
    'T181': '#F03A47',
}

linestyles = {
    'AD': '--',
    'CRL': 'dotted',
    'BSA': 'solid'
}

new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)


# =========Organise data========
# Read image
example_coloc = np.load(f'{input_folder}example_image.npy')

# Read datasets
mean_for_plotting_proportion = pd.read_csv(
    f'{input_folder}mean_for_plotting_proportion.csv')
filtered_disease = pd.read_csv(
    f'{input_folder}filtered_disease.csv')
AD_brightness_plotting = pd.read_csv(
    f'{input_folder}AD_brightness_plotting.csv')
brightness_ratio_stats = pd.read_csv(
    f'{input_folder}brightness_ratio_stats.csv')

# =========Generate figure========
fig = plt.figure(figsize=(18.4 * cm, 2 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=2, ncols=3, wspace=0.45, hspace=0.4)
axA = fig.add_subplot(gs1[0, 0])
axB = fig.add_subplot(gs1[0, 1])
axC = fig.add_subplot(gs1[0, 2])
axD = fig.add_subplot(gs1[1, 0])
axE = fig.add_subplot(gs1[1, 1])
axF = fig.add_subplot(gs1[1, 2])

for ax, label in zip([axA, axB, axC, axD, axE, axF], ['A', 'B', 'C', 'D', 'E', 'F']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-0.6, 0., fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
microim1 = microshow(
    images=[example_coloc[0, :, :], example_coloc[1, :, :]], 
    cmaps=['pure_magenta', 'pure_green'],
    label_color='black', ax=axA, unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 7000], [0, 3500]])

axA.axis('off')

# --------Panel B/C--------
axes = [axB, axC]
for i, (capture, df) in enumerate(mean_for_plotting_proportion.groupby('capture')):
    scatbar(
        dataframe=df, 
        xcol='channel', ycol='proportion_coloc', ax=axes[i], xorder=[641, 488],
        dotpalette=palette, barpalette=palette,
        hue_col='disease_state', hue_order=['AD', 'CRL'], 
        pairs=[((641, 'AD'), (641, 'CRL')), ((488, 'AD'), (488, 'CRL'))],
        groups=['AT8', 'T181'], group_label_y=-0.22, group_line_y=-0.15, edgecolor='white'
        )

    # add chance lines by disease state
    for disease, data in df.groupby('disease_state'):
        axes[i].axhline(data['chance_proportion_coloc'].mean(),
                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    axes[i].set_ylim(0, 110)
    axes[i].set_ylabel('Colocalised [%]')
    axes[i].set_title(f'{capture} capture', fontsize=8)

# --------Panel D--------
scatbar(
    dataframe=AD_brightness_plotting,
    xcol='capture', ycol='intensity_ratio', ax=axD, xorder=['AT8', 'T181'],
    dotpalette=palette, barpalette=palette,
    hue_col='antibody_state', hue_order=['AT8', 'T181'],
    groups=['AT8', 'T181'], group_label_y=-0.22, group_line_y=-0.15,
    edgecolor='white'
)
for i, capture in enumerate(['AT8', 'T181']):
    for detect in [641, 488]:
        star = brightness_ratio_stats[
            (brightness_ratio_stats['capture'] == capture) & 
            (brightness_ratio_stats['channel'] == detect)]['significance'].tolist()[0]
        j = -0.2 if detect == 641 else 0.2
        axD.annotate(star, xy=(i+j, 2.2),
                    xycoords='data', ha='center')

axD.set_ylabel('Intensity Ratio')
axD.set_ylim(0, 2.6)
axD.axhline(1, linestyle='--', linewidth=1.2, color='#4c4c52')
        
# --------Panel E--------
hexs4 = plot_hexbin(data=filtered_disease, ax=axE, xcol='norm_mean_intensity_641', ycol='norm_mean_intensity_488', vmin=0, vmax=900, colour=cmap, filter_col='capture', filter_val='AT8', kdeplot=True)
              
cb=plt.colorbar(hexs4, ax=axE)
axE.set_title('AT8 capture', fontsize=8)
axE.set(ylabel='Mean intensity 488', xlabel='Mean intensity 641')
axE.set_xlim(0, 9)
axE.set_ylim(0, 3)

# --------Panel F--------
hexs5 = plot_hexbin(data=filtered_disease, ax=axF, xcol='norm_mean_intensity_641', ycol='norm_mean_intensity_488', vmin=0, vmax=900, colour=cmap, filter_col='capture', filter_val='T181', kdeplot=True)
              
axF.set_title('T181 capture', fontsize=8)
axF.set(ylabel='Mean intensity 488', xlabel='Mean intensity 641')
axF.set_xlim(0, 9)
axF.set_ylim(0, 3)
cb=plt.colorbar(hexs5, ax=axF)
cb.set_label('Count', rotation=270, labelpad=15)


# --------Fig admin--------
plt.tight_layout()
plt.savefig(f'{output_folder}Figure4_coloc.svg')


