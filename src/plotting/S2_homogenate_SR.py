"""
Generating Figure S2
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
from statannotations.Annotator import Annotator

from src.utils import plot_hexbin, plot_interpolated_ecdf, scatbar

logger.info('Import OK')


# =================Set paths=================
input_folder = 'results/3_homogenate_SR/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
cm = 1/2.54
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8,
        }
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300

palette = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#F03A47',
    '55': '#F03A47',
    '246': '#F03A47',
    'BSA': '#A9A9A9',
    'AD Mix': '#A9A9A9',
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
    'high': '#F03A47',
    'low': '#F03A47',
}

# =================Defining functions=================


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
        ci='sd'
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
        s=5,
        order=order,

    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def scatbarplot2(ycol, ylabel, palette, ax, data):
    order = ['high', 'low']
    sns.barplot(
        data=data,
        x='smoothed_length_cat',
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
        x='smoothed_length_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('high', 'low')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='smoothed_length_cat', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.set_xticklabels(['Long', 'Short'])
    ax.legend('', frameon=False)


def scatbarplot_hue_ecc(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
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
        s=5,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    pairs = [(('fibril', 'AD'), ('round', 'AD')),
             (('fibril', 'CRL'), ('round', 'CRL')), (('fibril', 'AD'), ('fibril', 'CRL')), (('round', 'AD'), ('round', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='ecc_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)
    ax.legend('', frameon=False)


def scatbarplot_hue_two_param(ycol, ylabel, palette, ax, data, xcol, high, low, group_label_y=-0.18, group_line_y=-0.05):
    order = ['high', 'low']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x=xcol,
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
        x=xcol,
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    pairs = [(('high', 'AD'), ('low', 'AD')),
             (('high', 'CRL'), ('low', 'CRL')), (('high', 'AD'), ('high', 'CRL')), (('low', 'AD'), ('low', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x=xcol, y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([high, low])

    ax.legend('', frameon=False)



# =================Organise data=================
for_plotting = pd.read_csv(f'{input_folder}for_plotting.csv')

proportions = pd.read_csv(f'{input_folder}proportions.csv')
proportions.drop(
    [col for col in proportions.columns if 'Unnamed' in col], axis=1, inplace=True)
proportions['sample'] = proportions['sample'].astype(str)
proportion = {category: df.dropna(how='all', axis=1)
              for category, df in proportions.groupby('category')}

fil_proportion = pd.read_csv(f'{input_folder}fil_proportions.csv')
fil_proportion.drop(
    [col for col in fil_proportion.columns if 'Unnamed' in col], axis=1, inplace=True)
fil_proportion['sample'] = fil_proportion['sample'].astype(str)
fil_proportion = {category: df.dropna(how='all', axis=1)
                  for category, df in fil_proportion.groupby('category')}

for_plotting_mean = pd.read_csv(f'{input_folder}for_plotting_mean.csv')

fitted_ecdfs = pd.read_csv(f'{input_folder}fitted_ecdfs.csv')
fitted_ecdfs.drop(
    [col for col in fitted_ecdfs.columns if 'Unnamed' in col], axis=1, inplace=True)
fitted_ecdfs['sample'] = fitted_ecdfs['sample'].astype(str)
fitted_ecdfs = {category: df.dropna(how='all', axis=1)
                for category, df in fitted_ecdfs.groupby('category')}

for_plotting_fil_mean = pd.read_csv(f'{input_folder}for_plotting_fil_mean.csv')
parameter_by_parameter2_for_plotting = pd.read_csv(
    f'{input_folder}parameter_by_parameter2_for_plotting_all.csv')
parameter_by_parameter2_for_plotting.drop(
    [col for col in parameter_by_parameter2_for_plotting.columns if 'Unnamed' in col], axis=1, inplace=True)
parameter_by_parameter2_for_plotting['sample'] = parameter_by_parameter2_for_plotting['sample'].astype(
    str)
parameter_by_parameter2_for_plotting = {category: df.dropna(
    how='all', axis=1) for category, df in parameter_by_parameter2_for_plotting.groupby('category')}

# =================Plot figure=================



new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)


fig = plt.figure(figsize=(18.4 * cm, 4 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=4, ncols=4, wspace=0.9, hspace=0.4)
axA = fig.add_subplot(gs1[0:1, 0:2])
axB = fig.add_subplot(gs1[0:1, 2:4])
axC1 = fig.add_subplot(gs1[1:2, 0:1])
axC2 = fig.add_subplot(gs1[1:2, 1:2])
axC3 = fig.add_subplot(gs1[1:2, 2:3])
axC4 = fig.add_subplot(gs1[1:2, 3:4])

axD1 = fig.add_subplot(gs1[2:3, 0:1])
axD2 = fig.add_subplot(gs1[2:3, 1:2])
axD3 = fig.add_subplot(gs1[2:3, 2:3])
axD4 = fig.add_subplot(gs1[2:3, 3:4])

axE1 = fig.add_subplot(gs1[3:4, 0:1])
axE2 = fig.add_subplot(gs1[3:4, 1:2])
axE3 = fig.add_subplot(gs1[3:4, 2:3])
axE4 = fig.add_subplot(gs1[3:4, 3:4])


for ax, label in zip([axA, axB, axC1, axC2, axC3, axC4, axD1, axE1], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
hexs0 = plot_hexbin(data=for_plotting, ax=axA, xcol='eccentricity', ycol='smoothed_length', vmin=0, vmax=1200, colour=cmap, filter_col='disease_state', filter_val='AD', kdeplot=None)

cb = plt.colorbar(hexs0, ax=axA)

axA.set_title('AD', fontsize=8)
axA.set_xlim(0, 1)
axA.set_ylim(0, 550)
axA.set(ylabel='Length [nm]')
axA.set(xlabel='Eccentricity')

hexs1 = plot_hexbin(data=for_plotting, ax=axB, xcol='eccentricity', ycol='smoothed_length',
                    vmin=0, vmax=1200, colour=cmap, filter_col='disease_state', filter_val='CRL', kdeplot=None)
cb = plt.colorbar(hexs1, ax=axB)
cb.set_label('Count', rotation=270, labelpad=15)
axB.set_title('CRL', fontsize=8)
axB.set_xlim(0, 1)
axB.set_ylim(0, 550)
axB.set(ylabel='Length [nm]')
axB.set(xlabel='Eccentricity')


# --------Panel C--------
scatbar(
    dataframe=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'][parameter_by_parameter2_for_plotting['smoothed_length_eccentricity']['eccentricity_cat'] == 'low'], xcol='disease_state', ycol='label', ax=axC1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axC1.set(title='Round aggs.', ylabel='Long [%]', xlabel='')


scatbar(
    dataframe=parameter_by_parameter2_for_plotting['eccentricity_smoothed_length'][parameter_by_parameter2_for_plotting['eccentricity_smoothed_length']['disease_state'] == 'AD'], xcol='smoothed_length_cat', ycol='label', ax=axC2, xorder=['high', 'low'],
    dotpalette=palette, barpalette=palette,
    pairs=[('high', 'low')],
)
axC2.set_xticks([0, 1])
axC2.set_xticklabels(['Long', 'Short'])
axC2.set(title='All AD aggs.', ylabel='Fibrils [%]', xlabel='')

scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='#locs', ax=axC3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axC3.set(title='# locs',
         ylabel='Mean number of \n localisations per cluster', xlabel='')


scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='#locs_density', ax=axC4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axC4.set(title='Locs density',
         ylabel='Localisation density \n localisations per 10$^3$ nm$^2$', xlabel='')


# --------Panel D--------


scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='smoothed_length', ax=axD1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD1.set(title='Length', ylabel='Mean length [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='scaled_perimeter', ax=axD2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD2.set(title='Perimeter', ylabel='Mean perimeter [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='scaled_area', ax=axD3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD3.set(title='Area', ylabel='Mean area [x 10$^3$ nm$^2$]', xlabel='')

scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='eccentricity', ax=axD4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD4.set(title='Eccentricity', ylabel='Mean eccentricity', xlabel='')


# --------Panel E--------

scatbar(
    dataframe=fil_proportion['smoothed_length'], xcol='disease_state', ycol='high', ax=axE1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE1.set(title='Length', ylabel='Long [%]', xlabel='')


scatbar(
    dataframe=fil_proportion['scaled_perimeter'], xcol='disease_state', ycol='high', ax=axE2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE2.set(title='Perimeter', ylabel='Long perimeter [%]', xlabel='')


scatbar(
    dataframe=fil_proportion['scaled_area'], xcol='disease_state', ycol='high', ax=axE3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE3.set(title='Area', ylabel='Large [%]', xlabel='')

scatbar(
    dataframe=fil_proportion['eccentricity'], xcol='disease_state', ycol='high', ax=axE4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE4.set(title='Eccentricity', ylabel='Fibrils [%]', xlabel='')


plt.tight_layout()
plt.savefig(f'{output_folder}S2_homogenate_SR.svg')
plt.show()
