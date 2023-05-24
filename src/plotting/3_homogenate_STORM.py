"""
Generating Figure 3
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns
from loguru import logger
from statannotations.Annotator import Annotator

from src.utils import plot_interpolated_ecdf, scatbar

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
        ci = 'sd'
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


def hexbinplotting(colour, ax, data, disease_state):

    df = data[data['disease_state'] == disease_state].copy()
    hexs = ax.hexbin(data=df, x='eccentricity',
                     y='smoothed_length', cmap=colour, vmin=0, vmax=1200)
    ax.set(ylabel='Length [nm]')
    ax.set(xlabel='Eccentricity')
    # sns.kdeplot(data=df, x='smoothed_length', y='eccentricity',
    #             color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 550)
    

    return hexs


def plot_hexbins(dataframe, colour, ax, xcol='eccentricity', ycol='smoothed_length', vmin=0, vmax=1200):

    hexs = ax.hexbin(data=dataframe, x=xcol,
                     y=ycol, cmap=colour, vmin=vmin, vmax=vmax)
    ax.set(ylabel='Length [nm]')
    ax.set(xlabel='Eccentricity')
    # sns.kdeplot(data=df, x='smoothed_length', y='eccentricity',
    #             color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 550)

    return hexs


# =================Organise data=================
for_plotting = pd.read_csv(f'{input_folder}for_plotting.csv')

proportions = pd.read_csv(f'{input_folder}proportions.csv')
proportions.drop([col for col in proportions.columns if 'Unnamed' in col], axis=1, inplace=True)
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
fitted_ecdfs.drop([col for col in fitted_ecdfs.columns if 'Unnamed' in col], axis=1, inplace=True)
fitted_ecdfs['sample'] = fitted_ecdfs['sample'].astype(str)
fitted_ecdfs = {category: df.dropna(how='all', axis=1)
                for category, df in fitted_ecdfs.groupby('category')}

for_plotting_fil_mean = pd.read_csv(f'{input_folder}for_plotting_fil_mean.csv')
parameter_by_parameter2_for_plotting = pd.read_csv(
    f'{input_folder}parameter_by_parameter2_for_plotting_all.csv')
parameter_by_parameter2_for_plotting.drop(
    [col for col in parameter_by_parameter2_for_plotting.columns if 'Unnamed' in col], axis=1, inplace=True)
parameter_by_parameter2_for_plotting['sample'] = parameter_by_parameter2_for_plotting['sample'].astype(str)
parameter_by_parameter2_for_plotting = {category: df.dropna(how='all', axis=1) for category, df in parameter_by_parameter2_for_plotting.groupby('category')}

# =================Plot figure=================

fig = plt.figure(figsize=(12.1 * cm, 4 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=8, ncols=4, wspace=0.95, hspace=1.25)
axA1 = fig.add_subplot(gs1[0:1, 0:1])
axA2 = fig.add_subplot(gs1[0:1, 1:2])
axA3 = fig.add_subplot(gs1[1:2, 0:1])
axA4 = fig.add_subplot(gs1[1:2, 1:2])
axA5 = fig.add_subplot(gs1[0:1, 2:3])
axA6 = fig.add_subplot(gs1[1:2, 2:3])
axA7 = fig.add_subplot(gs1[0:1, 3:4])
axA8 = fig.add_subplot(gs1[1:2, 3:4])

axB1 = fig.add_subplot(gs1[2:4, 0:1])
axB2 = fig.add_subplot(gs1[2:4, 1:2])
axB3 = fig.add_subplot(gs1[2:4, 2:3])
axB4 = fig.add_subplot(gs1[2:4, 3:4])

axC1 = fig.add_subplot(gs1[4:6, 0:2])
axC2 = fig.add_subplot(gs1[4:6, 2:4])

axD1 = fig.add_subplot(gs1[6:8, 0:1])
axD2 = fig.add_subplot(gs1[6:8, 1:2])
axD3 = fig.add_subplot(gs1[6:8, 2:3])
axD4 = fig.add_subplot(gs1[6:8, 3:4])

for ax, label in zip([axA1, axB1, axC1, axD1], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
axA1.axis('off')
axA2.axis('off')
axA3.axis('off')
axA4.axis('off')
axA5.axis('off')
axA6.axis('off')
axA7.axis('off')
axA8.axis('off')

# --------Panel B--------
scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='smoothed_length', ax=axB1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')], 
    )
axB1.set(title='Length', ylabel='Mean length [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='scaled_perimeter', ax=axB2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axB2.set(title='Perimeter', ylabel='Mean perimeter [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='scaled_area', ax=axB3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axB3.set(title='Area', ylabel='Mean area [x 10$^3$ nm$^2$]', xlabel='')

scatbar(
    dataframe=for_plotting_mean, xcol='disease_state', ycol='eccentricity', ax=axB4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axB4.set(title='Eccentricity', ylabel='Mean eccentricity', xlabel='')


# --------Panel C--------

plot_interpolated_ecdf(
    fitted_ecdfs['smoothed_length'], ycol='smoothed_length', huecol='sample', palette=palette, ax=axC1, orientation='h')

plot_interpolated_ecdf(fitted_ecdfs['eccentricity'], ycol='eccentricity', huecol='sample', palette=palette, ax=axC2, orientation='h')

# --------Panel D--------

scatbar(
    dataframe=proportion['smoothed_length'], xcol='disease_state', ycol='high', ax=axD1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD1.set(title='Length', ylabel='Long [%]', xlabel='')


scatbar(
    dataframe=proportion['scaled_perimeter'], xcol='disease_state', ycol='high', ax=axD2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD2.set(title='Perimeter', ylabel='Long perimeter [%]', xlabel='')


scatbar(
    dataframe=proportion['scaled_area'], xcol='disease_state', ycol='high', ax=axD3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD3.set(title='Area', ylabel='Large [%]', xlabel='')

scatbar(
    dataframe=proportion['eccentricity'], xcol='disease_state', ycol='high', ax=axD4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axD4.set(title='Eccentricity', ylabel='Fibrils [%]', xlabel='')



handles, labels = axC1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9']}
for label, ax in zip(['Length (nm)', 'Eccentricity'], [axC1, axC2]):
    ax.legend(simple_legend.values(), simple_legend.keys(), frameon=False)
    ax.set_xlabel(label)
    ax.set_ylabel('Proportion')

for ax in fig.axes:
    ax.spines[['right', 'top']].set_visible(False)
    

plt.tight_layout()
plt.savefig(f'{output_folder}Figure3_homogenate_SR.svg')
plt.show()


