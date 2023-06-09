"""
Generating Figure S2
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import ListedColormap
from statannotations.Annotator import Annotator

from src.utils import plot_hexbin, scatbar

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
p1byp2_for_plotting = pd.read_csv(
    f'{input_folder}p1byp2_for_plotting_all.csv')
p1byp2_for_plotting.drop(
    [col for col in p1byp2_for_plotting.columns if 'Unnamed' in col], axis=1, inplace=True)
p1byp2_for_plotting['sample'] = p1byp2_for_plotting['sample'].astype(
    str)
p1byp2_for_plotting = {category: df.dropna(
    how='all', axis=1) for category, df in p1byp2_for_plotting.groupby('category')}

# =================Plot figure=================

new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)


fig = plt.figure(figsize=(18.4 * cm, 4 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=4, ncols=4, wspace=0.9, hspace=0.4)
axB = fig.add_subplot(gs1[0:1, 0:2])
axC = fig.add_subplot(gs1[0:1, 2:4])
axD1 = fig.add_subplot(gs1[1:2, 0:1])
axD2 = fig.add_subplot(gs1[1:2, 1:2])
axE1 = fig.add_subplot(gs1[1:2, 2:3])
axE2 = fig.add_subplot(gs1[1:2, 3:4])

axF1 = fig.add_subplot(gs1[2:3, 0:1])
axF2 = fig.add_subplot(gs1[2:3, 1:2])
axF3 = fig.add_subplot(gs1[2:3, 2:3])
axF4 = fig.add_subplot(gs1[2:3, 3:4])

axG1 = fig.add_subplot(gs1[3:4, 0:1])
axG2 = fig.add_subplot(gs1[3:4, 1:2])
axG3 = fig.add_subplot(gs1[3:4, 2:3])
axG4 = fig.add_subplot(gs1[3:4, 3:4])


for ax, label in zip([axB, axC, axD1, axE1, axF1, axG1,], ['B', 'C', 'D', 'E', 'F', 'G',]):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel B--------
hexs0 = plot_hexbin(data=for_plotting, ax=axB, xcol='eccentricity', ycol='smoothed_length', vmin=0, vmax=1200, colour=cmap, filter_col='disease_state', filter_val='AD', kdeplot=None)

cb = plt.colorbar(hexs0, ax=axB)

axB.set_title('AD', fontsize=8)
axB.set_xlim(0, 1)
axB.set_ylim(0, 550)
axB.set(ylabel='Length [nm]')
axB.set(xlabel='Eccentricity')

# --------Panel C--------
hexs1 = plot_hexbin(data=for_plotting, ax=axC, xcol='eccentricity', ycol='smoothed_length',
                    vmin=0, vmax=1200, colour=cmap, filter_col='disease_state', filter_val='CRL', kdeplot=None)
cb = plt.colorbar(hexs1, ax=axC)
cb.set_label('Count', rotation=270, labelpad=15)
axC.set_title('CRL', fontsize=8)
axC.set_xlim(0, 1)
axC.set_ylim(0, 550)
axC.set(ylabel='Length [nm]')
axC.set(xlabel='Eccentricity')


# --------Panel D--------
scatbar(
    dataframe=p1byp2_for_plotting['eccentricity_smoothed_length'][p1byp2_for_plotting['eccentricity_smoothed_length']['disease_state'] == 'AD'], xcol='smoothed_length_cat', ycol='label', ax=axD1, xorder=['high', 'low'],
    dotpalette=palette, barpalette=palette,
    pairs=[('high', 'low')],
)
axD1.set_xticks([0, 1])
axD1.set_xticklabels(['Long', 'Short'])
axD1.set(title='All AD aggs.', ylabel='Fibrils [%]', xlabel='')


scatbar(
    dataframe=p1byp2_for_plotting['eccentricity_smoothed_length'][p1byp2_for_plotting['eccentricity_smoothed_length']['disease_state'] == 'CRL'], xcol='smoothed_length_cat', ycol='label', ax=axD2, xorder=['high', 'low'],
    dotpalette={'high': '#345995', 'low': '#345995'}, barpalette={'high': '#345995', 'low': '#345995'},
    pairs=[('high', 'low')],
)
axD2.set_xticks([0, 1])
axD2.set_xticklabels(['Long', 'Short'])
axD2.set(title='All CRL aggs.', ylabel='Fibrils [%]', xlabel='')

# --------Panel E--------
scatbar(
    dataframe=p1byp2_for_plotting['smoothed_length_eccentricity'][p1byp2_for_plotting['smoothed_length_eccentricity']['eccentricity_cat'] == 'low'], xcol='disease_state', ycol='label', ax=axE1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE1.set(title='Round aggs.', ylabel='Long [%]', xlabel='')

# scatbar(
#     dataframe=for_plotting_mean, xcol='disease_state', ycol='#locs', ax=axC3, xorder=['AD', 'CRL'],
#     dotpalette=palette, barpalette=palette,
#     pairs=[('AD', 'CRL')],
# )
# axC3.set(title='# locs',
#          ylabel='Mean number of \n localisations per cluster', xlabel='')

scatbar(
    dataframe=p1byp2_for_plotting['smoothed_length_eccentricity'][p1byp2_for_plotting['smoothed_length_eccentricity']['eccentricity_cat'] == 'high'], xcol='disease_state', ycol='label', ax=axE2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axE2.set(title='Fibrilar aggs.', ylabel='Long [%]', xlabel='')

# scatbar(
#     dataframe=for_plotting_mean, xcol='disease_state', ycol='#locs_density', ax=axC4, xorder=['AD', 'CRL'],
#     dotpalette=palette, barpalette=palette,
#     pairs=[('AD', 'CRL')],
# )
# axC4.set(title='Locs density',
#          ylabel='Localisation density \n localisations per 10$^3$ nm$^2$', xlabel='')




# --------Panel F--------
scatbar(
    dataframe=fil_proportion['smoothed_length'], xcol='disease_state', ycol='high', ax=axF1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axF1.set(title='Length\n(>250 nm)', ylabel='Long [%]', xlabel='')


scatbar(
    dataframe=fil_proportion['scaled_perimeter'], xcol='disease_state', ycol='high', ax=axF2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axF2.set(title='Perimeter\n(>550 nm)',
         ylabel='Long perimeter [%]', xlabel='')


scatbar(
    dataframe=fil_proportion['scaled_area'], xcol='disease_state', ycol='high', ax=axF3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axF3.set(title='Area\n(>15x10$^3$nm$^2$)', ylabel='Large [%]', xlabel='')

scatbar(
    dataframe=fil_proportion['eccentricity'], xcol='disease_state', ycol='high', ax=axF4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axF4.set(title='Eccentricity\n(>0.9)', ylabel='Fibrils [%]', xlabel='')

# --------Panel G--------
scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='smoothed_length', ax=axG1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axG1.set(title='Length', ylabel='Mean length [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='scaled_perimeter', ax=axG2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axG2.set(title='Perimeter', ylabel='Mean perimeter [nm]', xlabel='')


scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='scaled_area', ax=axG3, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axG3.set(title='Area', ylabel='Mean area [x 10$^3$ nm$^2$]', xlabel='')

scatbar(
    dataframe=for_plotting_fil_mean, xcol='disease_state', ycol='eccentricity', ax=axG4, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')],
)
axG4.set(title='Eccentricity', ylabel='Mean eccentricity', xlabel='')


for ax in fig.axes:
    ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_folder}S2_homogenate_SR.svg')
plt.show()
