"""
Generating Figure 2
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from loguru import logger
from microfilm.microplot import microshow

from src.utils import plot_interpolated_ecdf, scatbar

logger.info('Import OK')

# =================Set paths=================
input_path = f'results/2_homogenate_DL/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
cm = 1/2.54
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
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
}

# =========Organise data========
spots_summary = pd.read_csv(f'{input_path}spots_count_summary.csv')
spots_summary = spots_summary[spots_summary['disease_state'] != 'BSA'].copy()
mean_intensity_plotting = pd.read_csv(f'{input_path}mean_intensity.csv')
proportion_intensity_plotting = pd.read_csv(f'{input_path}proportion_intensity.csv')
fitted_ecdf_HT7 = pd.read_csv(f'{input_path}fitted_ecdf_HT7.csv')
fitted_ecdf_HT7.drop([col for col in fitted_ecdf_HT7.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
fitted_ecdf_AT8 = pd.read_csv(f'{input_path}fitted_ecdf_AT8.csv')
fitted_ecdf_AT8.drop([col for col in fitted_ecdf_AT8.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

example_AD = np.load(f'{input_path}example_AD.npy')
example_CRL = np.load(f'{input_path}example_CRL.npy')

# =========Generate figure========
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
                               rescale_type='limits', limits=[400, 700])
axA.set_title('AD', fontsize=8)

# --------Panel B--------
microim1 = microshow(images=[example_CRL],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axB,
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
                               rescale_type='limits', limits=[400, 700])
axB.set_title('CRL', fontsize=8)
    
# --------Panel C--------
scatbar(
    dataframe=spots_summary, xcol='detect', ycol='spots_count', ax=axC, xorder=['AT8', 'HT7'], 
    dotpalette=palette, barpalette=palette,
    hue_col='disease_state', hue_order=['AD', 'CRL'], comparisons_correction=None, pairs=[(('AT8', 'AD'), ('AT8', 'CRL')), (('HT7', 'AD'), ('HT7', 'CRL'))],
    groups=['AT8', 'HT7'], group_label_y=-0.22, group_line_y=-0.15, edgecolor='white')

axC.set_ylabel('Number of spots')
# axC.set_xticks([-0.27, 0, 0.29, 0.73, 1, 1.29])
# axC.set_xticklabels(['AD', 'CRL', 'BSA']*2)

# --------Panel D--------
axD.axis('off')

# --------Panel E--------
scatbar(
    dataframe=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'AT8'], xcol='disease_state', ycol='norm_mean_intensity', ax=axE1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')]
    )
axE1.set_title('AT8', fontsize=8)
axE1.set_ylabel('Number of spots')
axE1.set_xlabel('')

scatbar(
    dataframe=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'HT7'], xcol='disease_state', ycol='norm_mean_intensity', ax=axE2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')]
    )
axE2.set_title('HT7', fontsize=8)
axE2.set_ylabel('')
axE2.set_xlabel('')

# --------Panel F--------
scatbar(
    dataframe=proportion_intensity_plotting, xcol='detect', ycol='bright', ax=axF, xorder=['AT8', 'HT7'],
    dotpalette=palette, barpalette=palette,
    hue_col='disease_state', hue_order=['AD', 'CRL'], comparisons_correction=None, pairs=[(('AT8', 'AD'), ('AT8', 'CRL')), (('HT7', 'AD'), ('HT7', 'CRL'))],
    groups=['AT8', 'HT7'], group_label_y=-0.22, group_line_y=-0.15,edgecolor='white'
    )
axF.set_ylabel('Bright spots (%)')

# --------Panel G--------
plot_interpolated_ecdf(
    fitted_ecdfs = fitted_ecdf_HT7[fitted_ecdf_HT7['sample']!= 'BSA'], ycol='norm_mean_intensity',
    huecol='sample', palette=palette, ax=axG, orientation='h')

# --------Panel H--------
plot_interpolated_ecdf(
    fitted_ecdfs = fitted_ecdf_AT8[fitted_ecdf_AT8['sample']!= 'BSA'], ycol='norm_mean_intensity',
    huecol='sample', palette=palette, ax=axH, orientation='h')

# --------Legend for G,H--------
handles, labels = axH.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9']}

axG.legend(simple_legend.values(), simple_legend.keys(),
               loc='lower right', frameon=False)
axH.legend(simple_legend.values(), simple_legend.keys(),
               loc='lower right', frameon=False)
for ax in fig.axes:
    ax.spines[['right', 'top']].set_visible(False)
    
# --------Fig. admin--------
plt.tight_layout()
plt.savefig(f'{output_folder}Figure2_homogenate_DL.svg')
plt.show()

