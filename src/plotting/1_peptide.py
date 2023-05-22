import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from microfilm.microplot import microshow
from skimage.io import imread
from statannotations.Annotator import Annotator

from utils import scatbar

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data/data_path.txt'):
    root_path = open("data/data_path.txt", "r").readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/peptide_data/spots_per_fov.csv'
output_folder = 'results/1_peptide/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
cm = 1/2.54
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

# ===============Organise data===============
# Read in summary FOV data
spots = pd.read_csv(f'{input_path}')
spots.drop([col for col in spots.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# Drop extra wells collected by default due to grid pattern
spots.dropna(subset=['sample'], inplace=True)

# expand sample info
spots[['capture', 'sample', 'detect']
      ] = spots['sample'].str.split('_', expand=True)
spots['spots_count'] = spots['spots_count'].fillna(0)

mean_number_spots = spots.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_number_spots[['sample', 'concentration']
                  ] = mean_number_spots['sample'].str.split('-', expand=True)
mean_number_spots.to_csv(f'{output_folder}mean_number_spots.csv')


spot_count = mean_number_spots[(mean_number_spots['concentration'].isin(
    ['02', '', 'low'])) & (mean_number_spots['capture'] == 'HT7')]

# ======Read in example images======
example_BSA = imread(f'{root_path}data/peptide_images/X6Y1R3W3_641.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)

example_monomer = imread(
    f'{root_path}data/peptide_images/X4Y1R3W2_641.tif')
example_monomer = np.mean(example_monomer[10:, :, :], axis=0)

example_dimer = imread(
    f'{root_path}data/peptide_images/X1Y1R2W2_641.tif')
example_dimer = np.mean(example_dimer[10:, :, :], axis=0)

# # =========Generate figure========
fig = plt.figure(figsize=(18.4 * cm, 2* 6.1 * cm))
gs1 = fig.add_gridspec(nrows=2, ncols=3, wspace=0.35, hspace=0.35)
axA = fig.add_subplot(gs1[0:1, 0:3])
axB = fig.add_subplot(gs1[1:2, 0:1])
axC = fig.add_subplot(gs1[1:2, 1:2])
axD = fig.add_subplot(gs1[1:2, 2:3])

for ax, label in zip([axA, axB, axC, axD], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-40/72, -11/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
axA.axis('off')

# --------Panel B--------
microim1 = microshow(images=[example_dimer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axB,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])
axB.set_title('Dimer', fontsize=8)

# --------Panel C--------
microim1 = microshow(images=[example_monomer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axC,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])
axC.set_title('Monomer', fontsize=8)

# --------Panel D--------
order = ['Dimer', 'Monomer', 'BSA']
pairs = [ ('Monomer', 'BSA'), ('Monomer', 'Dimer')]
scatbar(
    spot_count, xcol='sample', ycol='spots_count', dotcolor='#36454F', barcolor='darkgrey', 
    ax=axD, xorder=order, 
    capsize=0.2, errwidth=2, dotsize=5, 
    pairs=pairs, comparisons_correction='bonferroni'
    )
axD.set_ylim(0, 400)
axD.set_ylabel("Mean spots per FOV")
axD.set_xlabel("")

# --------Fig. elements--------
plt.tight_layout()
plt.savefig(f'{output_folder}Figure1_peptide.svg')
plt.show()
