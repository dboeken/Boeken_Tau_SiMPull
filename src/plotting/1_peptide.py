import os
import re
from logging import captureWarnings
from random import sample
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from microfilm.microplot import microshow
from microfilm import colorify

from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms




from skimage.io import imread

logger.info('Import OK')

cm = 1/2.54
#plt.subplots(figsize=(18.4 * cm, 6.1 * cm))

if os.path.exists('data/data_path.txt'):
    root_path = open("data/data_path.txt", "r").readlines()[0]
else:
    root_path = ''

# Set paths
input_path = f'{root_path}data/peptide_data/spots_per_fov.csv'
output_folder = 'results/1_peptide/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set default font parameters
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

palette = {
    '':'',
    '':'',
    '':'',
}

# =====================Organise data=====================
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

# =================Read in example images=================

# example_BSA = imread(f'{root_path}data/peptide_images/X6Y1R3W3_641.tif')
# example_BSA = np.mean(example_BSA[10:, :, :], axis=0)
# example_monomer = imread(f'{root_path}data/peptide_images/X4Y1R3W2_641.tif')
# example_monomer = np.mean(example_monomer[10:, :, :], axis=0)
# example_dimer = imread(f'{root_path}data/peptide_images/X1Y1R2W2_641.tif')
# example_dimer = np.mean(example_dimer[10:, :, :], axis=0)


example_BSA = imread('/Users/dorotheaboeken/Documents/GitHub/Boeken_Tau_SiMPull/data/peptide_data/example_BSA.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)
plt.imshow(example_BSA)

example_monomer = imread(
    '/Users/dorotheaboeken/Documents/GitHub/Boeken_Tau_SiMPull/data/peptide_data/example_monomer.tif')
example_monomer = np.mean(example_monomer[10:, :, :], axis=0)
plt.imshow(example_monomer)


example_dimer = imread(
    '/Users/dorotheaboeken/Documents/GitHub/Boeken_Tau_SiMPull/data/peptide_data/example_dimer.tif')
example_dimer = np.mean(example_dimer[10:, :, :], axis=0)
plt.imshow(example_dimer)


# ==================Generate main figure==================

fig, axes = plt.subplots(1, 4, figsize=(12, 5), gridspec_kw={'width_ratios': [6, 6, 6, 5]})

#Panel D: Example BSA image
microim1 = microshow(images=[example_BSA],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axes[0],
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)
microim1 = microshow(images=[example_monomer],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axes[2],
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)
microim1 = microshow(images=[example_dimer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axes[1],
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,)

# Panel E: Spot count
df = mean_number_spots[(mean_number_spots['concentration'].isin(['02', '', 'low'])) & (mean_number_spots['capture'] == 'HT7')]
sns.stripplot(
    data=df, 
    x='sample', 
    y='spots_count', 
    color='#36454F',
    s=10
)
sns.barplot(
    data=df,
    x='sample',
    y='spots_count',
    capsize=0.2,
    errwidth=2,
    color='darkgrey'
)
axes[2].set_ylim(0, 400)
axes[2].set_ylabel("Mean spots per FOV")
axes[2].set_xlabel("")


axes[0].annotate('B', xy=(0, 1.05), xycoords='axes fraction', size=24, weight='bold')
axes[1].annotate('C', xy=(0, 1.05), xycoords='axes fraction', size=24, weight='bold')
axes[2].annotate('D', xy=(-0.4, 0.925), xycoords='axes fraction', size=24, weight='bold')

plt.tight_layout()
#plt.savefig(f'{output_folder}1_peptide.svg')
plt.show()


########plot AT8 peptide data for supplementals #########

# sns.set(rc={'figure.figsize': (3, 4)})
# sns.set_theme(style="ticks", font_scale=1.4)
for (capture), df in mean_number_spots[mean_number_spots['capture'].isin(['AT8'])].groupby(['capture']):
    sns.stripplot(data=df, x='sample', y='spots_count', color='#36454F',
                  s=10)
    # plt.xlim(0, 6000)

    sns.barplot(
        data=df,
        x='sample',
        y='spots_count',
        capsize=0.2,
        errwidth=2,
        color='darkgrey'
    )

    plt.ylim(0, 500)
    #plt.ylabel("mean spots")

    plt.title(f'{capture}')
    #plt.show()
    plt.savefig(f'{output_folder}mean_number_spots_AT8.svg')







def scatbarplot(ycol, ylabel, ax, data):
    order = ['Dimer', 'Monomer', 'BSA']
    sns.barplot(
        data=data,
        x='sample',
        y=ycol,
        #hue='disease_state',
        #palette=palette,
        color='darkgrey',
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=False,
        order=order,
    )
    sns.stripplot(
        data=data,
        x='sample',
        y=ycol,
        #hue='disease_state',
        #palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        color='#36454F',
        s=10,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [ ('Monomer', 'BSA'), ('Monomer', 'Dimer')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='sample', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside', comparisons_correction='bonferroni')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)



fig = plt.figure(figsize=(18.4 * cm, 2* 6.1 * cm))
gs1 = fig.add_gridspec(nrows=2, ncols=3, wspace=0.35, hspace=0.35)
axA = fig.add_subplot(gs1[0:1, 0:3])
axB = fig.add_subplot(gs1[1:2, 0:1])
axC = fig.add_subplot(gs1[1:2, 1:2])
axD = fig.add_subplot(gs1[1:2, 2:3])


# axE = fig.add_subplot(gs1[4:6, 0:2])
# axF = fig.add_subplot(gs1[4:6, 2:4])
# axG = fig.add_subplot(gs1[4:6, 4:6])


for ax, label in zip([axA, axB, axC, axD], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-40/72, -11/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
axA.axis('off')
# --------Panel B--------


microim1 = microshow(images=[example_monomer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axC,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])
microim1 = microshow(images=[example_dimer],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axB,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])


df = mean_number_spots[(mean_number_spots['concentration'].isin(
    ['02', '', 'low'])) & (mean_number_spots['capture'] == 'HT7')]
# sns.stripplot(
#     data=df,
#     x='sample',
#     y='spots_count',
#     color='#36454F',
#     s=10
# )
# sns.barplot(
#     data=df,
#     x='sample',
#     y='spots_count',
#     capsize=0.2,
#     errwidth=2,
#     color='darkgrey'
# )

scatbarplot('spots_count', 'Mean spots', axD, df)

axD.set_ylim(0, 400)
axD.set_ylabel("Mean spots per FOV")
axD.set_xlabel("")


# scatbarplot('smoothed_length', 'Mean length [nm]',
#             palette, axB1, for_plotting_mean)
axB.set_title('Dimer', fontsize=8)
axC.set_title('Monomer', fontsize=8)

# scatbarplot('scaled_perimeter', 'Mean perimeter [nm]',
#             palette, axB2, for_plotting_mean)
# axB2.set_title('Perimeter')


plt.tight_layout()
plt.savefig(f'{output_folder}Figure1_peptide.svg')
plt.show()
