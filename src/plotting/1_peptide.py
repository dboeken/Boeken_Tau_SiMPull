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



# ########plot AT8 peptide data for supplementals #########

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

    plt.savefig(f'{output_folder}mean_number_spots_AT8.svg')


# # ==================Generate main figure==================


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
        s=5,
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


scatbarplot('spots_count', 'Mean spots', axD, df)

axD.set_ylim(0, 400)
axD.set_ylabel("Mean spots per FOV")
axD.set_xlabel("")


axB.set_title('Dimer', fontsize=8)
axC.set_title('Monomer', fontsize=8)


plt.tight_layout()
plt.savefig(f'{output_folder}Figure1_peptide.svg')
plt.show()
