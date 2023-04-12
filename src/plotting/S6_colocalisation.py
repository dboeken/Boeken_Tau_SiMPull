import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms
from skimage.io import imread
from microfilm.microplot import microshow

from loguru import logger

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_spots = f'{root_path}data/colocalisation_data/colocalisation_spots.csv'
input_image_r = f'{root_path}data/homogenate_DL_images/X8Y2R2W2_641.tif'
input_image_g = f'{root_path}data/homogenate_DL_images/X8Y2R2W2_641.tif'
output_folder = f'{root_path}results/S6_colocalisation/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
import matplotlib
font = {'family' : 'arial',
'weight' : 'normal',
'size'   : 8 }
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

cm = 1/2.54

# Read in raw colocalisation data
spots = pd.read_csv(input_spots)
spots.drop([col for col in spots.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# filter to image of interest
colocalised = spots[
    (spots['layout'] == 1) &
    (spots['well_info'].str.contains('X0Y0R1W2'))
    ].copy()


# Read example images, prepare max projection
red_img = imread(input_image_r)
red_proj = np.mean(red_img[10:, :, :], axis=0)

grn_img = imread(input_image_g)
grn_proj = np.mean(grn_img[10:, :, :], axis=0)
# plt.imshow(grn_proj)


# ==================Supplementary Figure==================

# Generate figure panels
fig, axes = plt.subplots(1, 3, figsize=(18.4 * cm, 6.1 * cm))
axes = axes.ravel()

# Add panel labels
for x, label in enumerate(['A', 'B', 'C']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-0.6, 0., fig.dpi_scale_trans)
    axes[x].text(0.0, 1.0, label, transform=axes[x].transAxes + trans,
                 fontsize=12, va='bottom', fontweight='bold')

# Add image panels
microim1 = microshow(
    images=[red_proj], 
    cmaps=['pure_magenta'], #flip_map=[True],
    label_color='black', ax=axes[0], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000]])

microim1 = microshow(
    images=[grn_proj], 
    cmaps=['pure_green'], #flip_map=[True],
    label_color='black', ax=axes[1], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000]])

microim1 = microshow(
    images=[red_proj, grn_proj],
    cmaps=['pure_magenta', 'pure_green'],  # flip_map=[True],
    label_color='black', ax=axes[2], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000], [0, 2000]])

# Add spots for individual channels
for i, channel in enumerate([641, 488]):
    
    df = colocalised[colocalised['channel'] == channel].copy()

    sns.scatterplot(
        data=df[~df['pair_id'].isnull()],
        x='centroid-0',
        y='centroid-1',
        marker='o',
        s=10,
        ax=axes[i],
        facecolor='None',
        edgecolor='#fff',
        linewidth=0.1
    )
    sns.scatterplot(
        data=df[df['pair_id'].isnull()],
        x='centroid-0',
        y='centroid-1',
        marker='x',
        s=10,
        ax=axes[i],
        color='white',
        linewidth=0.1
    )
    # axes[i].set_xlim(0, 512)
    # axes[i].set_ylim(0, 512)

# Add colocalised spots to overlay
df = colocalised[colocalised['channel'] == 641].copy().dropna(subset=['pair_id'])

sns.scatterplot(
    data=df,
    x='centroid-0',
    y='centroid-1',
    marker='o',
    s=10,
    ax=axes[2],
    facecolor='None',
    edgecolor='#fff',
    linewidth=0.1
)

# Add channel/detection labels to each panel
axes[0].set_title('AT8')
axes[1].set_title('T181')
axes[2].set_title('Overlay')


# Save final figure
plt.savefig(f'{output_folder}S6_colocalisation.svg')