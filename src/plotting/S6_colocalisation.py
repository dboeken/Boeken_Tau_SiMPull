import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms
from skimage.io import imread
from microfilm.microplot import microshow
from scipy.spatial import distance

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


def closest_node(node, nodes, threshold=False, verbose=False):
    min_distance = distance.cdist([node], nodes).min()
    if threshold and not min_distance < threshold:
        if verbose:
            logger.info(f'No node closer than {threshold} found for {node}')
        return (np.nan, np.nan)
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def find_pairs(seed_nodes, test_nodes, threshold=False):
    # Find closest points
    pairs = pd.DataFrame(seed_nodes, columns=['seed_coord_0', 'seed_coord_1'])
    pairs[['test_coord_0', 'test_coord_1']] = np.array(
        [closest_node(point, test_nodes, threshold=threshold) for point in seed_nodes])

    # Add distances
    pairs['distance'] = [np.nan if np.isnan(tx) else distance.euclidean(
        (sx, sy), (tx, ty)) for sx, sy, tx, ty in pairs.values]

    # Limit to a single point (with min distance) per seed node
    min_distances = pairs.groupby(['test_coord_0', 'test_coord_1'])[
        'distance'].idxmin().tolist()
    pairs[['test_coord_0', 'test_coord_1']] = [[tx, ty] if index in min_distances else [np.nan, np.nan]
                                               for index, (tx, ty) in zip(pairs.index, pairs[['test_coord_0', 'test_coord_1']].values)]

    return pairs
    

def plot_colocalisation(seed_spots, test_spots, threshold=False, ax=None):

    pairs = find_pairs(seed_nodes=seed_spots[['centroid-0', 'centroid-1']].values,
                       test_nodes=test_spots[['centroid-0', 'centroid-1']].values, threshold=threshold)
    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5))
    # Visualise spots with lines connecting pairs
    sns.scatterplot(
        data=seed_spots,
        x='centroid-0',
        y='centroid-1',
        color='magenta',
        alpha=0.3,
        size=5,
        ax=ax
    )
    sns.scatterplot(
        data=test_spots,
        x='centroid-0',
        y='centroid-1',
        color='forestgreen',
        marker='x',
        size=2,
        ax=ax
    )

    for x1, y1, x2, y2, dist_pair in pairs.dropna(subset=['test_coord_1']).values:
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=0.1)

    sns.scatterplot(
        data=pairs.dropna(subset=['test_coord_1']),
        x='seed_coord_0',
        y='seed_coord_1',
        color='firebrick',
        ec="black", fc="none",
        s=50,
        ax=ax
    )

    return pairs


# Read in raw colocalisation data
spots = pd.read_csv(input_spots)
spots.drop([col for col in spots.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# filter to image of interest
colocalised = spots[
    (spots['layout'] == 1) &
    (spots['well_info'].str.contains('X0Y0R1W2'))
    ].copy()
colocalised['pair_id'].unique()

# Calculate threshold optimisation
optimisation = []
for threshold in range(1, 11):
    threshold
    red_spots = colocalised[colocalised['channel'] == 641].copy()
    green_spots = colocalised[colocalised['channel'] == 488].copy()
    pairs = find_pairs(seed_nodes=red_spots[['centroid-0', 'centroid-1']].values,
                       test_nodes=green_spots[['centroid-0', 'centroid-1']].values, threshold=threshold)
    coloc = len(pairs.dropna(subset=['test_coord_0']))
    
    green_spots['centroid-1'] = 512 - green_spots['centroid-1']
    pairs = find_pairs(seed_nodes=red_spots[['centroid-0', 'centroid-1']].values,
                       test_nodes=green_spots[['centroid-0', 'centroid-1']].values, threshold=threshold)
    chance = len(pairs.dropna(subset=['test_coord_0']))
    
    optimisation.append([threshold, coloc, chance])
optimisation = pd.DataFrame(optimisation, columns=['threshold', 'coloc', 'chance'])

# Read example images, prepare max projection
red_img = imread(input_image_r)
red_proj = np.mean(red_img[10:, :, :], axis=0)

grn_img = imread(input_image_g)
grn_proj = np.mean(grn_img[10:, :, :], axis=0)
# plt.imshow(grn_proj)


# ==================Supplementary Figure==================

# Generate figure panels
fig, axes = plt.subplots(2, 3, figsize=(18.4 * cm, 2 * 6.1 * cm))
axes = axes.ravel()

# Add panel labels
for x, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-0.25, 0., fig.dpi_scale_trans)
    axes[x].text(0.0, 1.0, label, transform=axes[x].transAxes + trans,
                 fontsize=12, va='bottom', fontweight='bold')

# -------Add graph panels-------
# visualise colocalised spots with distance
red_spots = colocalised[colocalised['channel'] == 641].copy()
green_spots = colocalised[colocalised['channel'] == 488].copy()
plot_colocalisation(seed_spots=red_spots, test_spots=green_spots, threshold=4, ax=axes[0])
axes[0].legend('', frameon=False)
axes[0].set_ylim(512, 0)
axes[0].axis('off')

# visualise chance colocalised spots with distance
red_spots = colocalised[colocalised['channel'] == 641].copy()
green_spots = colocalised[colocalised['channel'] == 488].copy()
green_spots['centroid-1'] = 512 - green_spots['centroid-1']
plot_colocalisation(seed_spots=red_spots, test_spots=green_spots, threshold=4, ax=axes[1])
axes[1].legend('', frameon=False)
axes[1].set_ylim(512, 0)
axes[1].axis('off')

# visualise threshold effect
sns.lineplot(data=optimisation, x='threshold', y='coloc', ax=axes[2], color='black', linestyle='-', label='Original')
sns.lineplot(data=optimisation, x='threshold', y='chance', ax=axes[2], color='black', linestyle='--', label='Randomised')
axes[2].set(ylabel='Number of colocalised spots', xlabel='Distance threshold (px)', ylim=(-2, 102))
axes[2].legend(frameon=False, loc='upper left')

# -------Add image panels-------
microim1 = microshow(
    images=[red_proj], 
    cmaps=['pure_magenta'], #flip_map=[True],
    label_color='black', ax=axes[3], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000]])

microim1 = microshow(
    images=[grn_proj], 
    cmaps=['pure_green'], #flip_map=[True],
    label_color='black', ax=axes[4], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000]])

microim1 = microshow(
    images=[red_proj, grn_proj],
    cmaps=['pure_magenta', 'pure_green'],  # flip_map=[True],
    label_color='black', ax=axes[5], unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[[0, 2000], [0, 2000]])

# Add spots for individual channels
for i, channel in enumerate([641, 488]):
    
    df = colocalised[colocalised['channel'] == channel].copy()

    sns.scatterplot(
        data=df[~df['pair_id'].isnull()],
        x='centroid-0',
        y='centroid-1',
        marker='o',
        s=10,
        ax=axes[i+3],
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
        ax=axes[i+3],
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
    ax=axes[5],
    facecolor='None',
    edgecolor='#fff',
    linewidth=0.1
)

# Add channel/detection labels to each panel
axes[0].set_title('Original')
axes[1].set_title('Randomised')
axes[3].set_title('AT8')
axes[4].set_title('T181')
axes[5].set_title('Overlay')

# Save final figure
plt.tight_layout()
plt.savefig(f'{output_folder}S6_colocalisation.svg')

""" 
Figure S6: Colocalisation 



"""