"""
Generating Figure S5
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import ListedColormap
from scipy import sparse
from skan import Skeleton, draw, summarize
from skan.csr import skeleton_to_csgraph
from skimage import measure, morphology, transform
from skimage.io import imread

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_localisations = f'{root_path}data/homogenate_SR_data/X0Y0R1W1_641_dbscan.csv'
input_srimage = f'{root_path}data/homogenate_SR_images/X0Y0R1W1_641.tif'
output_folder = f'{root_path}results/S7_superres/'

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

SQUARE = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

# =======Functions=======
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(
                HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            'new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            'new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def process_multi_dil(im, num, element=SQUARE):
    for _ in range(num):
        im = morphology.dilation(im, element)
    return im


def process_multi_ero(im, num, element=SQUARE):
    for _ in range(num):
        im = morphology.erosion(im, element)
    return im


def make_cluster_array(clustered_df, cluster_col='group', scale=8, image_width=512, image_height=512, output='sparse_matrix'):

    single_coords = clustered_df[['x', 'y', cluster_col]].copy()
    # remove negative indices - possibly an artifact of the fitting process for localisations at the edge of the image?
    single_coords = single_coords[(single_coords['x'] > 0) & (
        single_coords['y'] > 0)].copy()
    single_coords = single_coords[single_coords[cluster_col] != -1].copy()
    single_coords[cluster_col] = single_coords[cluster_col] + \
        1  # fix zero-index for labelled image
    single_coords['x'] = (single_coords['x'] * scale).astype(int)
    single_coords['y'] = (single_coords['y'] * scale).astype(int)
    single_coords = single_coords[(single_coords['x'] < (
        scale*image_height)) & (single_coords['y'] < (scale*image_width))].copy()

    if output == 'sparse_matrix':
        single_coords.drop_duplicates(inplace=True)
        return sparse.coo_matrix(
            (single_coords[cluster_col], (single_coords['x'], single_coords['y'])), shape=(image_width*scale, image_height*scale)).toarray()
    elif output == 'scaled_coords':
        return single_coords
    else:
        logger.info(
            f'{output} not recognised. Please provide either sparse_matrix for the array format or scaled_coords for the scaled, filtered dataframe.')


def make_skeleton(clusters: np.array):

    # Morphological dilation sets the value of a pixel to the maximum over all pixel values within a local neighborhood centered about it. The values where the footprint is 1 define this neighborhood. Dilation enlarges bright regions and shrinks dark regions.
    multi_dilated = process_multi_dil(clusters, 4)

    # The morphological closing of an image is defined as a dilation followed by an erosion. Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks. This tends to “close” up (dark) gaps between (bright) features.
    closed = morphology.closing(multi_dilated)
    clustered_arr = process_multi_ero(closed, 3)

    # Finally, label individual objects with single object ID
    labelled_arr = measure.label(np.where(clustered_arr > 0, 1, 0))

    # Skeletonize - skimage.morphology, compare medial_axis and skeletonise(method='lee')
    clustered_skeleton = morphology.skeletonize(clustered_arr.astype(bool))
    clustered_skeleton = np.where(clustered_skeleton > 0, clustered_arr, 0)
    labelled_skeleton = morphology.skeletonize(labelled_arr.astype(bool))
    labelled_skeleton = np.where(labelled_skeleton > 0, labelled_arr, 0)

    return clustered_arr, clustered_skeleton, labelled_arr, labelled_skeleton


# Determine properties of skeletons - SKAN
def _clean_positions_dict(d, g):
    for k in list(d.keys()):
        if k not in g:
            del d[k]
        elif g.degree(k) == 0:
            g.remove_node(k)


def measure_SKAN(clusters_array, output_path=False, spacing_nm=107, scale=False):
    if scale:
        spacing_nm = spacing_nm / scale

    clustered_arr, clustered_skeleton, labelled_arr, labelled_skeleton = make_skeleton(
        clusters_array)

    skele = Skeleton(labelled_skeleton.astype(
        bool), spacing=spacing_nm, keep_images=False)
    measurements_skele = summarize(skele)

    if output_path:
        pixel_graph, coordinates = skeleton_to_csgraph(skele)
        plot_branch_overview(clustered_arr, labelled_skeleton,
                             pixel_graph, coordinates, measurements_skele, output_path)

    return labelled_skeleton, measurements_skele


def plot_branch_overview(original_array, skeleton_array, pixel_graph, coordinates, measurements_skele, output_path=False):

    #       0 endpoint-to-endpoint (isolated branch)
    #       1 junction-to-endpoint
    #       2 junction-to-junction
    #       3 isolated cycle
    branch_palette = {0: '#840032', 1: '#E5DADA', 2: '#002642', 3: '#E59500', }
    branch_palette = ListedColormap(branch_palette.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    axes[0, 0].imshow(original_array, cmap='hot')
    axes[0, 0].set_title('Labelled array (scaled)')

    draw.overlay_skeleton_2d(original_array, skeleton_array.astype(
        bool), dilate=0, axes=axes[0, 1])
    axes[0, 1].set_title('Skeleton ID')

    # Inbuilt function has an error, so build network graph from scratch
    gnx = nx.from_scipy_sparse_matrix(pixel_graph)
    # Note: we invert the positions because Matplotlib uses x/y for
    # scatterplot, but the coordinates are row/column NumPy indexing
    positions = dict(zip(range(coordinates.shape[0]), coordinates[:, ::-1]))
    _clean_positions_dict(positions, gnx)  # remove nodes not in Graph
    nx.draw_networkx(gnx, pos=positions,
                     ax=axes[0, 2], node_size=10, font_size=5)

    draw.overlay_euclidean_skeleton_2d(original_array, measurements_skele,
                                       skeleton_color_source='skeleton-id', axes=axes[1, 0], skeleton_colormap='magma')
    branch_positions = measurements_skele.groupby(
        'skeleton-id').mean()[['image-coord-src-0', 'image-coord-src-1']].reset_index()
    for branch_id, x, y in branch_positions.values:
        axes[1, 0].annotate(branch_id, (y, x), color='white')
    axes[1, 0].set_title('Euclidean Skeleton + ID')

    draw.overlay_euclidean_skeleton_2d(original_array, measurements_skele,
                                       skeleton_color_source='branch-type', axes=axes[1, 1], skeleton_colormap=branch_palette)
    axes[1, 1].set_title('Branch type')

    draw.overlay_euclidean_skeleton_2d(original_array, measurements_skele,
                                       skeleton_color_source='branch-distance', axes=axes[1, 2], skeleton_colormap='magma')
    axes[1, 2].set_title('Branch distance')
    if output_path:
        plt.savefigf(f'{output_path}')
    plt.show()


def plot_skeletonisation(clusters: np.array, viewport=None, output_folder=False):

    clustered_arr, clustered_skeleton, labelled_arr, labelled_skeleton = make_skeleton(
        clusters)

    if not viewport:
        viewport = [(0, clustered_arr.shape[0]), (0, clustered_arr.shape[1])]

    arrays = {
        'Clusters': clusters,
        'Labelled ROIs\nDilate ×4, Erode ×3': clustered_arr,
        'Labelled skeleton': clustered_skeleton,
        'Final ROIs': labelled_arr,
        'Final skeleton': labelled_skeleton,
    }

    fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    for i, (label, arr) in enumerate(arrays.items()):
        # A random colormap for matplotlib
        new_cmap = rand_cmap(len(np.unique(arr)+1), type='bright',
                             first_color_black=True, last_color_black=False)

        (xmin, xmax), (ymin, ymax) = viewport
        ax[i].imshow(arr[xmin:xmax, ymin:ymax],
                     interpolation='none', cmap=new_cmap)
        ax[i].set_title(label)

    if output_folder:
        plt.savefig(f'{output_folder}example_skeletonisation.png')
    plt.show()


# Read in localisation data
clustered_df = pd.read_csv(input_localisations)
clustered_df.drop([col for col in clustered_df.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# remove single-localisation clusters
locs = clustered_df.groupby('group').count()
locs = locs[locs['frame'] > 1].copy().reset_index()['group'].tolist()

cluster_arr = make_cluster_array(clustered_df[clustered_df['group'].isin(locs)].copy(), cluster_col='group', scale=8)
clustered_arr, clustered_skeleton, labelled_arr, labelled_skeleton = make_skeleton(
    cluster_arr)

arrays = {
    'Clustered localisations': make_cluster_array(clustered_df, cluster_col='group', scale=8),
    'Labelled ROIs\nDilate ×4, Erode ×3': clustered_arr,
    'Final ROIs': labelled_arr,
    'Final skeleton': labelled_skeleton,
}

# Read in raw image
# original = imread(input_srimage, key=np.arange(0, 5000))
# original_image = transform.rescale(
#     np.max(original, axis=0), 8, preserve_range=True)


# =======================Supplementary figure=======================
# Generate figure panels
fig, axes = plt.subplots(2, 2, figsize=(12.1 * cm, 2 * 6.1 * cm))
axes = axes.ravel()

# Add panel labels
for x, (label, offset)in enumerate(zip(['A', 'B', 'C', 'D'], [-0.25, -0.25, -0.25, -0.25])):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(offset, 0.01, fig.dpi_scale_trans)
    axes[x].text(0.0, 1.0, label, transform=axes[x].transAxes + trans,
                 fontsize=12, va='bottom', fontweight='bold')

viewport = [(700,800), (400,500)]
(xmin, xmax), (ymin, ymax) = viewport

# A random colormap for matplotlib
new_cmap = rand_cmap(len(np.unique(labelled_arr)+1), type='bright',
                        first_color_black=True, last_color_black=False)
for i, (label, arr) in enumerate(arrays.items()):

    axes[i].imshow(arr[xmin:xmax, ymin:ymax],
                    interpolation='none', cmap=new_cmap)
    axes[i].set_title(label)
    axes[i].axis('off')


# Save final figure
plt.tight_layout()
plt.savefig(f'{output_folder}S7_superres.svg')