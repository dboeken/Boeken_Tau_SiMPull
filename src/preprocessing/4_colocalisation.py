import os

from scipy import stats
import numpy as np
import pandas as pd
from loguru import logger
from skimage.io import imread

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data\data_path.txt'):
    root_path = open('data\data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/colocalisation_data/colocalisation_summary.csv'
input_path_spots = f'{root_path}data/colocalisation_data/colocalisation_spots.csv'
image_path = f'{root_path}data/colocalisation_images/Composite.tif'
output_folder = 'results/4_colocalisation/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# =================Set parameters=================
sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

antibody_dict = {488: 'T181', 641: 'AT8', 'colocalised': 'colocalised'}

# =================Prepare image=================
# example_image = imread(image_path)
# np.save(f'{output_folder}example_image.npy', example_image)

# =================Proportion calculations=================
# Read in summary FOV data
coloc_spots = pd.read_csv(f'{input_path}')
coloc_spots.drop([col for col in coloc_spots.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)

coloc_spots['position'] = coloc_spots['fov'].str[:4]

# calculate mean
meandf = coloc_spots.groupby(
    ['sample', 'position', 'channel', 'detect', 'capture', 'layout']).mean().reset_index()

meandf['disease_state'] = meandf['sample'].astype(str).map(sample_dict)

# only colocalised images
filtered_meandf = meandf[meandf['detect'] == 'AT8 r T181 b'].copy()

# remove one set of channels for coloc spots
filtered_coloc = filtered_meandf[filtered_meandf['channel'] == 488].copy()
filtered_coloc = filtered_coloc[['disease_state', 'channel', 'sample',
                                 'capture', 'coloc_spots']].rename(columns={'coloc_spots': 'total_spots'})
filtered_coloc['channel'] = 'colocalised'

for_plotting = pd.concat([filtered_meandf[[
                         'disease_state', 'channel', 'sample', 'capture', 'total_spots']], filtered_coloc])

for_plotting['antibody_state'] = for_plotting['channel'].map(antibody_dict)

for_plotting_proportion = filtered_meandf[[
    'disease_state', 'channel', 'sample', 'capture', 'proportion_coloc', 'chance_proportion_coloc']].copy()

for_plotting_proportion['antibody_state'] = for_plotting_proportion['channel'].map(antibody_dict)

mean_for_plotting_proportion = for_plotting_proportion.groupby(
    ['capture', 'sample', 'channel', 'disease_state']).mean().reset_index()

# =========Intensity calculations========
for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample', 'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(antibody_dict)

melted = pd.melt(for_plotting_intensity, id_vars=['disease_state', 'channel', 'sample', 'capture', 'mean_intensity', 'antibody_state'], value_vars=['mean_intensity-coloc', 'mean_intensity-noncoloc'], value_name='coloc_intensity', var_name='coloc_status')

melted['coloc_status'] = melted['coloc_status'].str.replace(
    'mean_intensity-', '')
melted['key'] = melted['coloc_status'] + \
    '_' + melted['antibody_state'].astype(str)

#==============intensity coloc vs non-coloc==============
for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample', 'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(antibody_dict)
for_plotting_intensity['intensity_ratio'] = for_plotting_intensity['mean_intensity-coloc'] / \
    for_plotting_intensity['mean_intensity-noncoloc']
for_plotting_intensity['log2_intensity_ratio'] = np.log2(
    for_plotting_intensity['intensity_ratio'])

# =====================stochiometry=====================
input_path_spots = f'{root_path}data/colocalisation_data/colocalisation_spots.csv'

colocalised_summary = pd.read_csv(f'{input_path_spots}')
colocalised = colocalised_summary[colocalised_summary['coloc?'] == 1].copy()
colocalised['disease_state'] = colocalised['sample'].astype(
    str).map(sample_dict)

# merge table on mean intenisty of colocalised spots in both channels and update labels
for_plotting = pd.pivot(colocalised, index=['fov', 'layout', 'sample', 'capture', 'pair_id', 'disease_state'], columns=['channel'], values=['mean_intensity']).reset_index()
for_plotting.columns = [f'{x}' if y == '' else f'{x}_{y}' for x, y in for_plotting.columns]
for_plotting['norm_mean_intensity_488'] = for_plotting['mean_intensity_488'] / 1000
for_plotting['norm_mean_intensity_641'] = for_plotting['mean_intensity_641'] / 1000

filtered_disease = for_plotting[for_plotting['disease_state'] == 'AD'].copy()

AD_brightness_plotting = for_plotting_intensity[for_plotting_intensity['sample'].isin(['13', '55', '246'])].groupby(
    ['capture', 'sample', 'channel', 'disease_state']).mean().reset_index()

brightness_ratio_pval = []
for (capture, channel), df in AD_brightness_plotting.groupby(['capture', 'channel']):
    _, pval = stats.ttest_1samp(df['intensity_ratio'], popmean=1)
    brightness_ratio_pval.append([capture, channel, pval])
    
# make new df with channel, capture, p value and star rating
brightness_ratio_stats = pd.DataFrame(brightness_ratio_pval, columns=[
    'capture', 'channel', 'pval'])
brightness_ratio_stats['significance'] = ['****' if val < 0.0001 else ('***' if val < 0.001 else (
    '**' if val < 0.01 else ('*' if val < 0.05 else 'ns')))for val in brightness_ratio_stats['pval']]

# Add antibody labels and save
for name, df in zip(['mean_for_plotting_proportion', 'filtered_disease','AD_brightness_plotting', 'brightness_ratio_stats'], [mean_for_plotting_proportion, filtered_disease,AD_brightness_plotting, brightness_ratio_stats]):
    try:
        df['antibody_state'] = df['channel'].map(antibody_dict)
    except:
        logger.info(df.columns)
    df.to_csv(
    f'{output_folder}{name}.csv')