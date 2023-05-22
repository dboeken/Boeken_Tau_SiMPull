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
    
# ===============Process data ===============
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


# ======Process example images======
example_BSA = imread(f'{root_path}data/peptide_images/X6Y1R3W3_641.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)
np.save(f'{output_folder}example_BSA.npy', example_BSA)

example_monomer = imread(
    f'{root_path}data/peptide_images/X4Y1R3W2_641.tif')
example_monomer = np.mean(example_monomer[10:, :, :], axis=0)
np.save(f'{output_folder}example_monomer.npy', example_monomer)

example_dimer = imread(
    f'{root_path}data/peptide_images/X1Y1R2W2_641.tif')
example_dimer = np.mean(example_dimer[10:, :, :], axis=0)
np.save(f'{output_folder}example_dimer.npy', example_dimer)