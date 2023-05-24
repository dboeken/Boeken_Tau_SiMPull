"""
Preprocessing of the raw data for plotting Figure 2
"""

import os

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
from skimage.io import imread

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path =''

input_path = f'{root_path}data/homogenate_DL_data/'
output_folder = f'results/2_homogenate_DL/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

# =================Processing functions=================
def read_in(input_path, detect):
    spots = pd.read_csv(f'{input_path}')
    spots.drop([col for col in spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)
    # Drop extra wells collected by default due to grid pattern
    spots.dropna(subset=['sample'], inplace=True)

    spots[['capture', 'sample', 'detect']
          ] = spots['sample'].str.split('_', expand=True)
    filtered_spots = spots[spots['detect'] == detect].copy()

    filtered_spots['disease_state'] = filtered_spots['sample'].astype(
        str).map(sample_dict)
    
    mean_spots = filtered_spots.groupby(['disease_state', 'channel', 'sample',
                                    'capture', 'detect']).mean().reset_index()

    return mean_spots


def intensity_processing(slide_params, spots_intensity, detect):
    slide_params = pd.read_csv(slide_params)
    slide_params.drop([col for col in slide_params.columns.tolist()
                       if 'Unnamed: ' in col], axis=1, inplace=True)

    # remove discard images
    slide_params.dropna(subset=['keep'], inplace=True)

    # expand sample info
    slide_params[['capture', 'sample', 'detect']
                 ] = slide_params['sample'].str.split('_', expand=True)

    spots_intensity = pd.read_csv(spots_intensity)
    spots_intensity['slide_position'] = spots_intensity['well_info'].str[:4]

    spots_intensity = pd.merge(
        slide_params[['slide_position', 'layout',
                      'sample', 'capture', 'detect', 'well_info']],
        spots_intensity,
        on=['slide_position', 'layout', 'well_info'],
        how='right')

    spots_intensity['log_intensity'] = np.log(
        spots_intensity['mean_intensity'])
    filtered_spots_intensity = spots_intensity[spots_intensity['detect'] == detect].copy(
    )
    # if detect == 'HT7':
    #     filtered_spots_intensity = filtered_spots_intensity[filtered_spots_intensity['mean_intensity']>600].copy()

    return filtered_spots_intensity


def fit_ecdf(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result


def sample_ecdf(df, value_cols, num_points=100, method='nearest', order=False):

    test_vals = pd.DataFrame(
        np.arange(0, 1.01, (1/num_points)), columns=['ecdf'])
    test_vals['type'] = 'interpolated'
    interpolated = test_vals.copy()
    for col in value_cols:
        test_df = df.dropna().drop_duplicates(subset=[col])
        ecdf = fit_ecdf(test_df[col])
        test_df['ecdf'] = ecdf(
            test_df.dropna().drop_duplicates(subset=[col])[col])
        combined = pd.concat([test_df.sort_values('ecdf').dropna(), test_vals])
        combined = combined.set_index('ecdf').interpolate(
            method=method, order=order).reset_index()
        interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[
            col].tolist()

    return interpolated


def fitting_ecfd_for_plotting(df_intensity, detect, maxval, col='mean_intensity'):
    fitted_ecdfs = []
    for (capture, sample, position), df in df_intensity.groupby(['capture', 'sample', 'slide_position']):
        filtered_df = df[df[col] < maxval]
        fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
            col], method='nearest', order=False)
        fitted_ecdf['sample'] = sample
        fitted_ecdf['capture'] = capture
        fitted_ecdf['slide_position'] = position
        fitted_ecdf['detect'] = detect
        fitted_ecdfs.append(fitted_ecdf)

    fitted_ecdfs = pd.concat(fitted_ecdfs)
    return fitted_ecdfs


# =================Prepare spot counts=================
spots_AT8 = read_in(f'{input_path}AT8_capture_spots_per_fov.csv', 'AT8')
spots_HT7 = read_in(f'{input_path}HT7_capture_spots_per_fov.csv', 'HT7')
spots_summary = pd.concat([spots_AT8, spots_HT7])
spots_summary.to_csv(f'{output_folder}spots_count_summary.csv')

# =================Prepare per-spot parameters=================
slide_params_AT8_path = f'{input_path}AT8_capture_slide_parameters.csv'
slide_params_HT7_path = f'{input_path}HT7_capture_slide_parameters.csv'
spots_intensity_HT7 = f'{input_path}HT7_capture_compiled_spots.csv'
spots_intensity_AT8 = f'{input_path}AT8_capture_compiled_spots.csv'

# Scale brightness values by 1000
AT8_spots_intensity = intensity_processing(
    slide_params_AT8_path, spots_intensity_AT8, 'AT8')
AT8_spots_intensity['norm_mean_intensity'] = AT8_spots_intensity['mean_intensity'] / 1000
HT7_spots_intensity = intensity_processing(
    slide_params_HT7_path, spots_intensity_HT7, 'HT7')
HT7_spots_intensity['norm_mean_intensity'] = HT7_spots_intensity['mean_intensity'] / 1000

# Fit cumulative distributions to brightness values
fitted_ecdf_HT7 = fitting_ecfd_for_plotting(HT7_spots_intensity, 'HT7', 15, col='norm_mean_intensity')
fitted_ecdf_HT7.to_csv(f'{output_folder}fitted_ecdf_HT7.csv')

fitted_ecdf_AT8 = fitting_ecfd_for_plotting(AT8_spots_intensity, 'AT8', 1, col='norm_mean_intensity')
fitted_ecdf_AT8.to_csv(f'{output_folder}fitted_ecdf_AT8.csv')

# =================Prepare summary parameters=================
# Mean intensity for HT7 capture
compiled_spots = pd.concat([HT7_spots_intensity, AT8_spots_intensity])
compiled_spots['disease_state'] = compiled_spots['sample'].map(sample_dict)

compiled_spots = compiled_spots[compiled_spots.detect != 'IgG'].copy()

mean_intensity_per_replicate = compiled_spots.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state']).mean().reset_index()

mean_intensity_per_replicate.to_csv(
    f'{output_folder}mean_intensity_per_replicate.csv')

mean_intensity_plotting = mean_intensity_per_replicate.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()

mean_intensity_plotting = mean_intensity_plotting[mean_intensity_plotting['sample']!= 'BSA'].copy()

mean_intensity_plotting.to_csv(f'{output_folder}mean_intensity.csv')

# Calculate proportion of spots > threshold intensity
thresholds = {'AT8': 500, 'HT7': 2000}
compiled_spots['bright_cat'] = ['bright' if val > thresholds[detect] else 'dim' for val, detect in compiled_spots[['mean_intensity', 'detect']].values]

proportion_intensity = (compiled_spots.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout', 'bright_cat']).count()['label'] / compiled_spots.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout']).count()['label']).reset_index()
proportion_intensity['label'] = proportion_intensity['label'] * 100
proportion_intensity = pd.pivot(
    proportion_intensity,
    index=['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'layout'],
    columns='bright_cat',
    values='label'
).fillna(0).reset_index()

proportion_intensity_plotting = proportion_intensity.groupby(['capture', 'sample', 'detect', 'disease_state']).mean().reset_index().drop('layout', axis=1)

proportion_intensity_plotting.to_csv(f'{output_folder}proportion_intensity.csv')

# =================Prepare example images=================
# Read in raw images
example_AD = imread(f'{root_path}data/homogenate_DL_images/X8Y2R2W2_641.tif')
example_CRL = imread(f'{root_path}data/homogenate_DL_images/X1Y0R4W2_641.tif')

# Make mean projections, save to numpy array
example_AD = np.mean(example_AD[10:, :, :], axis=0)
np.save(f'{output_folder}example_AD.npy', example_AD)

example_CRL = np.mean(example_CRL[10:, :, :], axis=0)
np.save(f'{output_folder}example_CRL.npy', example_CRL)