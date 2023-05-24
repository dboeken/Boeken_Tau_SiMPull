"""
Preprocessing of the raw data for plotting Figure 3
"""

import os
import pandas as pd

from src.utils import fitting_ecfd_for_plotting

from loguru import logger
logger.info('Import OK')

# =================Set paths=================
data_path = 'data/data_path.txt'

if os.path.exists(data_path):
    root_path = open(data_path, 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/homogenate_SR_data/properties_compiled.csv'
output_folder = 'results/3_homogenate_SR/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# =================Setting parameters=================

thresholds = {
    'smoothed_length_max': 250,
    'smoothed_length_min': 100,
    'scaled_area_max': 15000/1000,
    'scaled_area_min': 5000/1000,
    'eccentricity_max': 0.9,
    'eccentricity_min': 0.7,
    'scaled_perimeter_max': 600,
    'scaled_perimeter_min': 200,
    '#locs_max': 30,
    '#locs_min': 10,
    '#locs_density_max': 0.7,
    '#locs_density_min': 0.3
}

max_ecdf = {
    'smoothed_length': 1000,
    'scaled_area': 3000,
    'eccentricity': 1,
    'scaled_perimeter': 1000,
    '#locs': 120,
    '#locs_density': 2
}

sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

# =================Processing functions=================
def thresholding(data, parameter):
    parameter_cat = parameter + '_cat'
    data[parameter_cat] = ['high' if val > thresholds[parameter]
                           else ('low' if val < 100 else 'medium')for val, detect in data[[parameter, 'detect']].values]


def proportion_calc(data, parameter_cat):

    proportion = (data.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter_cat]).count(
    )['label'] / data.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
    proportion['label'] = proportion['label'] * 100
    proportion = pd.pivot(
        proportion,
        index=['capture', 'sample', 'slide_position',
               'detect', 'disease_state'],
        columns=parameter_cat,
        values='label'
    ).fillna(0).reset_index()

    proportion_plotting = proportion.groupby(
        ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()

    return proportion_plotting


# =================Preprocessing of data=================
# Read in summary measurements
properties = pd.read_csv(f'{input_path}')
properties.drop([col for col in properties.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)

properties['disease_state'] = properties['sample'].astype(
        str).map(sample_dict)

properties['sample'] = properties['sample'].astype(str)

for_plotting = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['#locs'] > 2)  
].copy()

for_plotting['scaled_area'] = (for_plotting['area'] * (107/8)**2)/1000
for_plotting['scaled_perimeter'] = for_plotting['perimeter'] * (107/8)

for_plotting_per_replicate = for_plotting.groupby(
    ['disease_state', 'sample', 'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_mean = for_plotting_per_replicate.groupby(
    ['disease_state', 'sample', 'capture', 'detect']).mean().reset_index()

## calculating the coefficients of variation (pre filtering)
# do additional filtering to get coefficients post filtering 
for_plotting_SD = for_plotting_mean.groupby(
    ['disease_state', 'capture', 'detect']).agg(['mean', 'std'])['smoothed_length'].reset_index()
for_plotting_SD['variation'] = for_plotting_SD['std'] / for_plotting_SD['mean']

parameters = ['smoothed_length', 'eccentricity', 'scaled_perimeter', 'scaled_area', '#locs', '#locs_density']
   
for parameter in parameters:
    parameter_cat = parameter + '_cat'
    for_plotting[parameter_cat] = ['high' if val > thresholds[parameter + '_max']
                                   else ('low' if val < thresholds[parameter + '_min'] else 'medium')for val in for_plotting[parameter].values]
    
# =================Proportion calculation=================
proportion = {}
df_by = {}
df_by_per_replicate = {}
fitted_ecdf= {}
parameter_by_parameter2 = {}
parameter_by_parameter2_for_plotting = {}
for parameter in parameters:
    parameter_cat = parameter + '_cat'

    # calculate percentage of big and small aggregates for CRL and AD
    proportion[parameter] = proportion_calc(for_plotting, parameter_cat)

    # calculate the mean of a parameter depending on another parameter, e.g.
    # mean length of a fibril
    df_by[parameter] = for_plotting.groupby(
        ['capture', 'sample', 'detect', 'disease_state', parameter_cat]).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()
    
    df_by_per_replicate[parameter] = for_plotting.groupby(
        ['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter_cat]).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()
    
    # fit ecdf 
    fitted_ecdf[parameter] = fitting_ecfd_for_plotting(
        for_plotting, 'AT8', max_ecdf[parameter], col=parameter)
    
    # calculate percentage of large/high for a given parameter for another 
    # percentage of long aggregates for fibrils vs round aggregates
    for parameter2 in parameters:
        if parameter2 != parameter:
            parameter2_cat = parameter2 + '_cat'

            parameter_by_parameter2[parameter + '_' + parameter2] = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter_cat, parameter2_cat]).count(
            )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter2_cat]).count()['label']).reset_index()
            
            parameter_by_parameter2[parameter + '_' + \
                                                          parameter2]['label'] = parameter_by_parameter2[parameter + '_' + parameter2]['label'] * 100
            
            parameter_by_parameter2_for_plotting[parameter + '_' +
                                                 parameter2] = parameter_by_parameter2[parameter + '_' +
                                                parameter2].groupby(
                ['capture', 'sample', 'detect', 'disease_state', parameter_cat, parameter2_cat]).mean().reset_index()
            
            parameter_by_parameter2_for_plotting[parameter + '_' +
                                                 parameter2] = parameter_by_parameter2_for_plotting[parameter + '_' +
                                                                                                    parameter2][parameter_by_parameter2_for_plotting[parameter +    '_' +parameter2][parameter_cat]=='high'].copy()

# =================Modified set of filters (length)=================
# additional filtering by length 
for_plotting_fil = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['#locs'] > 2)   &
    (properties['smoothed_length'] > 53)
].copy()

for_plotting_fil['scaled_area'] = (for_plotting_fil['area'] * (107/8)**2)/1000
for_plotting_fil['scaled_perimeter'] = for_plotting_fil['perimeter'] * (107/8)

for_plotting_fil_per_replicate = for_plotting_fil.groupby(
    ['disease_state', 'sample', 'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_fil_mean = for_plotting_fil_per_replicate.groupby(
    ['disease_state', 'sample', 'capture', 'detect']).mean().reset_index()

for parameter in parameters:
    #for_plotting = thresholding(for_plotting, parameter)
    parameter_cat = parameter + '_cat'
    for_plotting_fil[parameter_cat] = ['high' if val > thresholds[parameter + '_max']
                                       else ('low' if val < thresholds[parameter + '_min'] else 'medium')for val in for_plotting_fil[parameter].values]

fil_proportion = {}
fil_df_by = {}
fil_df_by_per_replicate = {}
fil_fitted_ecdf = {}
fil_parameter_by_parameter2 = {}
fil_parameter_by_parameter2_for_plotting = {}
for parameter in parameters:
    parameter_cat = parameter + '_cat'

    # calculate percentage of big and small aggregates for CRL and AD
    fil_proportion[parameter] = proportion_calc(for_plotting_fil, parameter_cat)

    # calculate the mean of a parameter depending on another parameter, e.g.
    # mean length of a fibril
    fil_df_by[parameter] = for_plotting_fil.groupby(
        ['capture', 'sample', 'detect', 'disease_state', parameter_cat]).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

    fil_df_by_per_replicate[parameter] = for_plotting_fil.groupby(
        ['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter_cat]).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

    # fit ecdf
    fil_fitted_ecdf[parameter] = fitting_ecfd_for_plotting(
        for_plotting_fil, 'AT8', max_ecdf[parameter], col=parameter)

    # calculate percentage of large/high for a given parameter for another
    # percentage of long aggregates for fibrils vs round aggregates
    for parameter2 in parameters:
        if parameter2 != parameter:
            parameter2_cat = parameter2 + '_cat'

            fil_parameter_by_parameter2[parameter + '_' + parameter2] = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter_cat, parameter2_cat]).count(
            )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', parameter2_cat]).count()['label']).reset_index()

            fil_parameter_by_parameter2[parameter + '_' +
                                    parameter2]['label'] = fil_parameter_by_parameter2[parameter + '_' + parameter2]['label'] * 100

            fil_parameter_by_parameter2_for_plotting[parameter + '_' +
                                                 parameter2] = fil_parameter_by_parameter2[parameter + '_' +
                                                                                       parameter2].groupby(
                ['capture', 'sample', 'detect', 'disease_state', parameter_cat, parameter2_cat]).mean().reset_index()

            fil_parameter_by_parameter2_for_plotting[parameter + '_' +
                                                 parameter2] = fil_parameter_by_parameter2_for_plotting[parameter + '_' +
                                                                                                    parameter2][fil_parameter_by_parameter2_for_plotting[parameter + '_' + parameter2][parameter_cat] == 'high'].copy()

# =================Saving dfs=================
proportions = []
for key, df in proportion.items():
    df['category'] = key
    proportions.append(df)
proportions = pd.concat(proportions)

fil_proportions = []
for key, df in fil_proportion.items():
    df['category'] = key
    fil_proportions.append(df)
fil_proportions = pd.concat(fil_proportions)

fitted_ecdfs = []
for key, df in fitted_ecdf.items():
    df['category'] = key
    fitted_ecdfs.append(df)
fitted_ecdfs = pd.concat(fitted_ecdfs)

parameter_by_parameter2_for_plotting

parameter_by_parameter2_for_plotting_all = []
for key, df in parameter_by_parameter2_for_plotting.items():
    df['category'] = key
    parameter_by_parameter2_for_plotting_all.append(df)
parameter_by_parameter2_for_plotting_all = pd.concat(
    parameter_by_parameter2_for_plotting_all)

for_plotting.to_csv(f'{output_folder}for_plotting.csv')
proportions.to_csv(f'{output_folder}proportions.csv')
fil_proportions.to_csv(f'{output_folder}fil_proportions.csv')
for_plotting_mean.to_csv(f'{output_folder}for_plotting_mean.csv')
fitted_ecdfs.to_csv(f'{output_folder}fitted_ecdfs.csv')
for_plotting_fil_mean.to_csv(f'{output_folder}for_plotting_fil_mean.csv')
parameter_by_parameter2_for_plotting_all.to_csv(
    f'{output_folder}parameter_by_parameter2_for_plotting_all.csv')