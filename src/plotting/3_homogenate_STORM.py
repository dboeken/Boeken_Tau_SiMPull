from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from matplotlib.colors import ListedColormap
import matplotlib
from statannotations.Annotator import Annotator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms

from loguru import logger
logger.info('Import OK')

data_path = 'data/data_path.txt'

if os.path.exists(data_path):
    root_path = open(data_path, 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/homogenate_SR_data/properties_compiled.csv'
output_folder = 'results/3_homogenate_SR/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
cm = 1/2.54

matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi']= 300

# Read in summary measurements
properties = pd.read_csv(f'{input_path}')
properties.drop([col for col in properties.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)

sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}

properties['disease_state'] = properties['sample'].astype(
        str).map(sample_dict)

properties['sample'] = properties['sample'].astype(str)

palette = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
}


palette_repl = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#F03A47',
    '55': '#F03A47',
    '246': '#F03A47',
    'BSA': '#A9A9A9',
    'AD Mix': '#A9A9A9',

}


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



def scatbarplot(ycol, ylabel, palette, ax, data):
    order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=False,
        order=order,
        ci = 'sd'
    )
    sns.stripplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order, 
        
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def scatbarplot2(ycol, ylabel, palette, ax, data):
    order = ['high', 'low']
    sns.barplot(
        data=data,
        x='smoothed_length_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=False,
        order=order,
    )
    sns.stripplot(
        data=data,
        x='smoothed_length_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('high', 'low')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='smoothed_length_cat', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.set_xticklabels(['Long', 'Short'])
    ax.legend('', frameon=False)


def plot_interpolated_ecdf(fitted_ecdfs, ycol, huecol, palette, ax=None, orientation=None):

    if not ax:
        fig, ax = plt.subplots()

    if orientation == 'h':
        means = fitted_ecdfs[fitted_ecdfs['type'] == 'interpolated'].groupby(
            [huecol, 'ecdf']).agg(['mean', 'std']).reset_index()
        means.columns = [huecol, 'ecdf', 'mean', 'std']
        means['pos_err'] = means['mean'] + means['std']
        means['neg_err'] = means['mean'] - means['std']

        for hue, data in means.groupby([huecol]):

            ax.plot(
                data['mean'],
                data['ecdf'],
                color=palette[hue],
                label=hue
            )
            ax.fill_betweenx(
                y=data['ecdf'].tolist(),
                x1=(data['neg_err']).tolist(),
                x2=(data['pos_err']).tolist(),
                color=palette[hue],
                alpha=0.3
            )

    else:
        sns.lineplot(
            data=fitted_ecdfs,
            y=ycol,
            x='ecdf',
            hue=huecol,
            palette=palette,
            ci='sd',
            ax=ax
        )

    return fitted_ecdfs, ax




def scatbarplot_hue_ecc(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['fibril', 'round']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='ecc_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x='ecc_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    pairs = [(('fibril', 'AD'), ('round', 'AD')),
             (('fibril', 'CRL'), ('round', 'CRL')), (('fibril', 'AD'), ('fibril', 'CRL')), (('round', 'AD'), ('round', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='ecc_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)
    ax.legend('', frameon=False)


def scatbarplot_hue_two_param(ycol, ylabel, palette, ax, data, xcol, high, low, group_label_y=-0.18, group_line_y=-0.05):
    order = ['high', 'low']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x=xcol,
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.2,
        errwidth=2,
        ax=ax,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x=xcol,
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=5,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )


    pairs = [(('high', 'AD'), ('low', 'AD')),
             (('high', 'CRL'), ('low', 'CRL')), (('high', 'AD'), ('high', 'CRL')), (('low', 'AD'), ('low', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x=xcol, y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    

    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([high, low])

    ax.legend('', frameon=False)


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
        test_df = df.dropna(subset=[col])
        ecdf = fit_ecdf(test_df[col])
        test_df['ecdf'] = ecdf(
            test_df.dropna(subset=[col])[col])
        combined = pd.concat([test_df.sort_values(
            'ecdf').dropna(subset=[col]), test_vals])
        combined = combined.set_index('ecdf').interpolate(
            method=method, order=order).reset_index()
        interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[
            col].tolist()

    return interpolated


def fitting_ecfd_for_plotting(df_intensity, detect, maxval, col):
    fitted_ecdfs = []
    for (sample, position), df in df_intensity.groupby(['sample', 'slide_position']):
        filtered_df = df[df[col] < maxval].copy()
        fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
            col], method='nearest', order=False)
        fitted_ecdf['sample'] = sample
        #fitted_ecdf['disease_state'] = disease_state
        # fitted_ecdf['capture'] = capture
        fitted_ecdf['slide_position'] = position
        fitted_ecdf['detect'] = detect
        fitted_ecdfs.append(fitted_ecdf)

    fitted_ecdfs = pd.concat(fitted_ecdfs)
    return fitted_ecdfs


def ecfd_plot(ycol, ylabel, palette, ax, df):
    sns.lineplot(
        data=df.reset_index(),
        y=ycol,
        x='ecdf',
        hue='sample',
        palette=palette_repl,
        ci='sd',
        ax=ax)

    ax.set(ylabel=ylabel, xlabel='Proportion of spots')
    ax.legend(frameon=False)


def hexbinplotting(colour, ax, data, disease_state):

    df = data[data['disease_state'] == disease_state].copy()
    hexs = ax.hexbin(data=df, x='eccentricity',
                     y='smoothed_length', cmap=colour, vmin=0, vmax=1200)
    ax.set(ylabel='Length [nm]')
    ax.set(xlabel='Eccentricity')
    # sns.kdeplot(data=df, x='smoothed_length', y='eccentricity',
    #             color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 550)
    

    return hexs




for_plotting = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['#locs'] > 2)  #&
    #(properties['smoothed_length'] > 53)
    #(properties['area'] > 2)
].copy()

#for_plotting.to_csv(f'{output_folder}test.csv')

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
    #for_plotting = thresholding(for_plotting, parameter)
    parameter_cat = parameter + '_cat'
    for_plotting[parameter_cat] = ['high' if val > thresholds[parameter + '_max']
                                   else ('low' if val < thresholds[parameter + '_min'] else 'medium')for val in for_plotting[parameter].values]

    #d[parameter] = pd.DataFrame()


proportion = {}
df_by = {}
df_by_per_replicate = {}
fitted_ecdf= {}
parameter_by_parameter2 = {}
parameter_by_parameter2_for_plotting = {}
for parameter in parameters:
    parameter_cat = parameter + '_cat'
    # df_by[parameter] = pd.DataFrame()
    # df_by_per_replicate[parameter] = pd.DataFrame()

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





############
# additional filtering by length 


for_plotting_fil = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['#locs'] > 2)   &
    (properties['smoothed_length'] > 53)
    #(properties['area'] > 2)
].copy()

#for_plotting_fil.to_csv(f'{output_folder}test.csv')

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

    #d[parameter] = pd.DataFrame()


fil_proportion = {}
fil_df_by = {}
fil_df_by_per_replicate = {}
fil_fitted_ecdf = {}
fil_parameter_by_parameter2 = {}
fil_parameter_by_parameter2_for_plotting = {}
for parameter in parameters:
    parameter_cat = parameter + '_cat'
    # fil_df_by[parameter] = pd.DataFrame()
    # fil_df_by_per_replicate[parameter] = pd.DataFrame()

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



# Compile single df from dictionary
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




###########

# # Make main figure
fig = plt.figure(figsize=(12.1 * cm, 4 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=8, ncols=4, wspace=0.95, hspace=0.8)
axA1 = fig.add_subplot(gs1[0:1, 0:1])
axA2 = fig.add_subplot(gs1[0:1, 1:2])
axA3 = fig.add_subplot(gs1[1:2, 0:1])
axA4 = fig.add_subplot(gs1[1:2, 1:2])
axA5 = fig.add_subplot(gs1[0:1, 2:3])
axA6 = fig.add_subplot(gs1[1:2, 2:3])
axA7 = fig.add_subplot(gs1[0:1, 3:4])
axA8 = fig.add_subplot(gs1[1:2, 3:4])

axB1 = fig.add_subplot(gs1[2:4, 0:1])
axB2 = fig.add_subplot(gs1[2:4, 1:2])
axB3 = fig.add_subplot(gs1[2:4, 2:3])
axB4 = fig.add_subplot(gs1[2:4, 3:4])

axC1 = fig.add_subplot(gs1[4:6, 0:2])
axC2 = fig.add_subplot(gs1[4:6, 2:4])

axD1 = fig.add_subplot(gs1[6:8, 0:1])
axD2 = fig.add_subplot(gs1[6:8, 1:2])
axD3 = fig.add_subplot(gs1[6:8, 2:3])
axD4 = fig.add_subplot(gs1[6:8, 3:4])



for ax, label in zip([axA1, axB1, axC1, axD1], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
axA1.axis('off')
axA2.axis('off')
axA3.axis('off')
axA4.axis('off')
axA5.axis('off')
axA6.axis('off')
axA7.axis('off')
axA8.axis('off')

# --------Panel B--------
scatbarplot('smoothed_length', 'Mean length [nm]',
            palette, axB1, for_plotting_mean)
axB1.set_title('Length')

scatbarplot('scaled_perimeter', 'Mean perimeter [nm]',
            palette, axB2, for_plotting_mean)
axB2.set_title('Perimeter')

scatbarplot('scaled_area', 'Mean area [x 10$^3$ nm$^2$]',
            palette, axB3, for_plotting_mean)
axB3.set_title('Area')

scatbarplot('eccentricity', 'Mean eccentricity',
            palette, axB4, for_plotting_mean)
axB4.set_title('Eccentricity')

# --------Panel C--------

plot_interpolated_ecdf(fitted_ecdf['smoothed_length'], ycol='smoothed_length', huecol='sample', palette=palette_repl, ax=axC1, orientation='h')

plot_interpolated_ecdf(fitted_ecdf['eccentricity'], ycol='eccentricity', huecol='sample', palette=palette_repl, ax=axC2, orientation='h')
# ecfd_plot('smoothed_length', 'Length',
#           palette, axC1, fitted_ecdf['smoothed_length'])

# --------Panel D--------
scatbarplot('high', 'Long [%]',
            palette, axD1, proportion['smoothed_length'])

scatbarplot('high', 'perimeter long [%]',
            palette, axD2, proportion['scaled_perimeter'])

scatbarplot('high', 'Large [%]',
            palette, axD3, proportion['scaled_area'])


scatbarplot('high', 'Fibrils [%]',
            palette, axD4, proportion['eccentricity'])


handles, labels = axC1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9']}
for label, ax in zip(['Length (nm)', 'Eccentricity'], [axC1, axC2]):
    ax.legend(simple_legend.values(), simple_legend.keys(), frameon=False)
    ax.set_xlabel(label)
    ax.set_ylabel('Proportion')


plt.tight_layout()
plt.savefig(f'{output_folder}Figure3_homogenate_SR.svg')
plt.show()




####################


# # ####supplemental figure####




new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)


fig = plt.figure(figsize=(18.4 * cm, 4 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=4, ncols=4, wspace=0.9, hspace=0.4)
axA = fig.add_subplot(gs1[0:1, 0:2])
axB = fig.add_subplot(gs1[0:1, 2:4])
axC1 = fig.add_subplot(gs1[1:2, 0:1])
axC2 = fig.add_subplot(gs1[1:2, 1:2])
axC3 = fig.add_subplot(gs1[1:2, 2:3])
axC4 = fig.add_subplot(gs1[1:2, 3:4])

axD1 = fig.add_subplot(gs1[2:3, 0:1])
axD2 = fig.add_subplot(gs1[2:3, 1:2])
axD3 = fig.add_subplot(gs1[2:3, 2:3])
axD4 = fig.add_subplot(gs1[2:3, 3:4])

axE1 = fig.add_subplot(gs1[3:4, 0:1])
axE2 = fig.add_subplot(gs1[3:4, 1:2])
axE3 = fig.add_subplot(gs1[3:4, 2:3])
axE4 = fig.add_subplot(gs1[3:4, 3:4])


for ax, label in zip([axA, axB, axC1, axC2, axC3, axC4, axD1, axE1], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------

hexs0 = hexbinplotting(
    colour=cmap, ax=axA, data=for_plotting, disease_state='AD')
cb = plt.colorbar(hexs0, ax=axA)
#cb.set_label('Count', rotation=270, labelpad=15)
axA.set_title('AD', fontsize=8)

hexs1 = hexbinplotting(
    colour=cmap, ax=axB, data=for_plotting, disease_state='CRL')
cb = plt.colorbar(hexs1, ax=axB)
cb.set_label('Count', rotation=270, labelpad=15)
axB.set_title('CRL', fontsize=8)



# --------Panel C--------


scatbarplot('label', 'Long [%]',
                          palette, axC1, parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'][parameter_by_parameter2_for_plotting['smoothed_length_eccentricity']['eccentricity_cat']=='low'].copy() )


scatbarplot2('label', 'Fibril [%]',
            palette, axC2, parameter_by_parameter2_for_plotting['eccentricity_smoothed_length'][parameter_by_parameter2_for_plotting['eccentricity_smoothed_length']['disease_state']=='AD'].copy())


scatbarplot('#locs', 'Mean number of \n localisations per cluster',
            palette, axC3, for_plotting_mean)


scatbarplot('#locs_density', 'Localisation density \n localisations per 10$^3$ nm$^2$',
            palette, axC4, for_plotting_mean)

# --------Panel D--------
scatbarplot('smoothed_length', 'Mean length [nm]',
            palette, axD1, for_plotting_fil_mean)
axB1.set_title('Length')

scatbarplot('scaled_perimeter', 'Mean perimeter [nm]',
            palette, axD2, for_plotting_fil_mean)
axB2.set_title('Perimeter')

scatbarplot('scaled_area', 'Mean area [x 10$^3$ nm$^2$]',
            palette, axD3, for_plotting_fil_mean)
axB3.set_title('Area')

scatbarplot('eccentricity', 'Mean eccentricity',
            palette, axD4, for_plotting_fil_mean)
axB4.set_title('Eccentricity')

# --------Panel E--------

scatbarplot('high', 'Long [%]',
            palette, axE1, fil_proportion['smoothed_length'])

scatbarplot('high', 'perimeter long [%]',
            palette, axE2, fil_proportion['scaled_perimeter'])

scatbarplot('high', 'Large [%]',
            palette, axE3, fil_proportion['scaled_area'])


scatbarplot('high', 'Fibrils [%]',
            palette, axE4, fil_proportion['eccentricity'])




plt.tight_layout()
plt.savefig(f'{output_folder}S2_homogenate_SR.svg')
plt.show()

