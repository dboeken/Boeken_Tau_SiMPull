###include:

# 1: example images
# 2: cumulative distribution + error bar
# 3: mean length and area
# 4: ratios of large aggregates

# import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from matplotlib.colors import ListedColormap
import matplotlib
# from sklearn.decomposition import PCA
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
output_folder = 'results/super-res/summary/'

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

    # sample_palette = fitted_ecdfs[['sample', huecol]].drop_duplicates()
    # sample_palette['color'] = sample_palette[huecol].map(palette)
    # sample_palette = dict(sample_palette[['sample', 'color']].values)
    # fitted_ecdfs['color'] = fitted_ecdfs['sample'].map(sample_palette)

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



# def scatbarplot_hue_two_param(ycol, ylabel, palette, ax, data, xcol, group_label_y=-0.18, group_line_y=-0.05):
#     order = ['fibril', 'round']
#     hue_order = ['AD', 'CRL']
#     sns.barplot(
#         data=data,
#         x=xcol,
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         capsize=0.2,
#         errwidth=2,
#         ax=ax,
#         dodge=True,
#         order=order,
#         hue_order=hue_order,
#         edgecolor='white'
#     )
#     sns.stripplot(
#         data=data,
#         x=xcol,
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         ax=ax,
#         edgecolor='#fff',
#         linewidth=1,
#         s=10,
#         order=order,
#         hue_order=hue_order,
#         dodge=True,
#     )

#     # add by chance lines by disease state
#     # for disease, df2 in data.groupby('disease_state'):
#     #     ax.axhline(df2['chance_proportion_coloc'].mean(),
#     #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

#     pairs = [(('fibril', 'AD'), ('round', 'AD')),
#              (('fibril', 'CRL'), ('round', 'CRL')), (('fibril', 'AD'), ('fibril', 'CRL')), (('round', 'AD'), ('round', 'CRL'))]
#     annotator = Annotator(
#         ax=ax, pairs=pairs, data=data, x='ecc_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
#     annotator.configure(test='t-test_ind', text_format='star',
#                         loc='inside')
#     annotator.apply_and_annotate()

#     ax.set(ylabel=ylabel)

#     # ax.set_xlabel('')
#     # ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
#     # ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
#     # ax.set_yticks([0, 25, 50, 75, 100])
#     # ax.set_yticklabels(['0', '25', '50', '75', '100'])
#     # ax.annotate('AT8', xy=(0.25, group_label_y),
#     #             xycoords='axes fraction', ha='center')
#     # ax.annotate('T181', xy=(0.75, group_label_y),
#     #             xycoords='axes fraction', ha='center')
#     # trans = ax.get_xaxis_transform()
#     # ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
#     #         color="black", transform=trans, clip_on=False)
#     # ax.plot([0.75, 1.25], [group_line_y, group_line_y],
#     #         color="black", transform=trans, clip_on=False)

#     ax.legend('', frameon=False)


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

    # add by chance lines by disease state
    # for disease, df2 in data.groupby('disease_state'):
    #     ax.axhline(df2['chance_proportion_coloc'].mean(),
    #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    pairs = [(('fibril', 'AD'), ('round', 'AD')),
             (('fibril', 'CRL'), ('round', 'CRL')), (('fibril', 'AD'), ('fibril', 'CRL')), (('round', 'AD'), ('round', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='ecc_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    # ax.set_xlabel('')
    # ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
    # ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
    # ax.set_yticks([0, 25, 50, 75, 100])
    # ax.set_yticklabels(['0', '25', '50', '75', '100'])
    # ax.annotate('AT8', xy=(0.25, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # ax.annotate('T181', xy=(0.75, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # trans = ax.get_xaxis_transform()
    # ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)
    # ax.plot([0.75, 1.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)

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

    # add by chance lines by disease state
    # for disease, df2 in data.groupby('disease_state'):
    #     ax.axhline(df2['chance_proportion_coloc'].mean(),
    #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

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
    # ax.set_yticks([0, 25, 50, 75, 100])
    # ax.set_yticklabels(['0', '25', '50', '75', '100'])
    # ax.annotate('AT8', xy=(0.25, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # ax.annotate('T181', xy=(0.75, group_label_y),
    #             xycoords='axes fraction', ha='center')
    # trans = ax.get_xaxis_transform()
    # ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)
    # ax.plot([0.75, 1.25], [group_line_y, group_line_y],
    #         color="black", transform=trans, clip_on=False)

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
    (properties['#locs'] > 2)  #5&
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
# additional filtering by lenth 


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


# new_colors = ['#ffffff'] + \
#     list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# # Turn this into a new colour map, and visualise it
# cmap = ListedColormap(new_colors)



# fig, axes = plt.subplots(3, 2, figsize=(18.4 * cm, 3 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.2)

# hexs0 = hexbinplotting(
#     colour=cmap, ax=axes[0], data=for_plotting, disease_state='AD')
# cb = plt.colorbar(hexs0, ax=axes[0])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[0].set_title('AD', fontsize=8)

# hexs1 = hexbinplotting(
#     colour=cmap, ax=axes[1], data=for_plotting, disease_state='CRL')
# cb = plt.colorbar(hexs1, ax=axes[1])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[1].set_title('CRL', fontsize=8)

# scatbarplot_hue_two_param(ycol='smoothed_length', ylabel='Mean length [nm]',
#                           palette=palette, ax=axes[2], data=df_by['eccentricity'], xcol='eccentricity_cat', high = 'fibril', low ='round')


# # scatbarplot_hue_ecc(ycol='eccentricity', ylabel='eccentricity',
# #                     palette=palette, ax=axes[2], data=whatever)


# # scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
# #                        palette=palette, ax=axes[5], data=whatever2)

# scatbarplot_hue_two_param(ycol='eccentricity', ylabel='Mean eccentricity',
#                           palette=palette, ax=axes[3], data=df_by['smoothed_length'], xcol = 'smoothed_length_cat', high = 'long', low = 'short')

# scatbarplot_hue_two_param('label', 'Long [%]',
#                     palette, axes[4], parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], xcol='eccentricity_cat', high='fibril', low='round')


# scatbarplot_hue_two_param('label', 'Fibril [%]',
#                           palette, axes[5], parameter_by_parameter2_for_plotting['eccentricity_smoothed_length'], xcol = 'smoothed_length_cat', high = 'long', low = 'short')


# plt.tight_layout()

# plt.savefig(f'{output_folder}Supp_homogenate_SR_AT8-AT8.svg')

# plt.savefig(f'{output_folder}Supp.svg')
# pg.anova(
#     data=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], dv= 'label', between= ['eccentricity_cat', 'disease_state']).round(3)
# pg.pairwise_tukey(
#     data=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], dv='label', between='disease_state').round(3)
# pg.pairwise_tukey(
#     data=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], dv='label', between='eccentricity_cat').round(3)




# ###########################################################


# fig, axes = plt.subplots(6, 2, figsize=(18.4 * cm, 6 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.2)


# ecfd_plot('#locs', 'Number of locs per cluster',
#           palette, axes[0], fitted_ecdf['#locs'])

# scatbarplot('#locs', 'Mean number of locs per cluster',
#             palette, axes[2], for_plotting_mean)


# scatbarplot('high', 'Bright [%]',
#             palette, axes[4], proportion['#locs'], )

# ecfd_plot('#locs_density', 'locs density',
#           palette, axes[1], fitted_ecdf['#locs_density'])

# scatbarplot('#locs_density', 'Mean locs density',
#             palette, axes[3], for_plotting_mean)


# scatbarplot('high', 'dense [%]',
#             palette, axes[5], proportion['#locs_density'])

# scatbarplot_hue_two_param(ycol='eccentricity', ylabel='Mean eccentricity',
#                        palette=palette, ax=axes[6], data=df_by['#locs'], xcol = '#locs_cat', high = 'bright', low = 'dim')

# scatbarplot_hue_two_param(ycol='smoothed_length', ylabel='Length',
#                           palette=palette, ax=axes[7], data=df_by['#locs'], xcol='#locs_cat', high='bright', low='dim')

# scatbarplot_hue_two_param(ycol='label', ylabel='Fibril %',
#                           palette=palette, ax=axes[8], data=parameter_by_parameter2_for_plotting['eccentricity_#locs'], xcol='#locs_cat', high='bright', low='dim')

# scatbarplot_hue_two_param(ycol='label', ylabel='Long %',
#                           palette=palette, ax=axes[9], data=parameter_by_parameter2_for_plotting['smoothed_length_#locs'], xcol='#locs_cat', high='bright', low='dim')

# scatbarplot_hue_two_param(ycol='label', ylabel='Bright %',
#                     palette=palette, ax=axes[10], data=parameter_by_parameter2_for_plotting['#locs_eccentricity'], xcol='eccentricity_cat', high='Fibril', low='round')


# scatbarplot_hue_two_param(ycol='#locs', ylabel='#locs',
#                           palette=palette, ax=axes[11], data=df_by['eccentricity'], xcol='eccentricity_cat', high='Fibril', low='round')


# plt.tight_layout()

# plt.savefig(f'{output_folder}locs_homogenate_SR_AT8-AT8.svg')



# ###########################################################


# # ####supplemental figure####


# new_colors = ['#ffffff'] + \
#     list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# # Turn this into a new colour map, and visualise it
# cmap = ListedColormap(new_colors)


# fig, axes = plt.subplots(3, 2, figsize=(18.4 * cm, 3 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.2)

# hexs0 = hexbinplotting(
#     colour=cmap, ax=axes[0], data=for_plotting, disease_state='AD')
# cb = plt.colorbar(hexs0, ax=axes[0])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[0].set_title('AD', fontsize=8)

# hexs1 = hexbinplotting(
#     colour=cmap, ax=axes[1], data=for_plotting, disease_state='CRL')
# cb = plt.colorbar(hexs1, ax=axes[1])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[1].set_title('CRL', fontsize=8)

# scatbarplot_hue_two_param(ycol='smoothed_length', ylabel='Mean length [nm]',
#                           palette=palette, ax=axes[2], data=df_by['eccentricity'], xcol='eccentricity_cat', high='fibril', low='round')


# # scatbarplot_hue_ecc(ycol='eccentricity', ylabel='eccentricity',
# #                     palette=palette, ax=axes[2], data=whatever)


# # scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
# #                        palette=palette, ax=axes[5], data=whatever2)

# scatbarplot_hue_two_param(ycol='eccentricity', ylabel='Mean eccentricity',
#                           palette=palette, ax=axes[3], data=df_by['smoothed_length'], xcol='smoothed_length_cat', high='long', low='short')

# scatbarplot_hue_two_param('label', 'Long [%]',
#                           palette, axes[4], parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], xcol='eccentricity_cat', high='fibril', low='round')


# scatbarplot_hue_two_param('label', 'Fibril [%]',
#                           palette, axes[5], parameter_by_parameter2_for_plotting['eccentricity_smoothed_length'], xcol='smoothed_length_cat', high='long', low='short')


# plt.tight_layout()

# plt.savefig(f'{output_folder}Supp_homogenate_SR_AT8-AT8.svg')

# plt.savefig(f'{output_folder}Supp.svg')
# pg.anova(
#     data=parameter_by_parameter2_for_plotting['eccentricity_smoothed_length'], dv='label', between=['smoothed_length_cat', 'disease_state']).round(3)
# pg.pairwise_tukey(
#     data=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], dv='label', between='disease_state').round(3)
# pg.pairwise_tukey(
#     data=parameter_by_parameter2_for_plotting['smoothed_length_eccentricity'], dv='label', between='eccentricity_cat').round(3)







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

###########################################################





























# # ---------------------Visualise mean length ( > 30 nm)---------------------
# # remove things outside range of interest
# for_plotting = properties[
#     (~properties['sample'].isin(['BSA', 'IgG'])) &
#     (properties['prop_type'] == 'smooth') &
#     (properties['detect'] == 'AT8') &
#     (properties['#locs'] >2) #&
#     (properties['smoothed_length'] > 53) 
#     #(properties['area'] > 2) 
# ].copy()

# #for_plotting.to_csv(f'{output_folder}test.csv')

# for_plotting['scaled_area'] = (for_plotting['area'] * (107/8)**2)/1000
# for_plotting['scaled_perimeter'] = for_plotting['perimeter'] * (107/8)

# ### Mean length >30 nm
# for_plotting_per_replicate = for_plotting.groupby(['disease_state', 'sample', 'capture', 'well_info', 'detect']).mean().reset_index()

# for_plotting_mean = for_plotting_per_replicate.groupby(['disease_state', 'sample','capture', 'detect']).mean().reset_index()


# ###
# # Calculate proportion of spots > threshold intensity
# thresholds = {
#     'length': 250,
#     'scaled_area': 15000/1000,
#     'eccentricity': 0.9, 
#     'perimeter': 600, 
#     'bright': 30, 
#     'dense': 0.7
# }

# for_plotting['length_cat'] = ['long' if val > thresholds['length']
#                                    else ('short' if val <100 else 'medium')for val, detect in for_plotting[['smoothed_length', 'detect']].values]

# proportion_length = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_length['label'] = proportion_length['label'] * 100
# proportion_length = pd.pivot(
#     proportion_length,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='length_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_length_plotting = proportion_length.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# # Calculate proportion of spots > threshold intensity
# for_plotting['area_cat'] = ['large' if val > thresholds['scaled_area']
#                              else 'small' for val, detect in for_plotting[['scaled_area', 'detect']].values]

# proportion_size = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'area_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_size['label'] = proportion_size['label'] * 100
# proportion_size = pd.pivot(
#     proportion_size,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='area_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_size_plotting = proportion_size.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# # perimeter ratio
# for_plotting['perimeter_cat'] = ['long' if val > thresholds['perimeter']
#                                  else 'short' for val, detect in for_plotting[['scaled_perimeter', 'detect']].values]

# proportion_perimeter = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'perimeter_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_perimeter['label'] = proportion_perimeter['label'] * 100
# proportion_perimeter = pd.pivot(
#     proportion_perimeter,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='perimeter_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_perimeter_plotting = proportion_perimeter.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



# #### calulcare proportion of fibrils

# for_plotting['ecc_cat'] = ['fibril' if val > thresholds['eccentricity']
#                               else ('round' if val < 0.7 else 'medium') for val, detect in for_plotting[['eccentricity', 'detect']].values]

# proportion_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_ecc['label'] = proportion_ecc['label'] * 100
# proportion_ecc = pd.pivot(
#     proportion_ecc,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='ecc_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_ecc_plotting = proportion_ecc.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



# #### ecdf

# fitted_ecdf_area = fitting_ecfd_for_plotting(
#     for_plotting, 'AT8', 30000, col='scaled_area')

# fitted_ecdf_smoothed_length = fitting_ecfd_for_plotting(
#     for_plotting, 'AT8', 1000, col='smoothed_length')

# fitted_ecdf_perimeter = fitting_ecfd_for_plotting(
#     for_plotting, 'AT8', 1000, col='scaled_perimeter')


# #### mean length/area of fibrils vs round

# whatever_per_replicate = for_plotting.groupby(
#     ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

# whatever = for_plotting.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()


# whatever2_per_replicate = for_plotting.groupby(
#     ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

# whatever2 = for_plotting.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'length_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()


# ####tar


# length_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'length_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count()['label']).reset_index()
# length_ecc['label'] = length_ecc['label'] * 100



# length_ecc_plotting = length_ecc.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'length_cat', 'ecc_cat']).mean().reset_index()

# # length_ecc_plotting = length_ecc_plotting[
# #     length_ecc_plotting['ecc_cat'] == 'fibril'].copy()
# length_ecc_plotting = length_ecc_plotting[
#     length_ecc_plotting['length_cat'] == 'long'].copy().reset_index()


# ###

# ecc_by_length = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'length_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count()['label']).reset_index()
# ecc_by_length['label'] = ecc_by_length['label'] * 100

# ecc_by_length_plotting = ecc_by_length.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'length_cat', 'ecc_cat']).mean().reset_index()

# ecc_by_length_plotting = ecc_by_length_plotting[
#     ecc_by_length_plotting['ecc_cat'] == 'fibril'].copy()
# # ecc_by_length_plotting = ecc_by_length_plotting[
# #     ecc_by_length_plotting['disease_state'] == 'AD'].copy().reset_index()


# # ----------Reset props df to original----------
# filtered = properties[
#     (~properties['sample'].isin(['BSA', 'IgG'])) &
#     (properties['prop_type'] == 'smooth') &
#     (properties['detect'] == 'AT8')
#     # (properties['smoothed_length'] > 50) &
#     # (properties['area'] > 2) 
# ].copy()

# # map number of localisations from clustered to smooth ROIs


# # Size and shape of 'brightest' (number locs) objects
# filtered = filtered.dropna(subset=['smoothed_label', 'key']) # loose two objects with no smooth label
# locs_dict = dict(properties[properties['prop_type'] == 'cluster'][['key', '#locs']].values)
# filtered['#locs'] = filtered['key'].map(locs_dict)


# filtered.head()[['minor_axis_length', 'major_axis_length', 'orientation',
#                  'well_info', '#locs', 'smoothed_label', ]]


# # Localisation density (number locs / area) - fibrils have larger surface area?
# # --> New column in properties_compiled (smoothed) which has #locs_density


# # length of smallest fibrillar (ecc > 0.9) aggregate for each sample type?





# from scipy.stats import f_oneway
# from statsmodels.stats.multicomp import pairwise_tukeyhsd



# length_ecc = ecc_by_length_plotting[ecc_by_length_plotting['length_cat']!='medium'].copy()
# ecc_by_length_plotting = pd.pivot(
#     ecc_by_length_plotting,
#     index=['capture', 'sample',
#            'detect', 'disease_state'],
#     columns='length_cat',
#     values='label'
# ).fillna(0).reset_index()

# AD_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state']=='AD']
# CRL_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'CRL']
# a= AD_df['long'].values
# b= AD_df['short'].values
# c = CRL_df['long'].values
# d = CRL_df['short'].values


# f_oneway(a, b, c, d)
# joined = [*a, *b, *c, *d]

# df_tukey = pd.DataFrame({'score': joined,
#                    'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# # perform Tukey's test
# tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
#                           groups=df_tukey['group'],
#                           alpha=0.05)

# #display results
# print(tukey)


# length_ecc = length_ecc_plotting[length_ecc_plotting['ecc_cat'] != 'medium'].copy(
# )
# length_ecc = pd.pivot(
#     length_ecc,
#     index=['capture', 'sample',
#            'detect', 'disease_state'],
#     columns='ecc_cat',
#     values='label'
# ).fillna(0).reset_index()

# AD_df = length_ecc[length_ecc['disease_state'] == 'AD']
# CRL_df = length_ecc[length_ecc['disease_state'] == 'CRL']
# a = AD_df['fibril'].values
# b = AD_df['round'].values
# c = CRL_df['fibril'].values
# d = CRL_df['round'].values


# f_oneway(a, b, c, d)
# joined = [*a, *b, *c, *d]

# df_tukey = pd.DataFrame({'score': joined,
#                          'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# # perform Tukey's test
# tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
#                           groups=df_tukey['group'],
#                           alpha=0.05)

# #display results
# print(tukey)

# # # Make main figure

# fig = plt.figure(figsize=(18.4 * cm, 2 * 6.1 * cm))
# gs1 = fig.add_gridspec(nrows=4, ncols=6, wspace=0.95, hspace=0.8)
# axA1 = fig.add_subplot(gs1[0:1, 0:1])
# axA2 = fig.add_subplot(gs1[0:1, 1:2])
# axA3 = fig.add_subplot(gs1[1:2, 0:1])
# axA4 = fig.add_subplot(gs1[1:2, 1:2])
# axB1 = fig.add_subplot(gs1[0:2, 2:3])
# axB2 = fig.add_subplot(gs1[0:2, 3:4])
# axB3 = fig.add_subplot(gs1[0:2, 4:5])
# axB4 = fig.add_subplot(gs1[0:2, 5:6])
# axC1 = fig.add_subplot(gs1[2:4, 0:2])

# axD1 = fig.add_subplot(gs1[2:4, 2:3])
# axD2 = fig.add_subplot(gs1[2:4, 3:4])
# axD3 = fig.add_subplot(gs1[2:4, 4:5])
# axD4 = fig.add_subplot(gs1[2:4, 5:6])

# # axE = fig.add_subplot(gs1[4:6, 0:2])
# # axF = fig.add_subplot(gs1[4:6, 2:4])
# # axG = fig.add_subplot(gs1[4:6, 4:6])


# for ax, label in zip([axA1, axB1, axC1, axD1 ], ['A', 'B', 'C', 'D']):
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
#     ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
#             fontsize=12, va='bottom', fontweight='bold')

# # --------Panel A--------
# axA1.axis('off')
# axA2.axis('off')
# axA3.axis('off')
# axA4.axis('off')

# # --------Panel B--------
# scatbarplot('smoothed_length', 'Mean length [nm]',
#             palette, axB1, for_plotting_mean)
# axB1.set_title('Length')

# scatbarplot('scaled_perimeter', 'Mean perimeter [nm]',
#             palette, axB2, for_plotting_mean)
# axB2.set_title('Perimeter')

# scatbarplot('scaled_area', 'Mean area [x 10$^3$ nm$^2$]',
#             palette, axB3, for_plotting_mean)
# axB3.set_title('Area')

# scatbarplot('eccentricity', 'Mean eccentricity',
#             palette, axB4, for_plotting_mean)
# axB4.set_title('Eccentricity')

# # --------Panel C--------
# ecfd_plot('scaled_perimeter', 'Perimeter',
#           palette, axC1, fitted_ecdf_perimeter)



# # --------Panel D--------
# scatbarplot('long', 'Long [%]',
#             palette, axD1, proportion_length_plotting)

# scatbarplot('long', 'perimeter long [%]',
#             palette, axD2, proportion_perimeter_plotting)

# scatbarplot('large', 'Large [%]',
#             palette, axD3, proportion_size_plotting)


# scatbarplot('fibril', 'Fibrils [%]',
#             palette, axD4, proportion_ecc_plotting)


# # # --------Panel E--------
# # scatbarplot_hue(ycol='smoothed_length', ylabel='length',
# #                 palette=palette, ax=axE, data=whatever)

# # # --------Panel F--------
# # scatbarplot_hue(ycol='scaled_perimeter', ylabel='perimeter',
# #                 palette=palette, ax=axF, data=whatever)

# # # --------Panel G--------
# # scatbarplot_hue(ycol='scaled_area', ylabel='area',
# #                 palette=palette, ax=axG, data=whatever)



# # # Legend for G,H
# # handles, labels = axH.get_legend_handles_labels()
# # by_label = dict(zip(labels, handles))
# # simple_legend = {'AD': by_label['13'],
# #                  'CRL': by_label['9'], 'BSA': by_label['BSA']}

# # axG.legend(simple_legend.values(), simple_legend.keys(),
# #            loc='upper left', frameon=False)
# # axH.legend(simple_legend.values(), simple_legend.keys(),
# #            loc='upper left', frameon=False)


# handles, labels = axC1.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# simple_legend = {'AD': by_label['13'],
#                  'CRL': by_label['9']}

# axC1.legend(simple_legend.values(), simple_legend.keys(),
#                loc='upper left', frameon=False)



# plt.tight_layout()
# plt.savefig(f'{output_folder}Figure3_homogenate_SR.svg')
# plt.show()


# #### supplemental figure

# # fig, axes = plt.subplots(3, 4, figsize=(18.4 * cm, 3 * 6.1 * cm))
# # axes = axes.ravel()
# # plt.subplots_adjust(left=None, bottom=None, right=None,
# #                     top=None, wspace=0.7, hspace=0.1)


# # scatbarplot_hue_ecc(ycol='scaled_area', ylabel='area',
# #                 palette=palette, ax=axes[0], data=whatever)


# # scatbarplot_hue_ecc(ycol='smoothed_length', ylabel='length',
# #                 palette=palette, ax=axes[1], data=whatever)

# # scatbarplot_hue_ecc(ycol='eccentricity', ylabel='perimeter',
# #                 palette=palette, ax=axes[2], data=whatever)

# # scatbarplot_hue_ecc(ycol='scaled_perimeter', ylabel='perimeter',
# #                     palette=palette, ax=axes[3], data=whatever)


# # scatbarplot_hue_length(ycol='scaled_area', ylabel='area',
# #                     palette=palette, ax=axes[4], data=whatever2)

# # scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
# #                        palette=palette, ax=axes[5], data=whatever2)

# # scatbarplot_hue_length(ycol='eccentricity', ylabel='eccentricity',
# #                     palette=palette, ax=axes[6], data=whatever2)

# # scatbarplot_hue_length(ycol='scaled_perimeter', ylabel='perimeter',
# #                     palette=palette, ax=axes[7], data=whatever2)

# # scatbarplot_hue_length('label', 'Fibril [%]',
# #             palette, axes[10], ecc_by_length_plotting)




# # plt.tight_layout()










# # ####supplementak figure####

# def hexbinplotting(colour, ax, data, disease_state):

#     df = data[data['disease_state'] == disease_state].copy()
#     hexs = ax.hexbin(data=df, x='eccentricity',
#                      y='smoothed_length', cmap=colour, vmin=0, vmax=1200)
#     ax.set(ylabel='Length [nm]')
#     ax.set(xlabel='Eccentricity')
#     # sns.kdeplot(data=df, x='smoothed_length', y='eccentricity',
#     #             color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 550)

#     return hexs


# new_colors = ['#ffffff'] + \
#     list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# # Turn this into a new colour map, and visualise it
# cmap = ListedColormap(new_colors)
# cmap


# fig, axes = plt.subplots(3, 2, figsize=(18.4 * cm, 3 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.2)

# hexs0 = hexbinplotting(
#     colour=cmap, ax=axes[0], data=for_plotting, disease_state='AD')
# cb = plt.colorbar(hexs0, ax=axes[0])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[0].set_title('AD', fontsize=8)

# hexs1 = hexbinplotting(
#     colour=cmap, ax=axes[1], data=for_plotting, disease_state='CRL')
# cb = plt.colorbar(hexs1, ax=axes[1])
# cb.set_label('Count', rotation=270, labelpad=15)
# axes[1].set_title('CRL', fontsize=8)

# scatbarplot_hue_ecc(ycol='smoothed_length', ylabel='Mean length [nm]',
#                     palette=palette, ax=axes[2], data=whatever)


# # scatbarplot_hue_ecc(ycol='eccentricity', ylabel='eccentricity',
# #                     palette=palette, ax=axes[2], data=whatever)


# # scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
# #                        palette=palette, ax=axes[5], data=whatever2)

# scatbarplot_hue_length(ycol='eccentricity', ylabel='Mean eccentricity',
#                        palette=palette, ax=axes[3], data=whatever2)

# scatbarplot_hue_ecc('label', 'Long [%]',
#                     palette, axes[4], length_ecc_plotting)


# scatbarplot_hue_length('label', 'Fibril [%]',
#                        palette, axes[5], ecc_by_length_plotting)


# plt.tight_layout()

# plt.savefig(f'{output_folder}Supp.svg')


# ###################################################
# ###################################################

# fitted_ecdf_locs = fitting_ecfd_for_plotting(
#     for_plotting, 'AT8', 100, col='#locs')

# fitted_ecdf_locsdense = fitting_ecfd_for_plotting(
#     for_plotting, 'AT8', 2, col='#locs_density')


# for_plotting['bright_cat'] = ['bright' if val > thresholds['bright']
#                               else ('dim' if val < 10 else 'medium') for val, detect in for_plotting[['#locs', 'detect']].values]

# proportion_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_bright['label'] = proportion_bright['label'] * 100
# proportion_bright = pd.pivot(
#     proportion_bright,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='bright_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_bright_plotting = proportion_bright.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# whatever3_per_replicate = for_plotting.groupby(
#     ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity']].reset_index()

# whatever3 = for_plotting.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'bright_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity']].reset_index()



# #####


# for_plotting['dense_cat'] = ['dense' if val > thresholds['dense']
#                               else 'small' for val, detect in for_plotting[['#locs_density', 'detect']].values]

# proportion_dense = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'dense_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
# proportion_dense['label'] = proportion_dense['label'] * 100
# proportion_dense = pd.pivot(
#     proportion_dense,
#     index=['capture', 'sample', 'slide_position',
#            'detect', 'disease_state'],
#     columns='dense_cat',
#     values='label'
# ).fillna(0).reset_index()

# proportion_dense_plotting = proportion_dense.groupby(
#     ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# def scatbarplot_hue_bright(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
#     order = ['bright', 'dim']
#     hue_order = ['AD', 'CRL']
#     sns.barplot(
#         data=data,
#         x='bright_cat',
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         capsize=0.2,
#         errwidth=2,
#         ax=ax,
#         dodge=True,
#         order=order,
#         hue_order=hue_order,
#         edgecolor='white'
#     )
#     sns.stripplot(
#         data=data,
#         x='bright_cat',
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         ax=ax,
#         edgecolor='#fff',
#         linewidth=1,
#         s=10,
#         order=order,
#         hue_order=hue_order,
#         dodge=True,
#     )

#     # add by chance lines by disease state
#     # for disease, df2 in data.groupby('disease_state'):
#     #     ax.axhline(df2['chance_proportion_coloc'].mean(),
#     #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

#     pairs = [(('bright', 'AD'), ('dim', 'AD')),
#              (('bright', 'CRL'), ('dim', 'CRL')), (('bright', 'AD'), ('bright', 'CRL')), (('dim', 'AD'), ('dim', 'CRL'))]
#     annotator = Annotator(
#         ax=ax, pairs=pairs, data=data, x='bright_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
#     annotator.configure(test='t-test_ind', text_format='star',
#                         loc='inside')
#     annotator.apply_and_annotate()

#     ax.set(ylabel=ylabel)

#     # ax.set_xlabel('')
#     # ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
#     # ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
#     # ax.set_yticks([0, 25, 50, 75, 100])
#     # ax.set_yticklabels(['0', '25', '50', '75', '100'])
#     # ax.annotate('AT8', xy=(0.25, group_label_y),
#     #             xycoords='axes fraction', ha='center')
#     # ax.annotate('T181', xy=(0.75, group_label_y),
#     #             xycoords='axes fraction', ha='center')
#     # trans = ax.get_xaxis_transform()
#     # ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
#     #         color="black", transform=trans, clip_on=False)
#     # ax.plot([0.75, 1.25], [group_line_y, group_line_y],
#     #         color="black", transform=trans, clip_on=False)

#     ax.legend('', frameon=False)


# ecc_by_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'bright_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count()['label']).reset_index()
# ecc_by_bright['label'] = ecc_by_bright['label'] * 100

# ecc_by_bright_plotting = ecc_by_bright.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'ecc_cat']).mean().reset_index()

# ecc_by_bright_plotting = ecc_by_bright_plotting[
#     ecc_by_bright_plotting['ecc_cat'] == 'fibril'].copy()
# # ecc_by_length_plotting = ecc_by_length_plotting[


# length_by_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat', 'bright_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count()['label']).reset_index()
# length_by_bright['label'] = length_by_bright['label'] * 100

# length_by_bright_plotting = length_by_bright.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'length_cat']).mean().reset_index()

# length_by_bright_plotting = length_by_bright_plotting[
#     length_by_bright_plotting['length_cat'] == 'long'].copy()
# # ecc_by_length_plotting = ecc_by_length_plotting[





# bright_by_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'bright_cat']).count(
# )['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count()['label']).reset_index()
# bright_by_ecc['label'] = bright_by_ecc['label'] * 100


# bright_by_ecc_plotting = bright_by_ecc.groupby(
#     ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'ecc_cat']).mean().reset_index()

# # bright_by_ecc_plotting = bright_by_ecc_plotting[
# #     bright_by_ecc_plotting['ecc_cat'] == 'fibril'].copy()
# bright_by_ecc_plotting = bright_by_ecc_plotting[
#     bright_by_ecc_plotting['bright_cat'] == 'bright'].copy().reset_index()






# fig, axes = plt.subplots(6, 2, figsize=(18.4 * cm, 6 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.2)


# ecfd_plot('#locs', 'Number of locs per cluster',
#           palette, axes[0], fitted_ecdf_locs)

# scatbarplot('#locs', 'Mean number of locs per cluster',
#             palette, axes[2], for_plotting_mean)


# scatbarplot('bright', 'Bright [%]',
#             palette, axes[4], proportion_bright_plotting)

# ecfd_plot('#locs_density', 'locs density',
#           palette, axes[1], fitted_ecdf_locsdense)

# scatbarplot('#locs_density', 'Mean locs density',
#             palette, axes[3], for_plotting_mean)


# scatbarplot('dense', 'dense [%]',
#             palette, axes[5], proportion_dense_plotting)

# scatbarplot_hue_bright(ycol='eccentricity', ylabel='Mean eccentricity',
#                        palette=palette, ax=axes[6], data=whatever3)

# scatbarplot_hue_bright(ycol='smoothed_length', ylabel='Length',
#                        palette=palette, ax=axes[7], data=whatever3)

# scatbarplot_hue_bright(ycol='label', ylabel='Fibril %',
#                        palette=palette, ax=axes[8], data=ecc_by_bright_plotting)


# scatbarplot_hue_bright(ycol='label', ylabel='Long %',
#                        palette=palette, ax=axes[9], data=length_by_bright_plotting)


# scatbarplot_hue_ecc(ycol='label', ylabel='Bright %',
#                     palette=palette, ax=axes[10], data=bright_by_ecc_plotting)


# scatbarplot_hue_ecc(ycol='#locs', ylabel='#locs',
#                     palette=palette, ax=axes[11], data=whatever)

# ######
# ######
# ######



# # import statsmodels.api as sm
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from statsmodels.graphics.gofplots import qqplot_2samples
# # x = for_plotting[for_plotting['disease_state'] == 'AD']['scaled_perimeter'].copy()
# # y = for_plotting[for_plotting['disease_state'] == 'CRL']['scaled_perimeter'].copy()
# # pp_x = sm.ProbPlot(x)
# # pp_y = sm.ProbPlot(y)
# # qqplot_2samples(pp_x, pp_y)
# # sns.lineplot(
# #     np.arange(0, 1000),
# #     np.arange(0, 1000),
# #     color='black',
# #     linestyle='--'
# # )
# # plt.show()

# # mean_ecdf = fitted_ecdf_perimeter.groupby(['ecdf', 'disease_state']).mean().reset_index()
# # mean_ecdf = pd.pivot(mean_ecdf, index=['ecdf'], columns='disease_state', values='scaled_perimeter').reset_index()

# # fig, ax = plt.subplots()
# # sns.lineplot(
# #     np.arange(0, 1000),
# #     np.arange(0, 1000),
# #     color='black',
# #     linestyle='--'
# # )
# # sns.scatterplot(
# #     data=mean_ecdf,
# #     x='AD',
# #     y='CRL'
# #     )









# #######


# #########


# # length_ecc = ecc_by_length_plotting[ecc_by_length_plotting['length_cat'] != 'medium'].copy(
# # )
# # ecc_by_length_plotting = pd.pivot(
# #     ecc_by_length_plotting,
# #     index=['capture', 'sample',
# #            'detect', 'disease_state'],
# #     columns='length_cat',
# #     values='label'
# # ).fillna(0).reset_index()

# # AD_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'AD']
# # CRL_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'CRL']
# # a = AD_df['long'].values
# # b = AD_df['short'].values
# # c = CRL_df['long'].values
# # d = CRL_df['short'].values


# # f_oneway(a, b, c, d)
# # joined = [*a, *b, *c, *d]

# # df_tukey = pd.DataFrame({'score': joined,
# #                          'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# # # perform Tukey's test
# # tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
# #                           groups=df_tukey['group'],
# #                           alpha=0.05)

# # #display results
# # print(tukey)


# # length_ecc = length_ecc_plotting[length_ecc_plotting['ecc_cat'] != 'medium'].copy(
# # )
# # length_ecc = pd.pivot(
# #     length_ecc,
# #     index=['capture', 'sample',
# #            'detect', 'disease_state'],
# #     columns='ecc_cat',
# #     values='label'
# # ).fillna(0).reset_index()

# # AD_df = length_ecc[length_ecc['disease_state'] == 'AD']
# # CRL_df = length_ecc[length_ecc['disease_state'] == 'CRL']
# # a = AD_df['fibril'].values
# # b = AD_df['round'].values
# # c = CRL_df['fibril'].values
# # d = CRL_df['round'].values


# # f_oneway(a, b, c, d)
# # joined = [*a, *b, *c, *d]

# # df_tukey = pd.DataFrame({'score': joined,
# #                          'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# # # perform Tukey's test
# # tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
# #                           groups=df_tukey['group'],
# #                           alpha=0.05)

# # #display results
# # print(tukey)
