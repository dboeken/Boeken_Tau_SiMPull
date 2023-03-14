from scipy.stats import pearsonr
from lib2to3.pgen2.pgen import DFAState
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import matplotlib

from matplotlib.colors import ListedColormap


# from smma.src.utilities import closest_node, find_pairs
# from smma.src.visualise import plot_colocalisation

from loguru import logger

logger.info('Import OK')

####### TO DO
# add new hex bin plot code
######

input_path = 'data/colocalisation_data/colocalisation_summary.csv'
input_path_spots = 'data/colocalisation_data/colocalisation_spots.csv'
output_folder = 'results/4_colocalisation/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

# Read in summary FOV data
coloc_spots = pd.read_csv(f'{input_path}')
coloc_spots.drop([col for col in coloc_spots.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)

coloc_spots['position'] = coloc_spots['fov'].str[:4]
#coloc_spots['spots_count'] = coloc_spots['spots_count'].fillna(0)

#calculate mean
meandf = coloc_spots.groupby(
    ['sample', 'position', 'channel', 'detect', 'capture', 'layout']).mean().reset_index()

sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}
antibody_dict = {488: 'T181', 641: 'AT8', 'colocalised': 'colocalised'}

#meandf['disease_state']= meandf['sample'].map(sample_dict)
meandf['disease_state'] = meandf['sample'].astype(str).map(sample_dict)

#only colocalised images
filtered_meandf = meandf[meandf['detect'] == 'AT8 r T181 b'].copy()

#remove one set of channels for coloc spots
filtered_coloc = filtered_meandf[filtered_meandf['channel'] == 488].copy()
filtered_coloc = filtered_coloc[['disease_state', 'channel', 'sample',
                                 'capture', 'coloc_spots']].rename(columns={'coloc_spots': 'total_spots'})
filtered_coloc['channel'] = 'colocalised'


for_plotting = pd.concat([filtered_meandf[[
                         'disease_state', 'channel', 'sample', 'capture', 'total_spots']], filtered_coloc])

for_plotting['antibody_state'] = for_plotting['channel'].map(antibody_dict)

palette = {
    'T181': '#17A398',
    'AT8': '#b0185E',
    'colocalised': '#e3b504',
}

palette2 = {
    'T181': 'black',
    'AT8': 'black',
    'colocalised': 'black',
}

sns.set_theme(style="ticks", font_scale=1.6)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.tight_layout()

for x, (capture, df) in enumerate(for_plotting.groupby('capture')):
    f = sns.barplot(
        data=df,
        x='disease_state',
        y='total_spots',
        hue='antibody_state',
        palette=palette,
        ax=axes[x],
        capsize=.15,
        errwidth=0.7,
        saturation=0.8,
        alpha=0.9
    )

    sns.stripplot(
        data=df,
        x='disease_state',
        y='total_spots',
        hue='antibody_state',
        ax=axes[x],
        palette=palette2,
        dodge=True,
        s=5,
        alpha=0.8

    )

    axes[x].set_title(capture)
    axes[x].set(xlabel='Disease State', ylabel='# of spots per FOV')
    #plt.legend(f, ['T181', 'AT8', 'colocalised'])

    #handles, labels = axes[x].get_legend_handles_labels()
    axes[0].legend('')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(5.0, 6.0))

    plt.tight_layout()

    sns.move_legend(
        f, "lower center",
        bbox_to_anchor=(-0.3, -0.45), ncol=3, title=None, frameon=False,
    )
    plt.savefig(f'{output_folder}coloc.svg')


#change error width and capsize, fade out, alpha, change size scatterplots use s =


#plotting proporion of colocalised spots

linestyles = {
    'AD': '--',
    'CRL': 'dotted',
    'BSA': 'solid'
}

for_plotting_proportion = filtered_meandf[[
    'disease_state', 'channel', 'sample', 'capture', 'proportion_coloc', 'chance_proportion_coloc']].copy()
# for_plotting_proportion = filtered_meandf[[
#     'disease_state', 'channel', 'sample', 'capture', 'proportion_coloc']].copy()
for_plotting_proportion['antibody_state'] = for_plotting_proportion['channel'].map(
    antibody_dict)
#sns.set(font_scale=1)
sns.set_theme(style="ticks", font_scale=1.4)

fig, axes = plt.subplots(1, 2, figsize=(7, 4))
fig.tight_layout()
for x, (capture, df) in enumerate(for_plotting_proportion.groupby('capture')):
    f = sns.barplot(
        data=df,
        x='disease_state',
        y='proportion_coloc',
        hue='antibody_state',
        palette=palette,
        ax=axes[x],
        capsize=.15,
        errwidth=0.7,
        saturation=0.8,
        alpha=0.9
    )

    sns.stripplot(
        data=df,
        x='disease_state',
        y='proportion_coloc',
        hue='antibody_state',
        ax=axes[x],
        palette=palette2,
        dodge=True,
        s=5,
        alpha=0.8

    )
    axes[x].set_title(capture)
    axes[x].set(xlabel='Disease State', ylabel='Fraction of spots coloaclised')
    axes[x].set_ylim(0, 100)

    #add by chance lines by disease state
    for disease, df2 in df.groupby('disease_state'):
        axes[x].axhline(df2['chance_proportion_coloc'].mean(),
                        linestyle=linestyles[disease], linewidth=0.7, color='darkgrey')
        logger.info(f"Chance av.: {df2['chance_proportion_coloc'].mean()}")

    axes[0].legend('')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(2.0, 1.0))

    plt.tight_layout()

    sns.move_legend(
        f, "lower center",
        bbox_to_anchor=(-0.3, -0.45), ncol=3, title=None, frameon=False,
    )
plt.savefig(f'{output_folder}proportion_coloc.svg')

for_plotting_proportion.to_csv(f'{output_folder}coloc_proportion.csv')


##################################################################
################### intensity plotting ###########################
##################################################################

for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample',
                                          'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(
    antibody_dict)

melted = pd.melt(for_plotting_intensity, id_vars=['disease_state', 'channel', 'sample', 'capture', 'mean_intensity', 'antibody_state'], value_vars=[
                 'mean_intensity-coloc', 'mean_intensity-noncoloc'], value_name='coloc_intensity', var_name='coloc_status')

melted['coloc_status'] = melted['coloc_status'].str.replace(
    'mean_intensity-', '')
melted['key'] = melted['coloc_status'] + \
    '_' + melted['antibody_state'].astype(str)

palette3 = {
    'coloc_T181': '#668cff',
    'coloc_AT8': '#e3b504',
    'noncoloc_T181': '#17A398',
    'noncoloc_AT8': '#b0185E',
    'colocalised': '#e3b504',
}


sns.set_theme(style="ticks", font_scale=1.4)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.tight_layout()
for x, (capture, df) in enumerate(melted.groupby(['capture'])):
    f = sns.barplot(
        data=df,
        x='disease_state',
        y='coloc_intensity',
        hue='key',
        palette=palette3,
        ax=axes[x],
        capsize=.15,
        errwidth=0.7,
        #width = 15,
        saturation=0.8,
        alpha=0.9,
        hue_order=['noncoloc_T181', 'coloc_T181',
                   'noncoloc_AT8', 'coloc_AT8']
    )

    sns.stripplot(
        data=df,
        x='disease_state',
        y='coloc_intensity',
        hue='key',
        ax=axes[x],
        palette=palette3,
        dodge=True,
        s=5,
        alpha=0.8,
        hue_order=['noncoloc_T181', 'coloc_T181', 'noncoloc_AT8', 'coloc_AT8']

    )
    axes[x].set_title(capture)
    axes[x].set(xlabel='Disease State', ylabel='Mean intensity')
axes[0].legend('')
axes[1].legend()
plt.tight_layout()
sns.move_legend(
    f, "lower center",
    bbox_to_anchor=(-0.4, -0.5), ncol=3, title=None, frameon=False,
)
axes[0].legend('')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[1].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(2.0, 1.0))

plt.tight_layout()

sns.move_legend(
    f, "lower center",
    bbox_to_anchor=(-0.6, -0.6), ncol=3, title=None, frameon=False,
)

#################################################################
##########plot ratio for intensity coloc vs non-coloc############
#################################################################
for_plotting_intensity = filtered_meandf[['disease_state', 'channel', 'sample',
                                          'capture', 'mean_intensity-coloc', 'mean_intensity', 'mean_intensity-noncoloc']].copy()
for_plotting_intensity['antibody_state'] = for_plotting_intensity['channel'].map(
    antibody_dict)
for_plotting_intensity['intensity_ratio'] = for_plotting_intensity['mean_intensity-coloc'] / \
    for_plotting_intensity['mean_intensity-noncoloc']
for_plotting_intensity['log2_intensity_ratio'] = np.log2(
    for_plotting_intensity['intensity_ratio'])


palette3 = {
    'T181': '#17A398',
    'AT8': '#b0185E',
    'colocalised': '#e3b504',
}

#filter for only AD samples
sns.set_theme(style="ticks", font_scale=1.4)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
for x, (capture, df) in enumerate(for_plotting_intensity[for_plotting_intensity['sample'].isin(['13', '55', '246'])].groupby(['capture'])):
    f = sns.barplot(
        data=df,
        x='disease_state',
        y='intensity_ratio',
        hue='antibody_state',
        palette=palette3,
        ax=axes[x],
        capsize=.15,
        errwidth=0.7,
        #width = 15,
        saturation=0.8,
        alpha=0.9,
    )

    sns.stripplot(
        data=df,
        x='disease_state',
        y='intensity_ratio',
        hue='antibody_state',
        ax=axes[x],
        palette=palette2,
        dodge=True,
        s=5,
        alpha=0.8,

    )
    axes[x].set_title(capture)
    axes[x].set(xlabel='Disease State',
                ylabel='intensity ratio \n(colocalised/non-colocalised)')
axes[0].legend('')
axes[1].legend()
plt.tight_layout()
sns.move_legend(
    f, "lower center",
    bbox_to_anchor=(-0.3, -0.4), ncol=3, title=None, frameon=False,
)

axes[0].legend('')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[1].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(2.0, 1.0))

plt.tight_layout()

sns.move_legend(
    f, "lower center",
    bbox_to_anchor=(-0.3, -0.45), ncol=3, title=None, frameon=False,
)
plt.savefig(f'{output_folder}ratio_intensity.svg')

#1 sample t test or 2 way anova

for_plotting_intensity[for_plotting_intensity['sample'].isin(
    ['13', '55', '246'])].to_csv(f'{output_folder}ratio_intensity.csv')




########### stochiometry
input_path_spots = 'data/colocalisation_data/colocalisation_spots.csv'

colocalised_summary = pd.read_csv(f'{input_path_spots}')

colocalised = colocalised_summary[colocalised_summary['coloc?'] == 1].copy()
sample_dict = {'13': 'AD', '9': 'CRL', 'BSA': 'BSA',
               '28': 'CRL', '159': 'CRL', '55': 'AD', '246': 'AD'}
colocalised['disease_state'] = colocalised['sample'].astype(
    str).map(sample_dict)

#sample names strings vs integers
# colocalised[['sample', 'capture', 'detect', 'disease_state']].drop_duplicates()['sample'].unique()


#merge table on mean intenisty of colocalised spots in both channels and update labels
for_plotting = pd.pivot(colocalised, index=['fov', 'layout', 'sample', 'capture', 'pair_id', 'disease_state'], columns=[
                        'channel'], values=['mean_intensity']).reset_index()
for_plotting.columns = [f'{x}' if y ==
                        '' else f'{x}_{y}' for x, y in for_plotting.columns]

palette = sns.color_palette(['#ffffff'] +
                            list(sns.color_palette('magma', n_colors=200).as_hex())[66:])

#from matplotlib.colors import LogNorm, Normalize
for (capture, disease_state), df in for_plotting.groupby(['capture', 'disease_state']):
    fig = sns.jointplot(data=df, x='mean_intensity_641', y='mean_intensity_488',
                        kind="hex", joint_kws={"color": None, 'cmap': ListedColormap(palette)}, gridsize=(120, 120), marginal_kws={'color': 'darkgrey'}),
    #fig.plot(sns.regplot, scatter=False)
    plt.xlim(0, 8000)
    plt.ylim(0, 3000)
    plt.colorbar()
    plt.title(f'{capture} {disease_state}')

for (capture, disease_state), df in for_plotting.groupby(['capture', 'disease_state']):
    matplotlib.pyplot.hexbin(df['mean_intensity_641'], df['mean_intensity_488'],
                             gridsize=120,
                             vmax=6000,
                             vmin=0,
                             #cmap = palette
                             )
    plt.colorbar()
    plt.title(f'{capture} {disease_state}')

# adjust colour of lowest range
# adjust adjust width of bins

for (capture, disease_state), df in for_plotting.groupby(['capture', 'disease_state']):
    sns.regplot(data=df, x='mean_intensity_641', y='mean_intensity_488',)
    # plt.xlim(0, 6000)
    # plt.ylim(0, 6000)
    plt.title(f'{capture} {disease_state}')
    plt.show()
    plt.savefig(
        f'{output_folder}{capture}{disease_state}proportion_coloc.svg')

for (capture, disease_state), df in for_plotting.groupby(['capture', 'disease_state']):
    sns.regplot(data=df, x='mean_intensity_641', y='mean_intensity_488',)
    # plt.xlim(0, 6000)
    # plt.ylim(0, 6000)
    plt.title(f'{capture} {disease_state}')
    plt.show()

    try:

        rval, rstat = pearsonr(
            df['mean_intensity_641'].dropna(), df['mean_intensity_488'].dropna())

        print(f'{capture} {disease_state} {rval} {rstat}')

    except:

        print(
            f"{len(df['mean_intensity_641'].dropna())} {len(df['mean_intensity_488'].dropna())}")



#############


new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cm = ListedColormap(new_colors)
cm

fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={
                         'width_ratios': [4, 4, 0.5]})
# Group by the categories created above, with enumerate to enable easy placement on axes


filtered_disease = for_plotting[for_plotting['disease_state'] == 'AD'].copy()

for i, (group, df) in enumerate(filtered_disease.groupby('capture')):
    # add hex bin, with optional variable assignment to enable colorbar creation
    hexplot = axes[i].hexbin(
        data=df, x='mean_intensity_641', y='mean_intensity_488', cmap=cm)
    # Add kde
    sns.kdeplot(data=df, x='mean_intensity_641', y='mean_intensity_488', color='darkgrey',
                linestyles='--', levels=np.arange(0, 1, 0.2), ax=axes[i])
    # Adjust the usual things
    axes[i].set_xlim(0, 9000)
    axes[i].set_ylim(0, 3000)
    axes[i].set_title(f'AD {group}')
    # add a shared colorbar in the last axes
cb = fig.colorbar(hexplot, cax=axes[2])
cb.set_label('Intensity')
plt.tight_layout()
plt.show()
