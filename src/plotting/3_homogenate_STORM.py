###include:

# 1: example images
# 2: cumulative distribution + error bar
# 3: mean length and area
# 4: ratios of large aggregates

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from matplotlib.colors import ListedColormap
import matplotlib
from sklearn.decomposition import PCA
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
    'AD': '#FB4D3D',
    'BSA': 'lightgrey',
}


palette_repl = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#FB4D3D',
    '55': '#FB4D3D',
    '246': '#FB4D3D',
    'BSA': 'lightgrey',
    'AD Mix': 'darkgrey',

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
        s=10,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


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
        s=10,
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


def scatbarplot_hue_length(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['long', 'short']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='length_cat',
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
        x='length_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=10,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    # add by chance lines by disease state
    # for disease, df2 in data.groupby('disease_state'):
    #     ax.axhline(df2['chance_proportion_coloc'].mean(),
    #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    pairs = [(('long', 'AD'), ('short', 'AD')),
             (('long', 'CRL'), ('short', 'CRL')), (('long', 'AD'), ('long', 'CRL')), (('short', 'AD'), ('short', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='length_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
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


# ---------------------Visualise mean length ( > 30 nm)---------------------
# remove things outside range of interest
for_plotting = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8') &
    (properties['#locs'] >2) #&
    #(properties['smoothed_length'] > 40) 
    #(properties['area'] > 2) 
].copy()

#for_plotting.to_csv(f'{output_folder}test.csv')

for_plotting['scaled_area'] = (for_plotting['area'] * (107/8)**2)/1000
for_plotting['scaled_perimeter'] = for_plotting['perimeter'] * (107/8)

### Mean length >30 nm
for_plotting_per_replicate = for_plotting.groupby(['disease_state', 'sample', 'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_mean = for_plotting_per_replicate.groupby(['disease_state', 'sample','capture', 'detect']).mean().reset_index()


###
# Calculate proportion of spots > threshold intensity
thresholds = {
    'length': 250,
    'scaled_area': 15000/1000,
    'eccentricity': 0.9, 
    'perimeter': 600, 
    'bright': 30, 
    'dense': 0.7
}

for_plotting['length_cat'] = ['long' if val > thresholds['length']
                                   else ('short' if val <100 else 'medium')for val, detect in for_plotting[['smoothed_length', 'detect']].values]

proportion_length = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_length['label'] = proportion_length['label'] * 100
proportion_length = pd.pivot(
    proportion_length,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='length_cat',
    values='label'
).fillna(0).reset_index()

proportion_length_plotting = proportion_length.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# Calculate proportion of spots > threshold intensity
for_plotting['area_cat'] = ['large' if val > thresholds['scaled_area']
                             else 'small' for val, detect in for_plotting[['scaled_area', 'detect']].values]

proportion_size = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'area_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_size['label'] = proportion_size['label'] * 100
proportion_size = pd.pivot(
    proportion_size,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='area_cat',
    values='label'
).fillna(0).reset_index()

proportion_size_plotting = proportion_size.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


# perimeter ratio
for_plotting['perimeter_cat'] = ['long' if val > thresholds['perimeter']
                                 else 'short' for val, detect in for_plotting[['scaled_perimeter', 'detect']].values]

proportion_perimeter = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'perimeter_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_perimeter['label'] = proportion_perimeter['label'] * 100
proportion_perimeter = pd.pivot(
    proportion_perimeter,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='perimeter_cat',
    values='label'
).fillna(0).reset_index()

proportion_perimeter_plotting = proportion_perimeter.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



#### calulcare proportion of fibrils

for_plotting['ecc_cat'] = ['fibril' if val > thresholds['eccentricity']
                              else ('round' if val < 0.7 else 'medium') for val, detect in for_plotting[['eccentricity', 'detect']].values]

proportion_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_ecc['label'] = proportion_ecc['label'] * 100
proportion_ecc = pd.pivot(
    proportion_ecc,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='ecc_cat',
    values='label'
).fillna(0).reset_index()

proportion_ecc_plotting = proportion_ecc.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()



#### ecdf

fitted_ecdf_area = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 30000, col='scaled_area')

fitted_ecdf_smoothed_length = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 1000, col='smoothed_length')

fitted_ecdf_perimeter = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 1000, col='scaled_perimeter')


#### mean length/area of fibrils vs round

whatever_per_replicate = for_plotting.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

whatever = for_plotting.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'ecc_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()


whatever2_per_replicate = for_plotting.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()

whatever2 = for_plotting.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'length_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity', '#locs']].reset_index()


####tar


length_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'length_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count()['label']).reset_index()
length_ecc['label'] = length_ecc['label'] * 100



length_ecc_plotting = length_ecc.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'length_cat', 'ecc_cat']).mean().reset_index()

# length_ecc_plotting = length_ecc_plotting[
#     length_ecc_plotting['ecc_cat'] == 'fibril'].copy()
length_ecc_plotting = length_ecc_plotting[
    length_ecc_plotting['length_cat'] == 'long'].copy().reset_index()


###

ecc_by_length = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'length_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count()['label']).reset_index()
ecc_by_length['label'] = ecc_by_length['label'] * 100

ecc_by_length_plotting = ecc_by_length.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'length_cat', 'ecc_cat']).mean().reset_index()

ecc_by_length_plotting = ecc_by_length_plotting[
    ecc_by_length_plotting['ecc_cat'] == 'fibril'].copy()
# ecc_by_length_plotting = ecc_by_length_plotting[
#     ecc_by_length_plotting['disease_state'] == 'AD'].copy().reset_index()


# ----------Reset props df to original----------
filtered = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth') &
    (properties['detect'] == 'AT8')
    # (properties['smoothed_length'] > 50) &
    # (properties['area'] > 2) 
].copy()

# map number of localisations from clustered to smooth ROIs


# Size and shape of 'brightest' (number locs) objects
filtered = filtered.dropna(subset=['smoothed_label', 'key']) # loose two objects with no smooth label
locs_dict = dict(properties[properties['prop_type'] == 'cluster'][['key', '#locs']].values)
filtered['#locs'] = filtered['key'].map(locs_dict)


filtered.head()[['minor_axis_length', 'major_axis_length', 'orientation',
                 'well_info', '#locs', 'smoothed_label', ]]


# Localisation density (number locs / area) - fibrils have larger surface area?
# --> New column in properties_compiled (smoothed) which has #locs_density


# length of smallest fibrillar (ecc > 0.9) aggregate for each sample type?





from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



length_ecc = ecc_by_length_plotting[ecc_by_length_plotting['length_cat']!='medium'].copy()
ecc_by_length_plotting = pd.pivot(
    ecc_by_length_plotting,
    index=['capture', 'sample',
           'detect', 'disease_state'],
    columns='length_cat',
    values='label'
).fillna(0).reset_index()

AD_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state']=='AD']
CRL_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'CRL']
a= AD_df['long'].values
b= AD_df['short'].values
c = CRL_df['long'].values
d = CRL_df['short'].values


f_oneway(a, b, c, d)
joined = [*a, *b, *c, *d]

df_tukey = pd.DataFrame({'score': joined,
                   'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
                          groups=df_tukey['group'],
                          alpha=0.05)

#display results
print(tukey)


length_ecc = length_ecc_plotting[length_ecc_plotting['ecc_cat'] != 'medium'].copy(
)
length_ecc = pd.pivot(
    length_ecc,
    index=['capture', 'sample',
           'detect', 'disease_state'],
    columns='ecc_cat',
    values='label'
).fillna(0).reset_index()

AD_df = length_ecc[length_ecc['disease_state'] == 'AD']
CRL_df = length_ecc[length_ecc['disease_state'] == 'CRL']
a = AD_df['fibril'].values
b = AD_df['round'].values
c = CRL_df['fibril'].values
d = CRL_df['round'].values


f_oneway(a, b, c, d)
joined = [*a, *b, *c, *d]

df_tukey = pd.DataFrame({'score': joined,
                         'group': np.repeat(['a', 'b', 'c', 'd'], repeats=3)})

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_tukey['score'],
                          groups=df_tukey['group'],
                          alpha=0.05)

#display results
print(tukey)

# # Make main figure

fig = plt.figure(figsize=(18.4 * cm, 2 * 6.1 * cm))
gs1 = fig.add_gridspec(nrows=4, ncols=6, wspace=0.95, hspace=0.8)
axA1 = fig.add_subplot(gs1[0:1, 0:1])
axA2 = fig.add_subplot(gs1[0:1, 1:2])
axA3 = fig.add_subplot(gs1[1:2, 0:1])
axA4 = fig.add_subplot(gs1[1:2, 1:2])
axB1 = fig.add_subplot(gs1[0:2, 2:3])
axB2 = fig.add_subplot(gs1[0:2, 3:4])
axB3 = fig.add_subplot(gs1[0:2, 4:5])
axB4 = fig.add_subplot(gs1[0:2, 5:6])
axC1 = fig.add_subplot(gs1[2:4, 0:2])

axD1 = fig.add_subplot(gs1[2:4, 2:3])
axD2 = fig.add_subplot(gs1[2:4, 3:4])
axD3 = fig.add_subplot(gs1[2:4, 4:5])
axD4 = fig.add_subplot(gs1[2:4, 5:6])

# axE = fig.add_subplot(gs1[4:6, 0:2])
# axF = fig.add_subplot(gs1[4:6, 2:4])
# axG = fig.add_subplot(gs1[4:6, 4:6])


for ax, label in zip([axA1, axB1, axC1, axD1 ], ['A', 'B', 'C', 'D']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# --------Panel A--------
axA1.axis('off')
axA2.axis('off')
axA3.axis('off')
axA4.axis('off')

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
ecfd_plot('scaled_perimeter', 'Perimeter',
          palette, axC1, fitted_ecdf_perimeter)



# --------Panel D--------
scatbarplot('long', 'Long [%]',
            palette, axD1, proportion_length_plotting)

scatbarplot('long', 'perimeter long [%]',
            palette, axD2, proportion_perimeter_plotting)

scatbarplot('large', 'Large [%]',
            palette, axD3, proportion_size_plotting)


scatbarplot('fibril', 'Fibrils [%]',
            palette, axD4, proportion_ecc_plotting)


# # --------Panel E--------
# scatbarplot_hue(ycol='smoothed_length', ylabel='length',
#                 palette=palette, ax=axE, data=whatever)

# # --------Panel F--------
# scatbarplot_hue(ycol='scaled_perimeter', ylabel='perimeter',
#                 palette=palette, ax=axF, data=whatever)

# # --------Panel G--------
# scatbarplot_hue(ycol='scaled_area', ylabel='area',
#                 palette=palette, ax=axG, data=whatever)



# # Legend for G,H
# handles, labels = axH.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# simple_legend = {'AD': by_label['13'],
#                  'CRL': by_label['9'], 'BSA': by_label['BSA']}

# axG.legend(simple_legend.values(), simple_legend.keys(),
#            loc='upper left', frameon=False)
# axH.legend(simple_legend.values(), simple_legend.keys(),
#            loc='upper left', frameon=False)


handles, labels = axC1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9']}

axC1.legend(simple_legend.values(), simple_legend.keys(),
               loc='upper left', frameon=False)



plt.tight_layout()
plt.savefig(f'{output_folder}Figure3_homogenate_SR.svg')
plt.show()


#### supplemental figure

# fig, axes = plt.subplots(3, 4, figsize=(18.4 * cm, 3 * 6.1 * cm))
# axes = axes.ravel()
# plt.subplots_adjust(left=None, bottom=None, right=None,
#                     top=None, wspace=0.7, hspace=0.1)


# scatbarplot_hue_ecc(ycol='scaled_area', ylabel='area',
#                 palette=palette, ax=axes[0], data=whatever)


# scatbarplot_hue_ecc(ycol='smoothed_length', ylabel='length',
#                 palette=palette, ax=axes[1], data=whatever)

# scatbarplot_hue_ecc(ycol='eccentricity', ylabel='perimeter',
#                 palette=palette, ax=axes[2], data=whatever)

# scatbarplot_hue_ecc(ycol='scaled_perimeter', ylabel='perimeter',
#                     palette=palette, ax=axes[3], data=whatever)


# scatbarplot_hue_length(ycol='scaled_area', ylabel='area',
#                     palette=palette, ax=axes[4], data=whatever2)

# scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
#                        palette=palette, ax=axes[5], data=whatever2)

# scatbarplot_hue_length(ycol='eccentricity', ylabel='eccentricity',
#                     palette=palette, ax=axes[6], data=whatever2)

# scatbarplot_hue_length(ycol='scaled_perimeter', ylabel='perimeter',
#                     palette=palette, ax=axes[7], data=whatever2)

# scatbarplot_hue_length('label', 'Fibril [%]',
#             palette, axes[10], ecc_by_length_plotting)




# plt.tight_layout()










# ####supplementak figure####

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


new_colors = ['#ffffff'] + \
    list(sns.color_palette('magma', n_colors=200).as_hex())[66:]
# Turn this into a new colour map, and visualise it
cmap = ListedColormap(new_colors)
cmap


fig, axes = plt.subplots(3, 2, figsize=(18.4 * cm, 3 * 6.1 * cm))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.2)

hexs0 = hexbinplotting(
    colour=cmap, ax=axes[0], data=for_plotting, disease_state='AD')
cb = plt.colorbar(hexs0, ax=axes[0])
cb.set_label('Count', rotation=270, labelpad=15)
axes[0].set_title('AD', fontsize=8)

hexs1 = hexbinplotting(
    colour=cmap, ax=axes[1], data=for_plotting, disease_state='CRL')
cb = plt.colorbar(hexs1, ax=axes[1])
cb.set_label('Count', rotation=270, labelpad=15)
axes[1].set_title('CRL', fontsize=8)

scatbarplot_hue_ecc(ycol='smoothed_length', ylabel='Mean length [nm]',
                    palette=palette, ax=axes[2], data=whatever)


# scatbarplot_hue_ecc(ycol='eccentricity', ylabel='eccentricity',
#                     palette=palette, ax=axes[2], data=whatever)


# scatbarplot_hue_length(ycol='smoothed_length', ylabel='length',
#                        palette=palette, ax=axes[5], data=whatever2)

scatbarplot_hue_length(ycol='eccentricity', ylabel='Mean eccentricity',
                       palette=palette, ax=axes[3], data=whatever2)

scatbarplot_hue_ecc('label', 'Long [%]',
                    palette, axes[4], length_ecc_plotting)


scatbarplot_hue_length('label', 'Fibril [%]',
                       palette, axes[5], ecc_by_length_plotting)


plt.tight_layout()

plt.savefig(f'{output_folder}Supp.svg')


###################################################
###################################################

fitted_ecdf_locs = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 100, col='#locs')

fitted_ecdf_locsdense = fitting_ecfd_for_plotting(
    for_plotting, 'AT8', 2, col='#locs_density')


for_plotting['bright_cat'] = ['bright' if val > thresholds['bright']
                              else ('dim' if val < 10 else 'medium') for val, detect in for_plotting[['#locs', 'detect']].values]

proportion_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_bright['label'] = proportion_bright['label'] * 100
proportion_bright = pd.pivot(
    proportion_bright,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='bright_cat',
    values='label'
).fillna(0).reset_index()

proportion_bright_plotting = proportion_bright.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


whatever3_per_replicate = for_plotting.groupby(
    ['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity']].reset_index()

whatever3 = for_plotting.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'bright_cat']).mean()[['scaled_area', 'smoothed_length', 'scaled_perimeter', 'eccentricity']].reset_index()



#####


for_plotting['dense_cat'] = ['dense' if val > thresholds['dense']
                              else 'small' for val, detect in for_plotting[['#locs_density', 'detect']].values]

proportion_dense = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'dense_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_dense['label'] = proportion_dense['label'] * 100
proportion_dense = pd.pivot(
    proportion_dense,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='dense_cat',
    values='label'
).fillna(0).reset_index()

proportion_dense_plotting = proportion_dense.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()


def scatbarplot_hue_bright(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['bright', 'dim']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='bright_cat',
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
        x='bright_cat',
        y=ycol,
        hue='disease_state',
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=10,
        order=order,
        hue_order=hue_order,
        dodge=True,
    )

    # add by chance lines by disease state
    # for disease, df2 in data.groupby('disease_state'):
    #     ax.axhline(df2['chance_proportion_coloc'].mean(),
    #                linestyle=linestyles[disease], linewidth=1.2, color='#4c4c52')

    pairs = [(('bright', 'AD'), ('dim', 'AD')),
             (('bright', 'CRL'), ('dim', 'CRL')), (('bright', 'AD'), ('bright', 'CRL')), (('dim', 'AD'), ('dim', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='bright_cat', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
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


ecc_by_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'bright_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count()['label']).reset_index()
ecc_by_bright['label'] = ecc_by_bright['label'] * 100

ecc_by_bright_plotting = ecc_by_bright.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'ecc_cat']).mean().reset_index()

ecc_by_bright_plotting = ecc_by_bright_plotting[
    ecc_by_bright_plotting['ecc_cat'] == 'fibril'].copy()
# ecc_by_length_plotting = ecc_by_length_plotting[


length_by_bright = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat', 'bright_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'bright_cat']).count()['label']).reset_index()
length_by_bright['label'] = length_by_bright['label'] * 100

length_by_bright_plotting = length_by_bright.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'length_cat']).mean().reset_index()

length_by_bright_plotting = length_by_bright_plotting[
    length_by_bright_plotting['length_cat'] == 'long'].copy()
# ecc_by_length_plotting = ecc_by_length_plotting[





bright_by_ecc = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat', 'bright_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'ecc_cat']).count()['label']).reset_index()
bright_by_ecc['label'] = bright_by_ecc['label'] * 100


bright_by_ecc_plotting = bright_by_ecc.groupby(
    ['capture', 'sample', 'detect', 'disease_state', 'bright_cat', 'ecc_cat']).mean().reset_index()

# bright_by_ecc_plotting = bright_by_ecc_plotting[
#     bright_by_ecc_plotting['ecc_cat'] == 'fibril'].copy()
bright_by_ecc_plotting = bright_by_ecc_plotting[
    bright_by_ecc_plotting['bright_cat'] == 'bright'].copy().reset_index()






fig, axes = plt.subplots(6, 2, figsize=(18.4 * cm, 6 * 6.1 * cm))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.2)


ecfd_plot('#locs', 'Number of locs per cluster',
          palette, axes[0], fitted_ecdf_locs)

scatbarplot('#locs', 'Mean number of locs per cluster',
            palette, axes[2], for_plotting_mean)


scatbarplot('bright', 'Bright [%]',
            palette, axes[4], proportion_bright_plotting)

ecfd_plot('#locs_density', 'locs density',
          palette, axes[1], fitted_ecdf_locsdense)

scatbarplot('#locs_density', 'Mean locs density',
            palette, axes[3], for_plotting_mean)


scatbarplot('dense', 'dense [%]',
            palette, axes[5], proportion_dense_plotting)

scatbarplot_hue_bright(ycol='eccentricity', ylabel='Mean eccentricity',
                       palette=palette, ax=axes[6], data=whatever3)

scatbarplot_hue_bright(ycol='smoothed_length', ylabel='Length',
                       palette=palette, ax=axes[7], data=whatever3)

scatbarplot_hue_bright(ycol='label', ylabel='Fibril %',
                       palette=palette, ax=axes[8], data=ecc_by_bright_plotting)


scatbarplot_hue_bright(ycol='label', ylabel='Long %',
                       palette=palette, ax=axes[9], data=length_by_bright_plotting)


scatbarplot_hue_ecc(ycol='label', ylabel='Bright %',
                    palette=palette, ax=axes[10], data=bright_by_ecc_plotting)


scatbarplot_hue_ecc(ycol='#locs', ylabel='#locs',
                    palette=palette, ax=axes[11], data=whatever)

######
######
######



# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.graphics.gofplots import qqplot_2samples
# x = for_plotting[for_plotting['disease_state'] == 'AD']['scaled_perimeter'].copy()
# y = for_plotting[for_plotting['disease_state'] == 'CRL']['scaled_perimeter'].copy()
# pp_x = sm.ProbPlot(x)
# pp_y = sm.ProbPlot(y)
# qqplot_2samples(pp_x, pp_y)
# sns.lineplot(
#     np.arange(0, 1000),
#     np.arange(0, 1000),
#     color='black',
#     linestyle='--'
# )
# plt.show()

# mean_ecdf = fitted_ecdf_perimeter.groupby(['ecdf', 'disease_state']).mean().reset_index()
# mean_ecdf = pd.pivot(mean_ecdf, index=['ecdf'], columns='disease_state', values='scaled_perimeter').reset_index()

# fig, ax = plt.subplots()
# sns.lineplot(
#     np.arange(0, 1000),
#     np.arange(0, 1000),
#     color='black',
#     linestyle='--'
# )
# sns.scatterplot(
#     data=mean_ecdf,
#     x='AD',
#     y='CRL'
#     )









#######


#########


# length_ecc = ecc_by_length_plotting[ecc_by_length_plotting['length_cat'] != 'medium'].copy(
# )
# ecc_by_length_plotting = pd.pivot(
#     ecc_by_length_plotting,
#     index=['capture', 'sample',
#            'detect', 'disease_state'],
#     columns='length_cat',
#     values='label'
# ).fillna(0).reset_index()

# AD_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'AD']
# CRL_df = ecc_by_length_plotting[ecc_by_length_plotting['disease_state'] == 'CRL']
# a = AD_df['long'].values
# b = AD_df['short'].values
# c = CRL_df['long'].values
# d = CRL_df['short'].values


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
