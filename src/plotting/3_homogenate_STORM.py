###include:

# 1: example images
# 2: cumulative distribution + error bar
# 3: mean length and area
# 4: ratios of large aggregates


from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from loguru import logger
logger.info('Import OK')

input_path = 'data/homogenate_SR_data/properties_compiled.csv'
#input_samplemap = 'raw_data/sample_map.csv'
output_folder = 'results/super-res/summary/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


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
    'AD': '#9A031E',
    'CRL': '#16507E',
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
        s=15,
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


def ecfd_plot(ycol, ylabel, palette, ax, df):
    sns.lineplot(
        data=df.reset_index(),
        y=ycol,
        x='ecdf',
        hue='sample',
        palette=palette,
        ci='sd',
        ax=ax)

    ax.set(ylabel=ylabel, xlabel='Proportion of spots')
    ax.legend(frameon=False)


# ---------------------Visualise mean length ( > 30 nm)---------------------
# remove things outside range of interest
for_plotting = properties[
    (~properties['sample'].isin(['BSA', 'IgG'])) &
    (properties['prop_type'] == 'smooth')
].copy()

for_plotting = for_plotting[for_plotting['detect']=='AT8'].copy()


### Mean length >30 nm
for_plotting_30nm = for_plotting[for_plotting['smoothed_length'] > 30].copy()

for_plotting_30nm_per_replicate = for_plotting_30nm.groupby(['disease_state', 'sample',
                                    'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_30nm_mean = for_plotting_30nm_per_replicate.groupby(['disease_state', 'sample',
                                    'capture', 'detect']).mean().reset_index()


### Mean area 
for_plotting['scaled_area'] = for_plotting['area'] * (107/8)**2
for_plotting_area_per_replicate = for_plotting.groupby(['disease_state', 'sample',
                                    'capture', 'well_info', 'detect']).mean().reset_index()

for_plotting_area_mean = for_plotting_area_per_replicate.groupby(['disease_state', 'sample',
                                    'capture', 'detect']).mean().reset_index()


###
# Calculate proportion of spots > threshold intensity
thresholds = 130
for_plotting_30nm['length_cat'] = ['long' if val > thresholds
                                   else 'short' for val, detect in for_plotting_30nm[['smoothed_length', 'detect']].values]

proportion_length = (for_plotting_30nm.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'length_cat']).count(
)['label'] / for_plotting_30nm.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
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
thresholds = 4000
for_plotting['large_cat'] = ['large' if val > thresholds
                              else 'small' for val, detect in for_plotting[['scaled_area', 'detect']].values]

proportion_size = (for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state', 'large_cat']).count(
)['label'] / for_plotting.groupby(['capture', 'sample', 'slide_position', 'detect', 'disease_state']).count()['label']).reset_index()
proportion_size['label'] = proportion_size['label'] * 100
proportion_size = pd.pivot(
    proportion_size,
    index=['capture', 'sample', 'slide_position',
           'detect', 'disease_state'],
    columns='large_cat',
    values='label'
).fillna(0).reset_index()

proportion_size_plotting = proportion_size.groupby(
    ['capture', 'sample', 'detect', 'disease_state']).mean().reset_index()




#### ecdf

fitted_ecdf_smoothed_length = fitting_ecfd_for_plotting(
    for_plotting_30nm, 'AT8', 1000, col='smoothed_length')





# Make main figure
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.7, hspace=0.1)


scatbarplot('smoothed_length', 'Average length [nm])',
            palette, axes[1], for_plotting_30nm_mean)

scatbarplot('scaled_area', 'Mean area [nm$^2$])',
            palette, axes[2], for_plotting_area_mean)

ecfd_plot('smoothed_length', 'Length',
          palette_repl, axes[3], fitted_ecdf_smoothed_length)


scatbarplot('long', 'Long [%])',
            palette, axes[4], proportion_length_plotting)

scatbarplot('large', 'Large [%])',
            palette, axes[5], proportion_size_plotting)



