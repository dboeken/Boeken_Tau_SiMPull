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

# def scatbarplot(ycol, ylabel, palette, ax, data):
#     order = ['AD', 'CRL']
#     sns.barplot(
#         data=data,
#         x='disease_state',
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         capsize=0.2,
#         errwidth=2,
#         ax=ax,
#         dodge=False,
#         order=order,
#     )
#     sns.stripplot(
#         data=data,
#         x='disease_state',
#         y=ycol,
#         hue='disease_state',
#         palette=palette,
#         ax=ax,
#         edgecolor='#fff',
#         linewidth=1,
#         s=15,
#         order=order
#     )

#     ax.set(ylabel=ylabel, xlabel='')
#     pairs = [('AD', 'CRL')]
#     annotator = Annotator(
#         ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
#     annotator.configure(test='t-test_ind', text_format='star',
#                         loc='inside')
#     annotator.apply_and_annotate()
#     ax.legend('', frameon=False)


# def sample_ecdf(df, value_cols, num_points=100, method='nearest', order=False):

#     test_vals = pd.DataFrame(
#         np.arange(0, 1.01, (1/num_points)), columns=['ecdf'])
#     test_vals['type'] = 'interpolated'
#     interpolated = test_vals.copy()
#     for col in value_cols:
#         test_df = df.dropna().drop_duplicates(subset=[col])
#         ecdf = fit_ecdf(test_df[col])
#         test_df['ecdf'] = ecdf(
#             test_df.dropna().drop_duplicates(subset=[col])[col])
#         combined = pd.concat([test_df.sort_values('ecdf').dropna(), test_vals])
#         combined = combined.set_index('ecdf').interpolate(
#             method=method, order=order).reset_index()
#         interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[
#             col].tolist()

#     return interpolated


# def fitting_ecfd_for_plotting(df_intensity, detect, maxval, col='mean_intensity'):
#     fitted_ecdfs = []
#     for (capture, sample, position), df in df_intensity.groupby(['capture', 'sample', 'slide_position']):
#         filtered_df = df[df[col] < maxval]
#         fitted_ecdf = sample_ecdf(filtered_df, value_cols=[
#             col], method='nearest', order=False)
#         fitted_ecdf['sample'] = sample
#         fitted_ecdf['capture'] = capture
#         fitted_ecdf['slide_position'] = position
#         fitted_ecdf['detect'] = detect
#         fitted_ecdfs.append(fitted_ecdf)

#     fitted_ecdfs = pd.concat(fitted_ecdfs)
#     return fitted_ecdfs


# def ecfd_plot(ycol, ylabel, palette, ax, df):
#     sns.lineplot(
#         data=df.reset_index(),
#         y=ycol,
#         x='ecdf',
#         hue='sample',
#         palette=palette,
#         ci='sd',
#         ax=ax)

#     ax.set(ylabel=ylabel, xlabel='Proportion of spots')
#     ax.legend(frameon=False)


