import matplotlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import matplotlib
from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms
from scipy.optimize import leastsq
import pingouin as pg

from microfilm.microplot import microshow
from skimage.io import imread

logger.info('Import OK')

if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}data/homogenate_DL_data/'
output_folder = f'{root_path}results/S1_method/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = {'family': 'arial',
        'weight': 'normal',
        'size': 7}
cm = 1/2.54


matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300


def read_in(input_path):
    spots = pd.read_csv(f'{input_path}')
    spots.drop([col for col in spots.columns.tolist()
                if 'Unnamed: ' in col], axis=1, inplace=True)
    # Drop extra wells collected by default due to grid pattern
    spots.dropna(subset=['sample'], inplace=True)

    spots['fov'] = spots['well_info'].str.split('_').str[0]
    spots['slide_position'] = spots['fov'].str[0:4]

    spots[['capture', 'sample', 'detect']
          ] = spots['sample'].str.split('_', expand=True)

    mean_spots = spots.groupby(['layout', 'channel', 'sample',
                                         'capture', 'detect', 'slide_position']).mean().reset_index()

    return mean_spots


def scatbarplot(ycol, ylabel, ax, data, capture, ):
    #order = ['D10', 'D100', 'D1000', 'D10000', 'Blank', 'BSA']

    sns.barplot(
        data=data[data['capture'] == capture],
        x='sample',
        y=ycol,
        #hue='disease_state',
        #palette=palette,
        capsize=0.2,
        errwidth=2,
        color='darkgrey',
        ax=ax,
        dodge=False,
        #order=order,
    )
    sns.stripplot(
        data=data[data['capture'] == capture],
        x='sample',
        y=ycol,
        #hue='disease_state',
        #palette=palette,
        ax=ax,
        color='#36454F',
        edgecolor='#fff',
        linewidth=1,
        s=5,
        #order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    # ax.tick_params(axis='x', labelrotation=0)
    # ax.set_xticklabels(['AD  ', 'CRL', '    BSA'])
    # pairs = [('AD', 'CRL')]
    # annotator = Annotator(
    #     ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    # annotator.configure(test='t-test_ind', text_format='star',
    #                     loc='inside')
    # annotator.apply_and_annotate()
    # ax.legend('', frameon=False)



mean_spots_Xreactivity = read_in('data/homogenate_DL_data/Xreactivity_spots_per_fov.csv')
mean_spots_dilution = read_in(
    'data/homogenate_DL_data/dilution_spots_per_fov.csv')


mean_spots_dilution = pd.read_csv(
    'data/homogenate_DL_data/dilution_spots_per_fov.csv')
mean_spots_dilution.drop([col for col in mean_spots_dilution.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# fill slides with no spots
mean_spots_dilution['spots_count'] = mean_spots_dilution['spots_count'].fillna(
    0)



mean_spots_dilution[['capture', 'sample', 'detect']
      ] = mean_spots_dilution['sample'].str.split('_', expand=True)

mean_spots_dilution = mean_spots_dilution[mean_spots_dilution['detect'] != 'IgG'].copy(
)


mean_spots_dilution = mean_spots_dilution.groupby(
    ['slide_position', 'capture', 'detect', 'sample', 'layout']).mean().reset_index()

mean_spots_dilution = mean_spots_dilution[
    (mean_spots_dilution['slide_position'] != 'X5Y0') & (
        mean_spots_dilution['layout'] != '1')
].copy()



dilution_dict = {
    'D10': 680000,
    'D100': 68000,
    'D1000': 6800,
    'D10000': 680,
    'BSA': 0
}

mean_spots_dilution['concentration'] = mean_spots_dilution['sample'].map(
    dilution_dict)


df_BSA = mean_spots_dilution[mean_spots_dilution['sample'] == 'BSA'].copy(
)

data = df_BSA.groupby(['capture',
                       'detect', 'layout']).mean().reset_index()

data['key'] = data['capture'] + '_' + data['detect']

# for (capture, detect), df in df_BSA.groupby(['capture', 'detect']):
#     data= df.groupby(['slide_position', 'sample', 'capture', 'detect', 'layout']).mean().reset_index()

dict_BSA = dict(data[['key', 'spots_count']].values)

mean_spots_dilution['key'] = mean_spots_dilution['capture'] + \
    '_' + mean_spots_dilution['detect']

mean_spots_dilution['control'] = mean_spots_dilution['key'].map(
    dict_BSA)

mean_spots_dilution['StoN'] = mean_spots_dilution['spots_count'] / \
    mean_spots_dilution['control']








example_BSA = imread('data/homogenate_DL_data/example_BSA.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)


mean_spots_Xreactivity = mean_spots_Xreactivity[
    (mean_spots_Xreactivity['capture'].isin(['HT7', 'AT8', 'SC211', '6E10'])) &
    (mean_spots_Xreactivity['sample'].isin(['R1E5', 'asyn', 'abeta', 'PBS'])) &
    (mean_spots_Xreactivity['detect'].isin(['HT7', 'AT8', 'SC211', '6E10']))
]

peptide_data = pd.read_csv(f'{root_path}data/peptide_data/spots_per_fov.csv')
peptide_data.drop([col for col in peptide_data.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# Drop extra wells collected by default due to grid pattern
peptide_data.dropna(subset=['sample'], inplace=True)

# expand sample info
peptide_data[['capture', 'sample', 'detect']
      ] = peptide_data['sample'].str.split('_', expand=True)
peptide_data['peptide_data_count'] = peptide_data['spots_count'].fillna(0)

mean_number_peptide_data = peptide_data.groupby(
    ['capture', 'sample', 'slide_position', 'detect']).mean().reset_index()

mean_number_peptide_data[['sample', 'concentration']
                         ] = mean_number_peptide_data['sample'].str.split('-', expand=True)




############


def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D


def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A, B, C, D = p
    err = y-logistic4(x, A, B, C, D)
    return err


def peval(x, p):
    """Evaluated value at x with current parameters."""
    A, B, C, D = p
    return logistic4(x, A, B, C, D)


palette = {
    'AT8': 'lightgrey',
    'HT7': 'darkgrey',

}


def LOD_cal(data, capture, detect):
    df_D10000 = data[(data['capture'] == capture) & (
        data['detect'] == detect) & (data['sample'] == 'D1000')]

    df_BSA = data[(data['capture'] == capture) & (
        data['detect'] == detect) & (data['sample'] == 'BSA')]

    D10000 = df_D10000['spots_count'].values
    BSA = df_BSA['spots_count'].values

    LOB = np.mean(BSA) + 1.645 * np.std(BSA)

    LOD = LOB + 1.645 * np.std(D10000)

    return LOD, LOB


def LOD_calculation(data, capture, detect, ax):
    df = data[(data['capture'] == capture) & (data['detect'] ==
                                              detect)].copy().dropna(subset=['concentration'])

    x = df['concentration'].values

    y_meas = df['spots_count'].values

    # Initial guess for parameters
    p0 = [0, 0.004, 10, 1000]

    # Fit equation using least squares optimization
    plsq = leastsq(residuals, p0, args=(y_meas, x))

    # Plot results
    # sns.scatterplot(
    #     x=x, y=y_meas, marker='o', color='orange', ci="sd")
    ax.plot(
        df.groupby('concentration').mean().reset_index()[
            'concentration'].tolist(),
        df.groupby('concentration').mean().reset_index()[
            'spots_count'].tolist(),
        marker='o', color=palette[detect],
        linewidth=0, markersize=4
    )
    ax.errorbar(
        df.groupby('concentration').mean().reset_index()[
            'concentration'].tolist(),
        df.groupby('concentration').mean().reset_index()[
            'spots_count'].tolist(),
        df.groupby('concentration').std().reset_index()[
            'spots_count'].tolist(),
        df.groupby('concentration').std().reset_index()[
            'spots_count'].tolist(),
        linestyle='', color=palette[detect], capsize=4, #axis=ax
    )
    # sns.plot(
    #     x=x, y=y_meas, marker='o', color='orange', ci="sd")
    sns.lineplot(
        x=np.arange(np.min(x), np.max(x), 10), y=peval(np.arange(np.min(x), np.max(x), 10), plsq[0]), color=palette[detect], label=detect, ax=ax
        ),

    a, b, c, d = plsq[0]

    LOD_spots, LOB_spots = LOD_cal(df, capture, detect)

    LOD_conc = c * np.power(((LOD_spots-a)/(d-LOD_spots)), 1/b)
    LOB_conc = c * np.power(((LOB_spots-a)/(d-LOB_spots)), 1/b)

    return LOD_conc, LOB_conc



# LOD_values = []
# for (detect, capture), df in mean_spots_dilution.groupby(['detect', 'capture']):
#     if detect == capture:
#         LOD_conc, LOB_conc = LOD_calculation(
#             mean_spots_dilution, detect, capture)
#         LOD_values.append([detect, capture, LOD_conc, LOB_conc])
# LOD_values = pd.DataFrame(
#     LOD_values, columns=['Detect', 'Capture', 'LOD', 'LOB'])

# plt.legend()
# plt.xlabel('Concentration of tau [pg/mL]')
# plt.ylabel('Number of aggregates per FOV')
# LOD_values

mean_number_peptide_data.to_csv(
    f'{output_folder}mean_number_peptide_data.csv')
mean_spots_Xreactivity.to_csv(
    f'{output_folder}mean_spots_Xreactivity.csv')
mean_spots_dilution.to_csv(
    f'{output_folder}mean_spots_dilution.csv')








fig = plt.figure(figsize=(18.4 * cm,  9.2 * cm))
gs1 = fig.add_gridspec(nrows=2, ncols=4, wspace=0.6, hspace=0.45)
axA = fig.add_subplot(gs1[0, 0:1])
axB = fig.add_subplot(gs1[0, 1:2])
axC = fig.add_subplot(gs1[0, 2:4])
axD = fig.add_subplot(gs1[1, 0:1])
axE = fig.add_subplot(gs1[1, 1:2])
axF = fig.add_subplot(gs1[1, 2:3])
axG = fig.add_subplot(gs1[1, 3:4])


for ax, label in zip([axA, axB, axC, axD, axE, axF, axG,], ['A', 'B', 'C', 'D', 'E', 'F', 'G']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.05, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')



microim1 = microshow(images=[example_BSA],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axA,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])


scatbarplot(
    data=mean_number_peptide_data[mean_number_peptide_data['capture'] == 'AT8'], ycol='spots_count', ylabel='Spots per FOV',  capture='AT8', ax=axB)

LOD_calculation(mean_spots_dilution, 'AT8', 'AT8', ax= axC)
LOD_calculation(mean_spots_dilution, 'HT7', 'HT7', ax=axC)
# scatbarplot(ycol='StoN', ylabel='Aggregates per FOV',
#             ax=axe, data=mean_spots_dilution, capture='HT7')

# scatbarplot(ycol='StoN', ylabel='Aggregates per FOV',
#             ax=axes[2], data=mean_spots_dilution, capture='AT8')

scatbarplot(ycol='spots_count', ylabel='Aggregates per FOV',
            ax=axD, data=mean_spots_Xreactivity, capture='AT8')

scatbarplot(ycol='spots_count', ylabel='Aggregates per FOV',
            ax=axE, data=mean_spots_Xreactivity, capture='HT7')

scatbarplot(ycol='spots_count', ylabel='Aggregates per FOV',
            ax=axF, data=mean_spots_Xreactivity, capture='6E10')

scatbarplot(ycol='spots_count', ylabel='Aggregates per FOV',
            ax=axG, data=mean_spots_Xreactivity, capture='SC211')


axC.set(ylabel='Aggregates per FOV',
        xlabel='Concentration of tau [pg/mL]')
plt.tight_layout()

plt.savefig(f'{output_folder}S1_methods.svg')
plt.show()


pg.anova(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'AT8'], dv='spots_count', between=['sample']).round(5)
pg.pairwise_tukey(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'AT8'], dv='spots_count', between=['sample']).round(5)


pg.anova(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'HT7'], dv='spots_count', between=['sample']).round(5)
pg.pairwise_tukey(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'HT7'], dv='spots_count', between=['sample']).round(5)


pg.anova(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'SC211'], dv='spots_count', between=['sample']).round(5)
pg.pairwise_tukey(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'SC211'], dv='spots_count', between=['sample']).round(5)

pg.anova(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == '6E10'], dv='spots_count', between=['sample']).round(5)
pg.pairwise_tukey(
    data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == '6E10'], dv='spots_count', between=['sample']).round(5)
