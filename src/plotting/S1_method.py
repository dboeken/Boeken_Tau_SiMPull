"""
Generating Figure S1
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from loguru import logger
from microfilm.microplot import microshow
from scipy.optimize import leastsq
from skimage.io import imread
from statannotations.Annotator import Annotator

from src.utils import scatbar

logger.info('Import OK')

# =================Set paths=================
if os.path.exists('data/data_path.txt'):
    root_path = open('data/data_path.txt', 'r').readlines()[0]
else:
    root_path = ''

input_path = f'{root_path}results/S1_method/'
output_folder = f'{root_path}results/S1_method/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
font = {'family': 'arial',
        'weight': 'normal',
        'size': 6.5}
cm = 1/2.54


matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300


palette = {
    'AT8': 'lightgrey',
    'HT7': 'darkgrey',
}

# =========Organise data========
mean_number_peptide_data = pd.read_csv(
    f'{input_path}mean_number_peptide_data.csv')
mean_spots_Xreactivity = pd.read_csv(
    f'{input_path}mean_spots_Xreactivity.csv')
mean_spots_dilution = pd.read_csv(
    f'{input_path}mean_spots_dilution.csv')

example_BSA = imread('data/homogenate_DL_data/example_BSA.tif')
example_BSA = np.mean(example_BSA[10:, :, :], axis=0)

# =======Functions=======

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
        linestyle='', color=palette[detect], capsize=4,
    )

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



# =======Plot figure=======


fig = plt.figure(figsize=(18.4 * cm,  9.2 * cm))
gs1 = fig.add_gridspec(nrows=2, ncols=4, wspace=0.6, hspace=0.55)
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

# ----------Panel A----------

microim1 = microshow(images=[example_BSA],
                     cmaps=['Greys'], flip_map=[True],
                     label_color='black', ax=axA,
                     unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0, rescale_type='limits', limits=[400, 800])

# ----------Panel B----------
scatbar(dataframe=mean_number_peptide_data[mean_number_peptide_data['capture'] == 'AT8'],
        xcol='sample', ycol='spots_count', ax=axB, xorder=['Dimer', 'R1E5'], dotcolor='#36454F', barcolor='darkgrey', pairs=[('Dimer', 'R1E5')])
axB.set(title='AT8 assay', ylabel='Aggregates per FOV', xlabel='')

# ----------Panel C----------

LOD_calculation(mean_spots_dilution, 'AT8', 'AT8', ax= axC)
LOD_calculation(mean_spots_dilution, 'HT7', 'HT7', ax=axC)

axC.set(ylabel='Aggregates per FOV',
        xlabel='Concentration of tau [pg/mL]')

# ----------Panel D----------
scatbar(dataframe=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'AT8'],
        xcol='sample', ycol='spots_count', ax=axD, xorder=['PBS', 'R1E5', 'abeta', 'asyn'], dotcolor='#36454F', barcolor='darkgrey')

pvalues_AT8 = [0.00022, 0.99986, 0.1]
pairs = [('PBS', 'R1E5'), ('PBS', 'abeta'), ('PBS', 'asyn')]
annotator = Annotator(
    ax=axD, pairs=pairs, data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'AT8'], x='sample', y='spots_count', order=['PBS', 'R1E5', 'abeta', 'asyn'])

annotator.set_pvalues(pvalues_AT8)
annotator.annotate()

axD.set(title='AT8 assay', ylabel='Aggregates per FOV', xlabel='')


# ----------Panel E----------
scatbar(dataframe=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'HT7'],
        xcol='sample', ycol='spots_count', ax=axE, xorder=['PBS', 'R1E5', 'abeta', 'asyn'], dotcolor='#36454F', barcolor='darkgrey')
axE.set(title='HT7 assay', ylabel='Aggregates per FOV', xlabel='')

pvalues_HT7 = [0.000001, 0.984, 0.99895]
pairs = [('PBS', 'R1E5'), ('PBS', 'abeta'), ('PBS', 'asyn')]
annotator = Annotator(
    ax=axE, pairs=pairs, data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'HT7'], x='sample', y='spots_count', order=['PBS', 'R1E5', 'abeta', 'asyn'])

annotator.set_pvalues(pvalues_HT7)
annotator.annotate()

# ----------Panel F----------

scatbar(dataframe=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == '6E10'],
        xcol='sample', ycol='spots_count', ax=axF, xorder=['PBS', 'R1E5', 'abeta', 'asyn'], dotcolor='#36454F', barcolor='darkgrey')
axF.set(title='6E10 assay', ylabel='Aggregates per FOV', xlabel='')


pvalues_6E10 = [0.127, 0.00000, 0.9995]
pairs = [('PBS', 'R1E5'), ('PBS', 'abeta'), ('PBS', 'asyn')]
annotator = Annotator(
    ax=axF, pairs=pairs, data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == '6E10'], x='sample', y='spots_count', order=['PBS', 'R1E5', 'abeta', 'asyn'])

annotator.set_pvalues(pvalues_6E10)
annotator.annotate()

# ----------Panel G----------
scatbar(dataframe=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'SC211'],
        xcol='sample', ycol='spots_count', ax=axG, xorder=['PBS', 'R1E5', 'abeta', 'asyn'], dotcolor='#36454F', barcolor='darkgrey')
axG.set(title='SC211 assay', ylabel='Aggregates per FOV', xlabel='')

pvalues_SC211 = [0.9776, 0.999998, 0.01868]
pairs = [('PBS', 'R1E5'), ('PBS', 'abeta'), ('PBS', 'asyn')]
annotator = Annotator(
    ax=axG, pairs=pairs, data=mean_spots_Xreactivity[mean_spots_Xreactivity['capture'] == 'SC211'], x='sample', y='spots_count', order=['PBS', 'R1E5', 'abeta', 'asyn'])

annotator.set_pvalues(pvalues_SC211)
annotator.annotate()


for ax in fig.axes:
    ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()

plt.savefig(f'{output_folder}S1_methods.svg')
plt.show()

#-------2 way Anova + Tukey-------
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
