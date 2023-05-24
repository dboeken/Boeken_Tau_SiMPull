import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.transforms as mtransforms
from loguru import logger
import pingouin as pg
from src.utils import scatbar

logger.info('Import OK')

# =================Set paths=================
output_folder = f'results/5_serum/'
input_path = f'results/5_serum/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# =======Set default plotting parameters=======
cm = 1 / 2.54
font = {'family': 'arial',

        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi']= 300



palette = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
    'F': '#FB4D3D',
    'M': '#345995', 
    'O': 'lightgrey'
}

pvalues = [0.443, 0.00063, 0.00013]

# =========Define functions========


def plot_lda(data, palette, hue='disease_state', style='tissue', ax=None, s=300):
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.scatterplot(
        data=data,
        x='dim1',
        y='dim2',
        hue=hue,
        style=style,
        palette=palette,
        s=s,
        ax=ax
    )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    # plt.legend(loc='upper left', ncol=2, columnspacing=0.1, title='')

    return ax


def plot_eigens(model, X, labels=None, num_labels=5):

    fitted = model.transform(X)
    eigens = model.scalings_
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    n = eigens.shape[0]

    dists = distance_eigens(model, labels=labels)

    fig, ax = plt.subplots()
    for i in range(n):
        plt.arrow(0, 0, eigens[i, 0], eigens[i, 1],
                  color='firebrick', linewidth=2)
    if labels is not None:
        for x, y, label in dists.sort_values('ori_dst', ascending=False).iloc[:num_labels][['xpos', 'ypos', 'label']].values:
            plt.text((x) * 1.15, y * 1.15, label,
                     color='firebrick', ha='center', va='center')

    return fig, ax, dists



# =========Organise data========
DL_spots_summary=pd.read_csv(
    f'{input_path}DL_spots_summary.csv')
SR_spots_mean = pd.read_csv(
    f'{input_path}SR_spots_mean.csv')
lda=pd.read_csv(
    f'{input_path}lda_summary.csv')
eigens_df=pd.read_csv(
    f'{input_path}eigens_df.csv')

# =========Generate figure========

fig = plt.figure(figsize=(18.4*cm, 6.1*cm))
gs1 = fig.add_gridspec(nrows=1, ncols=6, wspace=1.0, hspace=1.0)
axA = fig.add_subplot(gs1[0:2])
axB1 = fig.add_subplot(gs1[2:3])
axB2 = fig.add_subplot(gs1[3:4])
axC = fig.add_subplot(gs1[4:6])

for ax, label in zip([axA, axB1, axC ], ['A', 'B', 'C']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-35/72, -0/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=16, va='bottom', fontweight='bold')
# DL spot count
scatbar(
    dataframe=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'], xcol='disease_state', ycol='spots_count', ax=axA, xorder=['AD', 'CRL', 'BSA'],
    dotpalette=palette, barpalette=palette,
    
)
pairs = [('AD', 'CRL'), ('CRL', 'BSA'), ('AD', 'BSA')]
annotator = Annotator(
    ax=axA, pairs=pairs, data=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'], x='disease_state', y='spots_count', order=['AD', 'CRL', 'BSA'])

annotator.set_pvalues(pvalues)
annotator.annotate()
axA.set_yticks(np.arange(0, 1550, 250))
axA.set_yticklabels(np.arange(0, 1550, 250))
# SR params
axA.set(title='', ylabel='Aggregates per FOV', xlabel='')

# SR params
scatbar(
    dataframe=SR_spots_mean, xcol='disease_state', ycol='smoothed_length', ax=axB1, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs =[('AD', 'CRL')]
)
axB1.set(title='', ylabel='Mean length [nm]', xlabel='')

scatbar(
    dataframe=SR_spots_mean, xcol='disease_state', ycol='smoothed_length', ax=axB2, xorder=['AD', 'CRL'],
    dotpalette=palette, barpalette=palette,
    pairs=[('AD', 'CRL')]
)
axB2.set(title='', ylabel='Mean eccentricity', xlabel='')


# Plot LDA
plot_lda(data=lda, palette=palette, hue='disease_state', style='tissue', ax=axC, s=100)
axC.set(ylim=(-8, 8), xlim=(-22, 17))
axC.legend(frameon=False, ncol=2, columnspacing=-0.1, loc='upper right', bbox_to_anchor=(1.05, 1.1), handletextpad=0.2)
leg = axC.get_legend()
new_labels = ['', 'AD', 'CRL', '', 'Homog.', 'Serum']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)

plt.savefig(f'{output_folder}Figure5_serum.svg')
plt.show()


pg.anova(
    data=DL_spots_summary[DL_spots_summary['detect']=='HT7'], dv='spots_count', between=['disease_state']).round(6)

pg.pairwise_tukey(
    data=DL_spots_summary[DL_spots_summary['detect'] == 'HT7'], dv='spots_count', between=['disease_state']).round(5)
