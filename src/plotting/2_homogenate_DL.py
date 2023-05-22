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

from microfilm.microplot import microshow

logger.info('Import OK')

# =================Set paths=================
input_path = f'results/2_homogenate_DL/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =======Set default plotting parameters=======
cm = 1/2.54
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

palette = {
    '9': '#345995',
    '159': '#345995',
    '28': '#345995',
    '13': '#F03A47',
    '55': '#F03A47',
    '246': '#F03A47',
    'BSA': '#A9A9A9',
    'AD Mix': '#A9A9A9',

}

palette_DL = {
    'CRL': '#345995',
    'AD': '#F03A47',
    'BSA': '#A9A9A9',
}

def scatbarplot(ycol, ylabel, palette, ax, data):
    order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='disease_state',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.4,
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
        s=5,
        order=order
    )

    ax.set(ylabel=ylabel, xlabel='')
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xticklabels(['AD', 'CRL'])
    pairs = [('AD', 'CRL')]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='disease_state', y=ycol, order=order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    ax.legend('', frameon=False)


def scatbarplot_hue(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['AT8', 'HT7']
    hue_order = ['AD', 'CRL', 'BSA']
    sns.barplot(
        data=data,
        x='detect',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.1,
        errwidth=2,
        ax=ax,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x='detect',
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

    pairs = [(('AT8', 'AD'), ('AT8', 'CRL')), (('HT7', 'AD'), ('HT7', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='detect', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()
    
    ax.set(ylabel=ylabel)
    
    ax.set_xlabel('')
    ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
    ax.set_xticklabels(['AD  ', 'CRL', '    BSA', 'AD  ', 'CRL', '    BSA'])
    ax.tick_params(axis='x', labelrotation=0)

    ax.annotate('AT8', xy=(0.25, group_label_y), xycoords='axes fraction', ha='center')
    ax.annotate('HT7', xy=(0.75, group_label_y), xycoords='axes fraction', ha='center')
    trans = ax.get_xaxis_transform()
    ax.plot([-0.25,0.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
    ax.plot([0.75,1.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)


def scatbarplot_hue2(ycol, ylabel, palette, ax, data, group_label_y=-0.18, group_line_y=-0.05):
    order = ['AT8', 'HT7']
    hue_order = ['AD', 'CRL']
    sns.barplot(
        data=data,
        x='detect',
        y=ycol,
        hue='disease_state',
        palette=palette,
        capsize=0.1,
        errwidth=2,
        ax=ax,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x='detect',
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

    pairs = [(('AT8', 'AD'), ('AT8', 'CRL')), (('HT7', 'AD'), ('HT7', 'CRL'))]
    annotator = Annotator(
        ax=ax, pairs=pairs, data=data, x='detect', y=ycol, order=order, hue='disease_state', hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)

    ax.set_xlabel('')
    ax.set_xticks([-0.2, 0.2, 0.8, 1.2])
    ax.set_xticklabels(['AD', 'CRL', 'AD', 'CRL'])
    ax.tick_params(axis='x', labelrotation=0)

    ax.annotate('AT8', xy=(0.25, group_label_y),
                xycoords='axes fraction', ha='center')
    ax.annotate('HT7', xy=(0.75, group_label_y),
                xycoords='axes fraction', ha='center')
    trans = ax.get_xaxis_transform()
    ax.plot([-0.25, 0.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)
    ax.plot([0.75, 1.25], [group_line_y, group_line_y],
            color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)


def multipanel_scatbarplot(ycol, ylabel, palette, axes, data, left_lims=False, right_lims=False, group_label_y=-0.18, group_line_y=-0.05):
    order = ['AT8', 'HT7']
    for i, detect in enumerate(order):
        if i == 0:
            ax = axes
        else:
            ax = axes.twinx()
        sns.barplot(
            data=data[data['detect'] == detect],
            x='detect',
            y=ycol,
            hue='disease_state',
            palette=palette,
            capsize=0.1,
            errwidth=2,
            ax=ax,
            dodge=True,
            order=order,
            edgecolor='white'
        )
        sns.stripplot(
            data=data[data['detect'] == detect],
            x='detect',
            y=ycol,
            hue='disease_state',
            palette=palette,
            ax=ax,
            edgecolor='#fff',
            linewidth=1,
            s=5,
            order=order,
            dodge=True
        )
        if i == 0:
            # ax.set_title(detect, loc='left')
            ax.set(ylabel=ylabel, xlabel='')
            ax.set_xticks([])
            if left_lims:
                ax.set_ylim(*left_lims)
        else:
            # ax.set_title(detect, loc='right')
            ax.set_ylabel(ylabel, rotation=270, labelpad=15)
            ax.set_xlabel('')
            ax.set_xticks([-0.25, 0, 0.25, 0.75, 1, 1.25])
            ax.set_xticklabels(['AD', 'CRL', 'BSA', 'AD', 'CRL', 'BSA'])
            
            ax.annotate('AT8', xy=(0.25, group_label_y), xycoords='axes fraction', ha='center')
            ax.annotate('HT7', xy=(0.75, group_label_y), xycoords='axes fraction', ha='center')
            trans = ax.get_xaxis_transform()
            ax.plot([-0.25,0.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
            ax.plot([0.75,1.25],[group_line_y, group_line_y], color="black", transform=trans, clip_on=False)
            if left_lims:
                ax.set_ylim(*right_lims)

        pairs = [
            ((detect, 'AD'), (detect, 'CRL')),
        ]
        annotator = Annotator(
            ax=ax, pairs=pairs, data=data, x='detect', y='mean_intensity', order=order, hue='disease_state', hue_order=['AD', 'CRL', 'BSA'])
        annotator.configure(test='t-test_ind', text_format='star',
                            loc='inside', comparisons_correction='bonferroni')
        annotator.apply_and_annotate()
        ax.legend('', frameon=False)


# =========Organise data========
spots_summary = pd.read_csv(f'{input_path}spots_count_summary.csv')
mean_intensity_plotting = pd.read_csv(f'{input_path}mean_intensity.csv')
proportion_intensity_plotting = pd.read_csv(f'{input_path}proportion_intensity.csv')
fitted_ecdf_HT7 = pd.read_csv(f'{input_path}fitted_ecdf_HT7.csv')
fitted_ecdf_HT7.drop([col for col in fitted_ecdf_HT7.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
fitted_ecdf_AT8 = pd.read_csv(f'{input_path}fitted_ecdf_AT8.csv')
fitted_ecdf_AT8.drop([col for col in fitted_ecdf_AT8.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

example_AD = np.load(f'{input_path}example_AD.npy')
example_CRL = np.load(f'{input_path}example_CRL.npy')

# =========Generate figure========
fig = plt.figure(figsize=(18.4 * cm, 3 * 6.1 * cm))

gs1 = fig.add_gridspec(nrows=3, ncols=6, wspace=0.95, hspace=0.3)
axA = fig.add_subplot(gs1[0, 0:2])
axB = fig.add_subplot(gs1[0, 2:4])
axC = fig.add_subplot(gs1[0, 4:6])
axD = fig.add_subplot(gs1[1, 0:2])
axE1 = fig.add_subplot(gs1[1, 2:3])
axE2 = fig.add_subplot(gs1[1, 3:4])
axF = fig.add_subplot(gs1[1, 4:6])
axG = fig.add_subplot(gs1[2, 0:3])
axH = fig.add_subplot(gs1[2, 3:6])

for ax, label in zip([axA, axB, axC, axD, axE1, axF, axG, axH], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-30/72, -3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.05, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')
    
# --------Panel A--------
microim1 = microshow(images=[example_AD],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axA,
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
                               rescale_type='limits', limits=[400, 1000])
axA.set_title('AD', fontsize=8)

# --------Panel B--------
microim1 = microshow(images=[example_CRL],
                               cmaps=['Greys'], flip_map=[True],
                               label_color='black', ax=axB,
                               unit='um', scalebar_size_in_units=10, scalebar_unit_per_pix=0.107, scalebar_font_size=0,
                               rescale_type='limits', limits=[400, 1000])
axB.set_title('CRL', fontsize=8)
    
# --------Panel C--------
scatbarplot_hue('spots_count', 'Number of spots',
                palette_DL, axC, spots_summary, group_line_y=-0.15, group_label_y=-0.22)

# --------Panel D--------
axD.axis('off')

# --------Panel E--------
scatbarplot(ycol='norm_mean_intensity', ylabel='Mean intensity (AU)', palette=palette_DL, ax=axE1, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'AT8'])
axE1.set_title('AT8', fontsize=8)
scatbarplot(ycol='norm_mean_intensity', ylabel='', palette=palette_DL, ax=axE2, data=mean_intensity_plotting[mean_intensity_plotting['detect'] == 'HT7'])
axE2.set_title('HT7', fontsize=8)

# --------Panel F--------
scatbarplot_hue2(ycol='bright', ylabel='Bright spots (%)', palette=palette_DL, ax=axF, data=proportion_intensity_plotting, group_line_y=-0.15, group_label_y=-0.22)

# --------Panel G--------
plot_interpolated_ecdf(
    fitted_ecdfs = fitted_ecdf_HT7[fitted_ecdf_HT7['sample']!= 'BSA'], ycol='norm_mean_intensity',
    huecol='sample', palette=palette, ax=axG, orientation='h')

# --------Panel H--------
plot_interpolated_ecdf(
    fitted_ecdfs = fitted_ecdf_AT8[fitted_ecdf_AT8['sample']!= 'BSA'], ycol='norm_mean_intensity',
    huecol='sample', palette=palette, ax=axH, orientation='h')

# --------Legend for G,H--------
handles, labels = axH.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
simple_legend = {'AD': by_label['13'],
                 'CRL': by_label['9']}

axG.legend(simple_legend.values(), simple_legend.keys(),
               loc='lower right', frameon=False)
axH.legend(simple_legend.values(), simple_legend.keys(),
               loc='lower right', frameon=False)

# --------Fig. admin--------
plt.tight_layout()
plt.savefig(f'{output_folder}Figure2_homogenate_DL.svg')
plt.show()
