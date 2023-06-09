import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

logger.info('Import OK')

# =================Set paths=================
input_path = 'results/5_serum/lda_summary.csv'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# =======Set default plotting parameters=======
font = {'family' : 'arial',
'weight' : 'normal',
'size'   : 8 }
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['figure.dpi'] = 300

cm = 1/2.54

palette = {
    'AD Brain': '#5F0F40',
    'CRL Brain': '#16507E',
    'AD Serum': '#9A031E',
    'CRL Serum': '#7C90A0',
}

# =========Organise data========
lda = pd.read_csv(f'{input_path}')
lda.drop([col for col in lda.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

lda['key'] = [f'{disease} {tissue.capitalize()}' for disease, tissue in lda[['disease_state', 'tissue']].values]

# Add scaled nm conversion
for col in ['perimeter', 'minor_axis_length', 'major_axis_length']:
    lda[col] = lda[col] * (107/8)
lda['area'] = lda['area'] * (107/8)**2
    
value_cols = ['area', 'eccentricity', 'perimeter', 'minor_axis_length', 'major_axis_length', 'smoothed_length', '#locs', 'spots_count']
new_cols = ['Area [nm$^2$]', 'Eccentricity', 'Perimeter [nm]', 'Minor axis\nlength [nm]', 'Major axis\nlength [nm]', 'Length [nm]', '# Locs', 'Spot count']
lda.rename(columns=dict(zip(value_cols, new_cols)), inplace=True)

# =========Generate figure========
pp = sns.pairplot(
    data=lda[['key'] + new_cols],
    hue='key', hue_order=list(palette.keys()), palette=palette,
    plot_kws=dict(alpha=0.55) 
)
pp.fig.set_size_inches(18.4*cm, 18.4*cm)
pp._legend.set_title('Category')
pp._legend.set_bbox_to_anchor((1.1, 0.5))
for ax in pp.axes.flatten():
    ax.set_yticklabels('')
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_xticks([])
plt.savefig(f'{output_folder}S3_pairplot.svg')