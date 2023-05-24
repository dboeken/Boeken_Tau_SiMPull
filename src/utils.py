"""
Containing utility functions for preprocessing and plotting data
"""


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from loguru import logger
from statannotations.Annotator import Annotator

logger.info('Import OK')

# =======Plotting functionality=======
def scatbar(dataframe, xcol, ycol, ax, xorder, dotpalette=None, barpalette=None, dotcolor=None, barcolor=None, hue_col=None, hue_order=None, capsize=0.2, errwidth=2, dotsize=5, pairs=None, comparisons_correction=None, groups=None, group_label_y=-0.18, group_line_y=-0.05, edgecolor=None):
    """Generate barplot overlayed with individual biological replicates

    Args:
        dataframe (dataframe): _description_
        xcol (str): Values to be plotted on the x axis
        ycol (str): Values to be plotted on the y axis
        ax (str): ubpanel number in multipanel figures.
        xorder (list): Order of samples on the x axis
        dotpalette (dictionary, optional): Palette to change colour of the dots. Defaults to None.
        barpalette (dictionary, optional): Palette to change colour of the bars. Defaults to None.
        dotcolor (str, optional): Single colour of dots. Defaults to None.
        barcolor (str, optional): Single colour of bars. Defaults to None.
        hue_col (str, optional): Column on which the hue is based. Defaults to None.
        hue_order (list, optional): Hue order. Defaults to None.
        capsize (float, optional): Size of the error bar caps. Defaults to 0.2.
        errwidth (int, optional): Width of the error bars. Defaults to 2.
        dotsize (int, optional): Size of the dots. Defaults to 5.
        pairs (list, optional): (list of) pairs to perform statistical analysis. Defaults to None.
        comparisons_correction (str, optional): Statistical multiple comparison correction. Defaults to None.
        groups (str, optional): Sample groups, e.g. antibody. Defaults to None.
        group_label_y (float, optional): Label of group. Defaults to -0.18.
        group_line_y (float, optional): Line to indicate group. Defaults to -0.05.
        edgecolor (str, optional): Colour of the edge of bars. Defaults to None.
    """

    if groups:
        dodge = True
    else:
        dodge = False

    sns.barplot(
        data=dataframe,
        x=xcol,
        y=ycol,
        order=xorder,
        hue=hue_col,
        hue_order=hue_order,
        palette=barpalette,
        color=barcolor,
        capsize=capsize,
        errwidth=errwidth,
        ax=ax,
        dodge=dodge,
        edgecolor=edgecolor
    )
    sns.stripplot(
        data=dataframe,
        x=xcol,
        y=ycol,
        order=xorder,
        hue=hue_col,
        hue_order=hue_order,
        palette=dotpalette,
        color=dotcolor,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=dotsize,
        dodge=dodge
    )

    if pairs:
        annotator = Annotator(
            ax=ax, pairs=pairs, data=dataframe, x=xcol, y=ycol, order=xorder, hue=hue_col, hue_order=hue_order)
        annotator.configure(test='t-test_ind', text_format='star',
                            loc='inside', comparisons_correction=comparisons_correction)
        annotator.apply_and_annotate()

    if groups:

        ax.set_xlabel('')
        pos = []
        if len(hue_order) == 2:
            pos = [-0.2, 0.2, 0.8, 1.2]
        if len(hue_order) == 3:
            pos = [-0.25, 0, 0.25, 0.75, 1, 1.25]

        ax.set_xticks(pos)
        ax.set_xticklabels(hue_order * len(groups))
        ax.tick_params(axis='x', labelrotation=0)

        for group, xpos in zip(groups, [0.25, 0.75]):
            ax.annotate(group, xy=(xpos, group_label_y),
                        xycoords='axes fraction', ha='center')

        trans = ax.get_xaxis_transform()
        for i, group in enumerate(groups):
            ax.plot([pos[0+i*len(hue_order)], pos[len(hue_order)-1+i*len(hue_order)]],
                    [group_line_y, group_line_y], color="black", transform=trans, clip_on=False)

    ax.legend('', frameon=False)



def plot_interpolated_ecdf(fitted_ecdfs, ycol, huecol, palette, ax=None, orientation=None):
    """Generate fitted cumulative distribution

    Args:
        fitted_ecdfs (dataframe): Dataframe containing interpolated cumulative distributions
        ycol (str): Value of the cumulative distribution
        huecol (str): Column on which the hue is based
        palette (dictionary): Dictionary containing the palette
        ax (str, optional): Subpanel number in multipanel figures. Defaults to None.
        orientation (str, optional): Orientation of the plot, typically h. Defaults to None.

    Returns:
        _type_: _description_
    """

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

# =======Analysis functionality=======
def fit_ecdf(x):
    """Function to fit ecdfs for cumulative distributions

    Args:
        x (array): Array to be sorted for fitting the ecdf

    Returns:
        _type_: Sorted array
    """
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result


def sample_ecdf(df, value_cols, num_points=100, method='nearest', order=False):
    """Function to interpolate sample ecdf

    Args:
        df (dataframe): dataframe to compute cumulative distribution
        value_cols (_type_): Column names to fit ecdf on
        num_points (int, optional): Percentage. Defaults to 100.
        method (str, optional): Fitting method. Defaults to 'nearest'.
        order (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

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
    """Fitting ecdf for cumulative distrubutions with multiple replicates

    Args:
        df_intensity (dataframe): Dataframe to compute the cumulative distribution
        detect (str): Detection antibody used in experiment
        maxval (_type_): Maximum signal level
        col (_type_): _description_

    Returns:
        _type_: _description_
    """
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


def plot_hexbin(data, ax, xcol, ycol, vmin, vmax, colour, filter_col=None, filter_val=None, kdeplot=None):
    if filter_col and filter_val:
        data = data[data[filter_col] == filter_val].copy()
    hexs = ax.hexbin(data=data, x=xcol,
                     y=ycol, cmap=colour, vmin=vmin, vmax=vmax)
    if kdeplot:
        sns.kdeplot(data=data, x=xcol, y=ycol,
                    color='darkgrey', linestyles='--', levels=np.arange(0, 1, 0.2), ax=ax)
    return hexs

