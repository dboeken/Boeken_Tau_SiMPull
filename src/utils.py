"""
Containing utility functions for preprocessing and plotting data
"""


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from statannotations.Annotator import Annotator

logger.info('Import OK')

# =======Plotting functionality=======
def scatbar(dataframe, xcol, ycol, ax, xorder, dotpalette=None, barpalette=None, dotcolor=None, barcolor=None, hue_col=None, hue_order=None, capsize=0.2, errwidth=2, dotsize=5, pairs=None, comparisons_correction=None):

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
        dodge=False,
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
    )

    if pairs:
        annotator = Annotator(
            ax=ax, pairs=pairs, data=dataframe, x=xcol, y=ycol, order=xorder)
        annotator.configure(test='t-test_ind', text_format='star',
                            loc='inside', comparisons_correction=comparisons_correction)
        annotator.apply_and_annotate()

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
