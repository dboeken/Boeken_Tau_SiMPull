
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
