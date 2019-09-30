import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "clay",
    "pink",
    "greyish",
    "mint",
    "light cyan",
    "steel blue",
    "forest green",
    "pastel purple",
    "salmon",
    "dark brown",
]

colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

def make_cmap_sns(nsteps=256, bounds=None):
    cmap = gradient_cmap(colors, nsteps=nsteps, bounds=bounds)
    return cmap
#%% General helpers


def make_cmap(number_colors, cmap="cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def make_discrete_cmap(num_colors):
    colors = sns.xkcd_palette(color_names[:5])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colors, num_colors)
    bounds = np.linspace(0, num_colors, num_colors + 1)
    norm = mpl.colors.BoundaryNorm(bounds, num_colors + 1)
    return cmap, bounds, norm


def gradient_cmap(gcolors, nsteps=256, bounds=None):
    """
    Make a colormap that interpolates between a set of colors
    """
    from matplotlib.colors import LinearSegmentedColormap

    ncolors = len(gcolors)
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, gcolors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1.0, 1.0))

    cdict = {
        "red": tuple(reds),
        "green": tuple(greens),
        "blue": tuple(blues),
        "alpha": tuple(alphas),
    }

    cmap = LinearSegmentedColormap("grad_colormap", cdict, nsteps)
    return cmap



def state_correlation(
        z1, z2, k1=None, k2=None, figsize=None, OUTDIR="", FIGURE_STORE=False, title=""
):
    """
    Calculate state correlation
    """
    if k1 is None:
        k1 = z1.max() - z1.min() + 1
    if k2 is None:
        k2 = z2.max() - z2.min() + 1

    if figsize is None:
        figsize = (k1, k2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # C_ij count when z1 == z2
    C = np.zeros(shape=(k1, k2))
    for i in np.arange(k1):
        for j in np.arange(k2):
            C[i, j] = np.logical_and(z1 == i, z2 == j).sum()

    # C_ij normalized (check for non-used states)
    C_pr = C / C.sum(0)
    # C_pr = C.copy()
    # for col in np.arange(k2):
    #    if C[:, col].sum() != 0:
    #        C_pr[:, col] = (C[:, col] / C[:, col]).sum()

    avg_ind = np.sum(C_pr * np.arange(k1)[:, None], axis=0)
    perm = np.argsort(avg_ind)
    ends_ = (avg_ind == 0).sum()
    new_l = perm.copy()
    new_l[0 : k2 - ends_] = perm[ends_:]
    new_l[k2 - ends_ :] = perm[0:ends_]
    # update order
    Cpr = C_pr[:, new_l]


    im = ax.imshow(Cpr, cmap="Reds", vmin=Cpr.min(), vmax=Cpr.max())
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel(r"$K = %d$" % k1)
    ax.set_xlabel(r"$K = %d$" % k2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = plt.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        spacing="uniform",
        format="%.2f",
        ticks=np.linspace(Cpr.min(), Cpr.max(), 5),
    )

    plt.tight_layout()

    if FIGURE_STORE:
        fig.savefig(
            os.path.join(
                OUTDIR,
                "Model_state_overlap_k1_{}_k2_{}_".format(k1, k2) + title + ".pdf",
                )
        )
    else:
        plt.show()
    plt.close()
    return
