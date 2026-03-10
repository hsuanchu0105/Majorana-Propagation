import matplotlib.pyplot as plt
import numpy as np
import os


def comp_plot(
    x,
    comps,
    x_label,
    y_label,
    filename=None,
    labels=None,
    saveOpt=False,
    logx=False,
    logy=False,
    markers=None,
    linestyles=None,
):
    """
    x      : 1D array-like for x values
    comps  : list of 1D array-like, each one is a curve to plot
    labels : list of labels for each curve
    """

    plt.figure()

    n = len(comps)

    if labels is None:
        labels = [f"comp{i+1}" for i in range(n)]
    if markers is None:
        markers = ['o'] * n
    if linestyles is None:
        linestyles = ['-'] * n

    for i, y in enumerate(comps):
        plt.plot(
            x,
            y,
            marker=markers[i] if i < len(markers) else 'o',
            linestyle=linestyles[i] if i < len(linestyles) else '-',
            label=labels[i] if i < len(labels) else f"comp{i+1}"
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if saveOpt and filename is not None:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(filename, dpi=200, bbox_inches="tight")

    plt.show()

def plt_hist(hist, nf2, level, x_label = "Majorana monomial length", y_label = "Count", logy = False):  
    x = np.arange(1, nf2 + 1)                        

    plt.figure()
    plt.bar(x, hist, width=0.9, align="center", edgecolor="black")
    plt.xticks(x) 

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if logy:
        plt.yscale("log")
        
    plt.title(f'Length Distribution at {level}th level')
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
                


def plot_length_counts(sampled_levels, counts, nf2, x_label = "Gate", y_label = "Count", logy=False):
    plt.figure()
    for j in range(1, nf2 + 1):
        plt.plot(sampled_levels, counts[:, j - 1], label=f"len={j}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if logy:
        plt.yscale("log")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Legend is only readable for small nf2
    if nf2 <= 12:
        plt.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()