import matplotlib.pyplot as plt
import numpy as np
import os


def comp_plot(x, comp1, comp2, x_label, y_label, filename, saveOpt = False, logx = False, logy = False):
     
    plt.figure()
    #plt.plot(tc_len, rel_mp_global, marker='o', linestyle='-')
    plt.plot(x, comp1, marker='o', linestyle='-', label='rel_mp_global')
    plt.plot(x, comp2, marker='o', linestyle='-', label='rel_rot_global')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if(logy):
        plt.yscale('log')  
    if(logx):
        plt.xscale('log')  
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if(saveOpt):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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