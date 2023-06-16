import json
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from itertools import cycle

DATA = 'data'
FIGURES = 'figures/'

rc_fonts = {
    "font.family": "serif",
    "font.size": 22,
    "text.usetex": True,
    "text.latex.preamble": r"""
    \usepackage{libertine}
    \usepackage{amsmath}
    \usepackage[libertine]{newtxmath}
    """
}

mpl.rcParams.update(rc_fonts)


FILES = {
    "Kaplan": {
        "Custom and Trained": "custom_nofreeze_kaplan.json",
        "Custom and Frozen": "custom_freeze_kaplan.json",
        "Random and Trained": "random_nofreeze_kaplan.json",
        "Random and Frozen": "random_freeze_kaplan.json",
    },
    "Chinchilla": {
        "Custom and Trained": "custom_nofreeze_chinchilla.json",
        "Custom and Frozen": "custom_freeze_chinchilla.json",
        "Random and Trained": "random_nofreeze_chinchilla.json",
        "Random and Frozen": "random_freeze_chinchilla.json",
    }
}

YLIMS = {
    #"kaplan": (1.9, 3.2),
    "Chinchilla": (1.9, 3.6),
    "Kaplan": (1.9, 3.6)
}


def save(filename):
    file = os.path.join(FIGURES, filename.lower())
    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")


def plot_all_training():
    plt.figure(figsize=(9, 6))
    colors = cycle(["tab:orange", "tab:blue"])

    for scaling, files in FILES.items():
        color = next(colors)
        for i, file in enumerate(files.values()):
            with open(os.path.join(DATA, file), 'r') as f:
                content = json.load(f)
        
            _, _, loss = list(zip(*content))
            loss = loss[:-1]
            print("start", loss[0])
            
            if i != 0:
                label = None
            else:
                label = scaling + " Scaling"
            plt.plot(np.linspace(0, 10, len(loss)), loss, label=label, color=color)
    
    plt.xlabel(r"$\textbf{Epochs}$")
    plt.ylabel(r"$\textbf{Loss}$")
    
    plt.legend(prop=dict(size=24, weight="bold"))
    plt.ylim(YLIMS[scaling])
    save(f"training_loss_all.pdf")
    #plt.show()

def plot_loss(scaling):
    plt.figure(figsize=(9, 6))

    files = FILES[scaling]
    for name, file in files.items():
        with open(os.path.join(DATA, file), 'r') as f:
            content = json.load(f)
    
        _, _, loss = list(zip(*content))
        loss = loss[:-1]
        print("start", loss[0])
    
        plt.plot(np.linspace(0, 10, len(loss)), loss, label=name)
    
    plt.xlabel(r"$\textbf{Epochs}$")
    plt.ylabel(r"$\textbf{Loss}$")
    
    plt.legend(prop=dict(size=24, weight="bold"))
    plt.ylim(YLIMS[scaling])
    save(f"training_loss_{scaling}.pdf")
    #plt.show()

if __name__ == "__main__":
    plot_loss("Kaplan")
    plot_loss("Chinchilla")
    plot_all_training()
