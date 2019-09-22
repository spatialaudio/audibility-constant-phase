import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib import rcParams
from os import path, mkdir

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

rcParams['figure.figsize'] = [5, 3]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'sans serif'
rcParams['font.stretch'] = 'condensed'
rcParams['font.size'] = 13
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def initial_plot(phi_deg, ax, col1, col2, extension='pdf'):
    phi = np.deg2rad(phi_deg)
    h = util.discrete_ir_constant_phase(n, phi)
    h[n == 0] = None
    # 0-degree component
    l1 = ax.stem([0], [np.cos(phi)], markerfmt=col1 + 'o',
                 linefmt=col1, basefmt=col1)
    # 90-degree component (Hilbert)
    l2 = ax.stem(n, h, markerfmt=col2 + 'o', linefmt=col2, basefmt=col2)
    decorate(ax)
    # save figure
    filename = 'discrete-ir'
    plt.savefig(dir_fig + filename + '.' + extension, bbox_inches='tight')
    return l1, l2


def update_plot_and_save(phase_angles_deg, ax, l1, l2, col1, col2, extension):
    for phi_deg in phase_angles_deg:
        phi = np.deg2rad(phi_deg)
        h = util.discrete_ir_constant_phase(n, phi)
        h[n == 0] = None
        l1.remove()  # remove old ir
        l2.remove()
        # 0-degree component
        l1 = ax.stem([0], [np.cos(phi)], markerfmt=col1 + 'o',
                     linefmt=col1, basefmt=col1)
        # 90-degree component (Hilbert)
        l2 = ax.stem(n, h, markerfmt=col2 + 'o', linefmt=col2, basefmt=col2)
        txt = ax.text(-15, 0.5, r'$\varphi = {:+0.0f}$'.format(phi_deg),
                      fontsize=30, color='lightgray', zorder=0,
                      va='bottom', ha='left')
        decorate(ax)
        # save figure
        filename = 'discrete-ir-phi{}'.format(phi_deg)
        plt.savefig(dir_fig + filename + '.' + extension, bbox_inches='tight')
        txt.remove()  # remove old annotation


def decorate(ax):
    ax.set_xlabel('$n$ / sample')
    ax.set_ylabel('$h[n]$')
    ax.set_xlim(nmin - 0.5, nmax + 0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(color='lightgray')


# Discrete time indices
nmin, nmax = -16, 16
n = np.arange(nmin, nmax + 1)

# Initial plot
phi = -45

# Speical cases
phase_angles_deg = [45, -45, 90, -90, 180, -180]

# colors
col_zero = 'C3'
col_nonzero = 'C0'
col_ani = 'k'

# Plot
ext = 'pdf'
fig, ax = plt.subplots(figsize=(5, 3))
l1, l2 = initial_plot(phi, ax, col_zero, col_nonzero, ext)
update_plot_and_save(phase_angles_deg, ax, l1, l2, col_ani, col_ani, ext)
