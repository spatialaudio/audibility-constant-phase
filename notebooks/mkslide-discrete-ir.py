import numpy as np
import matplotlib.pyplot as plt
import util
import importlib
from matplotlib import rcParams
from os import path, mkdir

importlib.reload(util)

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

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


def plot_dirac_and_hilbert(ax, phi_deg, col1, col2):
    phi = np.deg2rad(phi_deg)
    h = util.discrete_ir_constant_phase(n, phi)
    # 0-degree component
    l1 = ax.stem([0], [1], label=r'$\delta[n]$',
                 markerfmt=col1 + 'o', linefmt=col1, basefmt=col1)
    # 90-degree component (Hilbert)
    l2 = ax.stem(n, h, label='$h_{H}[n]$',
                 markerfmt=col2 + 'o', linefmt=col2, basefmt=col2)
    decorate(ax)
    ax.legend()
    ax.set_ylabel('$h[n]$', color='none')
    filename = 'discrete-ir'
    plt.savefig(dir_fig + filename + '.' + ext, bbox_inches='tight')


def plot_selected_phase_angles(ax, phase_angles_deg, col1, col2, ext):
    for phi_deg in phase_angles_deg:
        phi = np.deg2rad(phi_deg)
        h = util.discrete_ir_constant_phase(n, phi)
        h[n == 0] = None
        # 0-degree component
        l1 = ax.stem([0], [np.cos(phi)], markerfmt=col1 + 'o',
                     linefmt=col1, basefmt=col1)
        # 90-degree component (Hilbert)
        l2 = ax.stem(n, h, markerfmt=col2 + 'o', linefmt=col2, basefmt=col2)
        txt = ax.text(-15, 0.5, r'$\varphi = {:+0.0f}$'.format(phi_deg),
                      fontsize=30, color='lightgray', zorder=0,
                      va='bottom', ha='left')
        decorate(ax)
        ax.set_ylabel('$h[n]$')
        # save figure
        filename = 'discrete-ir-phi{}'.format(phi_deg)
        plt.savefig(dir_fig + filename + '.' + ext, bbox_inches='tight')
        l1.remove()  # remove old ir
        l2.remove()
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
phi = -90

# Selected phase angles
phase_angles_deg = [45, -45, 90, -90, 180, -180]

# colors
col_zero = 'C3'
col_nonzero = 'C0'
col_ani = 'k'

# Plots
ext = 'pdf'

fig, ax = plt.subplots(figsize=(5, 3))
plot_dirac_and_hilbert(ax, phi, col_zero, col_nonzero)

fig, ax = plt.subplots(figsize=(5, 3))
plot_selected_phase_angles(ax, phase_angles_deg, col_ani, col_ani, ext)
