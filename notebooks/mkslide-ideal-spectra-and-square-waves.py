import numpy as np
import matplotlib.pyplot as plt
import util
import importlib
from scipy.signal import kaiser
from matplotlib import rcParams, cm
from os import path, mkdir
from util import db

importlib.reload(util)

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

rcParams['figure.figsize'] = [12, 5]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'sans serif'
rcParams['font.stretch'] = 'condensed'
rcParams['font.size'] = 12
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'


def plot_magnitude(ax, fmin, fmax, H_mag, color):
    ax.hlines(db(H_mag), fmin, fmax, colors=color)
    ax.set_yticks([-10, 0, 10])
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(-12, 12)
    ax.set_xscale('log')
    ax.set_xlabel('$f$ / Hz')
    ax.set_ylabel('Magnitude / dB')
    ax.grid(True)
    # annotation
    flabel = np.sqrt(fmin * fmax)
    ax.text(flabel, 0, 'all-pass', fontsize=14,
            ha='center', va='bottom', color='k')


def plot_phase(ax, fmin, fmax, phase_angles, colors, annotations):
    num_phase = len(phase_angles)
    flabel = np.sqrt(fmin * fmax)
    fontsize = 14
    voffset = -5
    for i, phi in enumerate(phase_angles):
        ax.hlines(phi, fmin, fmax, color=colors(i / num_phase))
        if phi in annotations:
            ax.text(flabel, phi + voffset, annotations[phi], fontsize=fontsize,
                    color=colors(i / num_phase), ha='center', va='bottom')
    ax.set_yticks(phase_angles)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(-200, 220)
    ax.set_xscale('log')
    ax.set_xlabel('$f$ / Hz')
    ax.set_ylabel(r'Phase / $^\circ$')
    ax.grid(True)


def plot_square_waves(ax, fs, f0, duration, amplitude, num_partials,
                      modal_window, phase_angles, colors):
    voffset = 5
    num_phase = len(phase_angles)
    for j, phi in enumerate(phase_angles):
        _, s, _ = util.square_wave(f0, num_partials, amplitude, duration,
                                   fs, phi, modal_window)
        t = 1000 * np.arange(len(s)) / fs
        ax.plot(t, s + j * voffset, c=colors(j / num_phase))
    ax.xaxis.set_ticks(np.arange(0, duration * 1000 + 20, 20))
    ax.yaxis.set_ticks(voffset * np.arange(num_phase))
    ax.yaxis.set_ticklabels(np.arange(-180, 180 + 45, 45))
    ax.xaxis.grid()
    ax.set_xlim(-1, duration * 1000 + 1)
    ax.set_ylim(-0.5 * voffset, (num_phase - 0.5) * voffset)
    ax.set_xlabel('$t$ / ms')
    ax.set_ylabel(r'Phase shift / $^\circ$')
    ax.set_title('Phase shifted square waves')


# Ideal magnitude response
H_mag = 1

# Frequency axis
fmin, fmax, numf = 20, 22000, 2
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf, endpoint=True)

# Phase angles
phimin, phimax, dphi = -180, 180, 45  # in degree
phase_angles_deg = np.arange(phimin, phimax + dphi, dphi)
phase_angles_rad = np.deg2rad(phase_angles_deg)
annotations = {0: 'original', -90: 'Hilbert transform',
               180: 'reverse polarity', -180: 'reverse polarity'}

# Square Wave
fs = 44100
f0 = 50
duration = 10 / f0
amplitude = 1
num_partials = 50
modal_window = kaiser(2 * num_partials + 1, beta=4)[num_partials + 1:]

# Plots
colors = cm.get_cmap('viridis')  # line colors

fig = plt.figure(figsize=(10, 8))

gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                      wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

plot_magnitude(ax1, fmin, fmax, H_mag, color='k')
plot_phase(ax2, fmin, fmax, phase_angles_deg, colors, annotations)
plot_square_waves(ax3, fs, f0, duration, amplitude, num_partials,
                  modal_window, phase_angles_rad, colors)

filename = 'ideal-spectra-and-square-waves'
extension = '.pdf'
plt.savefig(dir_fig + filename + extension, bbox_inches='tight')
