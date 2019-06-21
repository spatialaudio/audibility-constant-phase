import numpy as np
import matplotlib.pyplot as plt
import util
import importlib
import re
from scipy.signal import kaiser
from matplotlib import rcParams, cm

importlib.reload(util)
dir_crest_factor = '../data/crest-factor/'
dir_fig = '../paper/graphics/'

rcParams['figure.figsize'] = [12, 5]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'


def plot_crest_factor(ax, dir_crest_factor, stimuli, labels, phase_angles,
                      colors, xticks):
    for i, stm in enumerate(stimuli):
        phi, C = np.loadtxt(dir_crest_factor + stm + '.txt')
        phi -= np.pi  # shift angle by -np.pi
        color = colors(i * 0.25)
        label = labels[i]
        Cdiff = util.db(C[-90] / C[0])
        ax.plot(np.rad2deg(phi), util.db(C), c=color, label=label)
        ax.plot(-90, util.db(C[-90]), c=color, marker='o')
        color = (color if i!=3 else 'k')
        ax.text(-90, util.db(C[-90]) + 0.5, '${:+0.1f}$ dB'.format(Cdiff),
                color=color, va='bottom', ha='center')
        ax.text(100, util.db(C[100]) + 0.3, label,
                color=color, va='bottom', ha='center')
        if re.match(re.compile('square.'), stm):
            Cdiff = util.db(C[-45] / C[0])
            ax.plot(-45, util.db(C[-45]), c=color, marker='o')
            ax.text(-45, util.db(C[-45]) + 0.5, '${:+0.1f}$ dB'.format(Cdiff),
                    color=color, va='bottom', ha='left')
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel(r'Phase Shift $\varphi$ / $\degree$')
    ax.set_ylabel('Crest Factor (CF) / dB')
    ax.grid(True, color='lightgray')

    color_legend = [0.5, 0.5, 0.5]
    ax.plot([-145, -125], [21, 21], c=color_legend)
    ax.plot(-135, 21, c=color_legend, marker='o')
    ax.text(-135, 21.5, 'CF change \n relative to $0\\degree$',
            color='k', va='bottom', ha='center')


def plot_square_waves(ax, fs, f0, duration, amplitude, num_partials,
                      modal_window, phase_angles, selected_angles, color):

    voffset = 1
    t_label = -2
    phi_labels = [r'${:0.0f}\degree$'.format(np.rad2deg(phi))
                  for phi in phase_angles]
    for j, phi in enumerate(phase_angles):
        alpha = (1 if phi in selected_angles else 0.33)
        _, s, _ = util.square_wave(f0, num_partials, amplitude, duration,
                                   fs, phi, modal_window)
        t = 1000 * np.arange(len(s)) / fs
        ax.plot(t, s - j * voffset, c=color, alpha=alpha)
        ax.text(t_label, - j * voffset, phi_labels[j], fontsize=9, ha='right')
    ax.xaxis.set_ticks(np.arange(0, 80, 20))
    ax.yaxis.set_ticks([])
    ax.xaxis.grid()
    ax.set_xlim(-15, duration * 1000 + 0)
    ax.set_ylim(-8.5, 1)
    ax.set_xlabel('$t$ / ms')
    ax.text(20, 0.5, 'square wave', ha='center', color=color)


# Crest Factor
stimuli = ['square_crestfactor',
           'pnoise_crestfactor_selection',
           'hotelcalifornia_crestfactor_selection',
           'castanets_crestfactor']
labels = ['square wave bursts', 'pink noise', 'hotel california', 'castanets']
phi_crestfactor = np.linspace(0, 2 * np.pi, endpoint=True, num=360)
colors = cm.get_cmap('viridis')
phi_ticks = np.arange(-180, 180 + 90, 90)

# Square Wave
fs = 44100
f0 = 50
duration = 3 / f0
amplitude = 0.2
num_partials = 50
modal_window = kaiser(2 * num_partials + 1, beta=4)[num_partials + 1:]
dphi = np.pi / 4
num_angles = int(2 * np.pi / dphi) + 1
phi_square = np.linspace(-np.pi, np.pi, num=9, endpoint=True)
phi_stimuli = [0, -np.pi / 4, -np.pi / 2]


# Single plot (crest factor)
fig, ax = plt.subplots(figsize=(7, 4.5))
plot_crest_factor(ax, dir_crest_factor, stimuli, labels, phi_crestfactor,
                  colors, phi_ticks)
plt.savefig(dir_fig + 'crest-factor.pdf', bbox_inches='tight')

# Suplots (crest factor + square wave)
fig, ax = plt.subplots(figsize=(12, 5), ncols=2,
                       gridspec_kw={'width_ratios': [5, 2], 'wspace': 0.03})
plot_crest_factor(ax[0], dir_crest_factor, stimuli, labels,
                  phi_crestfactor, colors, phi_ticks)
plot_square_waves(ax[1], fs, f0, duration, amplitude, num_partials,
                  modal_window, phi_square, phi_stimuli, colors(0))
plt.savefig(dir_fig + 'crest-factor-and-square-waves.pdf', bbox_inches='tight')
