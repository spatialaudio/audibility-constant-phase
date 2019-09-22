import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib import rcParams, font_manager
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
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.size'] = 13
rcParams['font.stretch'] = 'condensed'
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Discrete-time impulse response
phi = -np.pi / 2
nmin, nmax = -16, 16
n = np.arange(nmin, nmax + 1)
h = util.discrete_ir_constant_phase(n, phi)

# Amplitude decay in linear scale
n_dense_neg = np.linspace(nmin, -0.1, num=100, endpoint=True)
n_dense_pos = np.linspace(0.1, nmax, num=100, endpoint=True)
envelope_neg = 2 / np.pi / n_dense_neg
envelope_pos = 2 / np.pi / n_dense_pos

# Amplitude decay in log-log scale
fs = 44100
tmin, tmax, tnum = 0.001, 360, 10000
t = np.logspace(np.log10(tmin), np.log10(tmax), num=tnum)
sample = t * fs
nmin, nmax = sample.min(), sample.max()
envelope = 2 / np.pi / sample

# Quantization errors
E16 = 2**-16
E24 = 2**-24

# Plots
col = 'C0'

fig, ax = plt.subplots(figsize=(12, 4), ncols=2, gridspec_kw={'wspace': 0.33})

# impulse response and decay curve
ax[0].stem(n, h, markerfmt=col + 'o', linefmt=col, basefmt=col, label='$h[n]$')
ax[0].plot(n_dense_neg, envelope_neg, 'k--', zorder=0, label=r'$2 / \pi n$')
ax[0].plot(n_dense_pos, envelope_pos, 'k--', zorder=0)
ax[0].set_ylim(-0.8, 0.8)
ax[0].set_xlabel('$n$ / sample')
ax[0].legend()

# decay curve in log-log axes with quantization errors
ax[1].semilogx(sample, util.db(envelope), 'k--', label=r'$|2 / \pi n|$')
ax[1].hlines(util.db(E16), nmin, nmax, colors='C1', zorder=0)
ax[1].hlines(util.db(E24), nmin, nmax, colors='C3', zorder=0)
ax[1].grid(color='lightgray')
ax[1].set_xlabel('$n$ / sample')
ax[1].set_ylabel('Amplitude / dB')
ax[1].set_xlim(nmin, nmax)
ax[1].text(1000, -60, r'$\frac{2}{\pi n}$', rotation=-35, fontsize=25)
ax[1].text(100, util.db(E16), r'16 bit $\Delta_{Q}$', va='bottom')
ax[1].text(100, util.db(E24), r'24 bit $\Delta_{Q}$', va='bottom')

# secondary x-axis in seconds
ax2 = ax[1].twiny()
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 60))
ax2.set_xlabel('$t$ / s')
ax2.set_xlim(tmin, tmax)
ax2.set_xscale('log')

filename = 'amplitude-decay'
ext = '.pdf'
plt.savefig(dir_fig + filename + ext, dpi=300, bbox_inches='tight')
