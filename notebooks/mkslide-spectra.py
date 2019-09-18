import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib import cm
from util import db
from scipy.signal import freqz
from matplotlib import rcParams
from os import path, mkdir

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)


rcParams['figure.figsize'] = [12, 4.5]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'sans serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 13
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'

# FIR phase shifter
fs = 44100
phimin, phimax, num_phase = 0, np.pi, 9
phase_angles = np.linspace(phimin, phimax, num=num_phase, endpoint=True)
filter_order = 510
beta = 8.6
half_length = filter_order / 2
group_delay = half_length / fs

# Compare varying filter orders for a selected phase angle
idx_varN = 3
phase_varN = phase_angles[idx_varN]
Filter_orders = 2**9 - 2, 2**11 - 2, 2**13 - 2

# Frequencies
fmin, fmax, fnum = 1, fs / 2, 5000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=fnum)
omega = 2 * np.pi * f

# Plots
colors = cm.get_cmap('viridis', len(phase_angles)).colors
phi_ticks = np.arange(0, 180 + 45, 45)
phi_lim = -10, 190
fontsize_label = 9
f_maglabel = 8000
f_phaselabel = 8000
voffset_tf = 6
voffset_maglabel = 0.5
voffset_phaselabel = 1.875

fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.25})

for i, fo in enumerate(Filter_orders):
    _, h = util.constant_phase_shifter(fo, phase_varN, beta=beta)
    _, H = freqz(h, 1, f, fs=fs)
    gd = fo / 2 / fs
    H *= np.exp(1j * omega * gd)  # compensate group delay
    phase = np.unwrap(np.mod(np.angle(H), 2 * np.pi))
    phase_deg = np.rad2deg(phase)

    vshift = -idx_varN * voffset_tf
    color = colors[idx_varN]
    opacity = 1.2**(-i)

    # Frequency responses for varying filter order
    ax[0].semilogx(f, vshift + db(H), color=color, lw=1.5,
                   ls=':', alpha=opacity)
    ax[1].semilogx(f, phase_deg, color=color, lw=1.5, ls=':', alpha=opacity)

    # Labels: filter order
    ax[0].text(4**(-i) * 30, -21.5, r'$N={:0.0f}$'.format(fo + 1), rotation=33,
               va='center', ha='left', fontsize=8, color='k')
    ax[1].text(4**(-i) * 30, 53, r'$N={:0.0f}$'.format(fo + 1), rotation=45,
               va='center', ha='left', fontsize=8, color='k')

for i, phi in enumerate(phase_angles):
    n, h = util.constant_phase_shifter(filter_order, phi, beta=beta)
    _, H = freqz(h, 1, f, fs=fs)
    H *= np.exp(1j * omega * group_delay)
    phase = np.unwrap(np.mod(np.angle(H), 2 * np.pi))
    phase_deg = np.rad2deg(phase)
    vshift = -i * voffset_tf
    label = r'${:0.1f}\degree$'.format(np.rad2deg(phi))

    # Frequency responses for different phase angles
    ax[0].semilogx(f, vshift + db(H), color=colors[i], lw=1.5)
    ax[1].semilogx(f, phase_deg, color=colors[i], lw=1.5)

    # DC gain
    ax[0].plot(fmin * 1.1, vshift + db(np.cos(phi)), 'k<', ms=3)

    # Labels: phase angles
    ax[0].text(f_maglabel, vshift + voffset_maglabel, label, ha='right',
               fontsize=fontsize_label)
    ax[1].text(f_phaselabel, np.rad2deg(phi) + voffset_phaselabel, label,
               ha='right', fontsize=fontsize_label)

ax[0].set_yticks(np.arange(-num_phase * voffset_tf, 12, 6))
ax[0].set_ylim((-num_phase + 1) * voffset_tf - 3, 3)
ax[0].set_ylabel('Magnitude / dB')
ax[1].set_yticks(phi_ticks)
ax[1].set_ylim(phi_lim)
ax[1].set_ylabel(r'Phase / $^\circ$')
for axi in ax:
    axi.set_xlim(fmin, fmax + 500)
    axi.set_xlabel('$f$ / Hz')
    axi.grid(color='lightgray')

file_name = 'spectra_filterorder{}.pdf'.format(filter_order)
plt.savefig(dir_fig + file_name, dpi=300, bbox_inches='tight')
