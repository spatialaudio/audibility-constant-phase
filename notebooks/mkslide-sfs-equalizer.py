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

rcParams['figure.figsize'] = [8, 3]
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


def plot_frequency_response(f, H, ax, **kw):
    plot_magnitude(ax[0], f, H, **kw)
    plot_phase(ax[1], f, H, **kw)
    decorate(ax)


def plot_magnitude(ax, f, H, **kw):
    ax.semilogx(f, db(H), **kw)


def plot_phase(ax, f, H, **kw):
    ax.semilogx(f, np.rad2deg(np.angle(H)), **kw)


def decorate(ax):
    ax[0].set_ylim(-33, 33)
    ax[0].set_ylabel('Magnitude / dB')
    ax[1].set_ylim(-100, 100)
    ax[1].set_yticks(np.arange(-90, 90 + 45, 45))
    ax[1].set_ylabel(r'Phase / $\degree$')
    for axi in ax:
        axi.set_xlim(fmin, fmax)
        axi.set_xlabel(r'$\log f$')
        axi.set_xticklabels('')
        axi.grid(True)


def annotate_magnitude(ax, f, H, f0, string, **kw):
    H0 = H[np.argmin(np.abs(f - f0))]

    p0 = ax.transData.transform_point((f[0], db(H[0])))
    p1 = ax.transData.transform_point((f[-1], db(H[-1])))
    dx, dy = p1 - p0
    angle = np.rad2deg(np.arctan2(dy, dx))
    ax.text(f0, db(H0), string, rotation=angle, rotation_mode='anchor', **kw)


def annotate_phase(ax, f, H, f0, string, **kw):
    H0 = H[np.argmin(np.abs(f - f0))]
    ax.text(f0, np.rad2deg(np.angle(H0)), string, **kw)


c = 343
fmin, fmax, numf = 20, 22000, 20
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf, endpoint=True)
omega = 2 * np.pi * f
k = omega / c

# Phase angles
phimin, phimax, dphi = -180, 180, 45  # in degree
phase_angles_deg = np.arange(phimin, phimax + dphi, dphi)
phase_angles_rad = np.deg2rad(phase_angles_deg)
annotations = {0: 'original', -90: 'Hilber transform',
               180: 'reverse polarity', -180: 'reverse polarity'}

# Ideal planar and line sources and the corresponding equalizers
S_plane = 3 / 1j / k
S_line = np.sqrt(3 / 1j / k)
H_plane = 1 / S_plane
H_line = 1 / S_line

# Practical system
fl = 100
fh = 3000
Hl = H_line[np.argmin(np.abs(f - fl))]
Hh = H_line[np.argmin(np.abs(f - fh))]
magnitude = np.clip(np.abs(H_line), np.abs(Hl), np.abs(Hh)).astype('complex')
phase = np.stack([np.angle(H_line)[0] if (fi > fl) and (fi < fh) else 0
                  for fi in f])
H_minph = magnitude * np.exp(1j * phase)

# Linear-phase design
H_linph = magnitude

# Plots
col_source = 'Gray'
col_eq = 'C0'
lw = 3
ext = 'pdf'

# Ideal plane source
fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.4})
plot_frequency_response(f, S_plane, ax, lw=lw, color=col_source)
plot_frequency_response(f, H_plane, ax, lw=lw, color=col_eq)
annotate_magnitude(ax[0], f, S_plane, 1000, 'Planar Source \n (-6 dB/oct)',
                   va='top', ha='center')
annotate_magnitude(ax[0], f, H_plane, 1000, 'EQ (+6 dB/oct)',
                   va='bottom', ha='center')
annotate_phase(ax[1], f, S_plane, 1000, r'Planar Source (-90 $\degree$)',
               va='bottom', ha='center')
annotate_phase(ax[1], f, H_plane, 1000, r'EQ (+90 $\degree$)',
               va='top', ha='center')
plt.savefig(dir_fig + 'equalizer-ideal-plane-source.' + ext,
            bbox_inches='tight')

# Ideal line source
fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.4})
plot_frequency_response(f, S_line, ax, lw=lw, color=col_source)
plot_frequency_response(f, H_line, ax, lw=lw, color=col_eq)
annotate_magnitude(ax[0], f, S_line, 1000, 'Line Source \n (-3 dB/oct)',
                   va='top', ha='center')
annotate_magnitude(ax[0], f, H_line, 1000, 'EQ (+3 dB/oct)',
                   va='bottom', ha='center')
annotate_phase(ax[1], f, S_line, 1000, r'Line Source (-45 $\degree$)',
               va='bottom', ha='center')
annotate_phase(ax[1], f, H_line, 1000, r'EQ (+45 $\degree$)',
               va='bottom', ha='center')
plt.savefig(dir_fig + 'equalizer-ideal-line-source.' + ext,
            bbox_inches='tight')

# Minimum phase
fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.4})
plot_frequency_response(f, H_minph, ax, color=col_eq, lw=lw)
ax[1].text(663, 80, 'Minimum Phase', ha='center', va='top')
plt.savefig(dir_fig + 'equalizer-minimum-phase.' + ext, bbox_inches='tight')

# Linear phase
fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.4})
plot_frequency_response(f, H_linph, ax, color=col_eq, lw=lw)
plot_phase(ax[1], f, S_line, lw=lw, color=col_source)
decorate(ax)
annotate_phase(ax[1], f, S_line, 1000, r'Line Source (-45 $\degree$)',
               va='bottom', ha='center')
ax[1].text(663, 80, r'Linear Phase (constant $\tau_G$)', ha='center', va='top')
ax[1].text(663, 0, r'$\varphi - \omega \cdot \tau_G$',
           color='Gray', fontsize=22, ha='center', va='bottom', zorder=0)
plt.savefig(dir_fig + 'equalizer-linear-phase.' + ext, bbox_inches='tight')
