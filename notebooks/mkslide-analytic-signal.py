import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import util
from os.path import join
from scipy.signal import resample


def plot_original(time, s0, ax, c='C0', lw=2, label='Original'):
    ax.plot(time, s0, c=c, lw=lw, label=label, zorder=3)


def plot_hilbert(time, sH, ax, c='C3', lw=2, label='Hilbert'):
    ax.plot(time, sH, c=c, lw=lw, label=label, zorder=3)


def plot_envelope(time, envelope, ax, c=[0.9, 0.9, 0.9], label='Envelope'):
    ax.fill_between(time, -envelope, envelope, color=c, label=label)


def plot_phase_shifts(time, phase_angles, signals, ax, c='Gray', lw=0.5):
    for phi, s in zip(phase_angles, signals):
        line = ax.plot(time, s, c=c, lw=lw, alpha=1)
    line[0].set_label('$0,45,...,315$')


def decorate(ax, title=None, legend=True):
    ax.set_xlabel('$t$ / ms')
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(-1, 1)
    if legend:
        ax.legend(title=r'$\varphi$ / $^\circ$', ncol=2)
    ax.set_title(title)


# Audio signal
dir_src = '../data/source-signals/'
filename = 'castanets.wav'
start, stop = 6650, 6850
s0, fs = sf.read(join(dir_src, filename), start=start, stop=stop)

# Upsampling
oversample = 2
s0 = resample(s0, num=len(s0) * oversample)
fs *= oversample

# Phase angles and shifter order
phimin, phimax, phinum = 0, 2 * np.pi, 8
phase_angles = np.linspace(phimin, phimax, num=phinum, endpoint=False)
filter_order = 2**15

# Hilbert transform
hH = util.constant_phase_shifter(filter_order, -0.5 * np.pi, beta=8.6)[1]
sH = util.acausal_filter(s0, hH)
envelope = np.sqrt(s0**2 + sH**2)

# Constant phase shift
signals = [np.cos(phi) * s0 - np.sin(phi) * sH for phi in phase_angles]

# time axis
time = util.n2t(start + np.arange(len(s0)), fs=fs, ms=True)


# Plots
fmt = 'pdf'

# Original
fig, ax = plt.subplots(figsize=(12, 6))
plot_original(time, s0, ax)
decorate(ax, legend=False)
plt.savefig('original', bbox_inches='tight', format=fmt)

# Original and Hilbert transform
plot_hilbert(time, sH, ax)
decorate(ax, legend=True)
plt.savefig('original-hilbert', bbox_inches='tight', format=fmt)

# Original, Hilbert transform, and enevelope
plot_envelope(time, envelope, ax)
decorate(ax, legend=True)
plt.savefig('original-hilbert-envelope', bbox_inches='tight', format=fmt)

# Original, Hilbert, envelope, and phase shifted signals
plot_phase_shifts(time, phase_angles, signals, ax)
decorate(ax, legend=True)
plt.savefig('original-hilbert-envelope-phaseshifts',
            bbox_inches='tight', format=fmt)


# Phase shifted signals and envelope
fig, ax = plt.subplots(figsize=(12, 6))
plot_phase_shifts(time, phase_angles, signals, ax)
plot_envelope(time, envelope, ax)
decorate(ax, legend=True)
plt.savefig('envelope-phaseshifts', bbox_inches='tight', format=fmt)
