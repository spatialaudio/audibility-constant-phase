import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import util
from os.path import join
from scipy.signal import resample

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

# Plot
time = util.n2t(np.arange(len(s0)), fs=fs, ms=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(time, -envelope, envelope,
                color=[0.9, 0.9, 0.9], label='Envelope')
for phi, s in zip(phase_angles, signals):
    if phi == 0:
        ax.plot(time, s0, c='C0', lw=2, label='Original')
    elif phi == 1.5 * np.pi:
        ax.plot(time, s, c='C3', lw=2,
                label='{:0.0f} (Hilb.)'.format(np.rad2deg(phi)))
    else:
        ax.plot(time, s, c='k', lw=0.5, alpha=0.5,
                label='{:0.0f}'.format(np.rad2deg(phi)))
ax.set_xlabel('$t$ / ms')
ax.legend(title=r'$\varphi$ / $^\circ$', ncol=3)
ax.set_title('Phase shifted signals and envelope')
ax.set_xlim(time.min(), time.max())
plt.savefig('phase-shifted-signals-and-envelope.png', bbox_inches='tight')
