import numpy as np
import soundfile as sf
import util
import importlib
import matplotlib.pyplot as plt
from scipy.signal import kaiser

importlib.reload(util)

# Phase angles
phase_angles = np.array([0, -0.25 * np.pi, -0.5 * np.pi])
filter_order = 3963530

# Square wave bursts
fs = 44100
f0 = 44.1
amplitude = 0.25
duration = 10 / f0
num_partials = 10
modal_window = kaiser(2 * num_partials + 1, beta=4)[num_partials + 1:]
_, x, _ = util.square_wave(f0, num_partials, amplitude, duration,
                           fs, 0, modal_window)
n_taper = util.t2n(2 / f0, fs, ms=False)
x = util.fade(x, n_taper, n_taper, type='h')  # one square wave burst

t_period, t_predelay, t_intro, t_outro = 500, 20, 0, 0  # in milliseconds
t_fadein, t_fadeout = 0, 0
t_signal = util.n2t(len(x), fs, ms=True)
t_prepend = t_predelay
t_append = t_period - t_signal - t_predelay
n_prepend = util.t2n(t_prepend, fs, ms=True)
n_append = util.t2n(t_append, fs, ms=True)
x_1p = util.prepend_append_zeros(x, n_prepend, n_append)  # one period

# Phase shift in the DFT domain
repetition = 3  # three square burst train
N = repetition * len(x_1p)
y_dft = np.zeros((len(phase_angles), N))
Y_dft = np.zeros((len(phase_angles), N // 2 + 1), dtype='complex128')
for i, phi in enumerate(phase_angles):
    y = util.constant_phase_shift_dft(x_1p, phi)
    y = util.periodic_signal(y, repetition)
    y_dft[i] = y
    Y_dft[i] = np.fft.rfft(y)

# Phase shift using FIR filter
long_repetition = 601  # very long burst train
n_start = util.t2n(300 * t_period, fs=fs, ms=True)
n_stop = util.t2n(303 * t_period, fs=fs, ms=True)
x_Mp = util.periodic_signal(x_1p, long_repetition)
y_fir = np.zeros((len(phase_angles), N))  # selection of long signal
Y_fir = np.zeros((len(phase_angles), N // 2 + 1), dtype='complex128')
for i, phi in enumerate(phase_angles):
    h = util.constant_phase_shifter(filter_order, phi, beta=8.6)[1]
    y = util.acausal_filter(x_Mp, h)[n_start:n_stop]
    y_fir[i] = y
    Y_fir[i] = np.fft.rfft(y)


# I. Float64

# Waveforms - DFT vs FIR methods
t = np.arange(len(y_fir[0])) / fs * 1000
fig, ax = plt.subplots(figsize=(12, 5), ncols=3, sharey=True)
for i, (phi, yd, yf) in enumerate(zip(phase_angles, y_dft, y_fir)):
    ax[i].plot(t, util.db(yd), c='lightgray', label='DFT')
    ax[i].plot(t, util.db(yd - yf), label='diff')
    ax[i].set_xlabel('$t$ / ms')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylabel('Amplitude / dB')
ax[0].legend(loc='upper right')
fig.suptitle('[FLOAT] Waveforms - DFT vs FIR')

# Spectra - DFT vs FIR methods
Nrfft = N // 2 + 1
f = np.arange(Nrfft) / N * fs
fig, ax = plt.subplots(figsize=(12, 5), ncols=3, sharey=True)
for i, (phi, Yd, Yf) in enumerate(zip(phase_angles, Y_dft, Y_fir)):
    ax[i].plot(f, util.db(Yd), 'lightgray', label='DFT')
    ax[i].plot(f, util.db(Yd - Yf), label='diff')
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylabel('Magnitdue / dB')
ax[0].legend(loc='upper right')
fig.suptitle('[FLOAT] Spectra - DFT vs FIR')

# Spectra for different phase angles - DFT
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Y) in enumerate(zip(phase_angles[1:], Y_dft[1:])):
    ax[i].plot(f[::repetition],
        util.db(np.abs(Y_dft[0, ::repetition]) / np.abs(Y[::repetition])))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylim(-0.01, 0.01)
ax[0].set_ylabel('Magnitdue Difference / dB')
fig.suptitle(r'[FLOAT] Spectra 0 vs $\varphi$ - DFT')

# Spectra for different phase angles - FIR
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Y) in enumerate(zip(phase_angles[1:], Y_fir[1:])):
    ax[i].plot(f[::repetition],
        util.db(np.abs(Y_dft[0, ::repetition]) / np.abs(Y[::repetition])))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylim(-0.01, 0.01)
ax[0].set_ylabel('Magnitdue Difference / dB')
fig.suptitle(r'[FLOAT] Spectra 0 vs $\varphi$ - FIR')


# II. WAV

# Save and reload
z_dft = np.zeros_like(y_dft)
Z_dft = np.zeros_like(Y_dft)
z_fir = np.zeros_like(y_dft)
Z_fir = np.zeros_like(Y_dft)
temp_filename = 'temp.wav'
subtype = 'PCM_24'
for i, (yd, yf) in enumerate(zip(y_dft, y_fir)):
    sf.write(temp_filename, yd, fs, subtype=subtype)
    z, _ = sf.read(temp_filename)
    z_dft[i] = z
    Z_dft[i] = np.fft.rfft(z)

    sf.write(temp_filename, yf, fs, subtype=subtype)
    z, _ = sf.read(temp_filename)
    z_fir[i] = z
    Z_fir[i] = np.fft.rfft(z)

# Compare waveforms - DFT vs FIR methods
t = np.arange(N) / fs * 1000
fig, ax = plt.subplots(figsize=(12, 5), ncols=3, sharey=True)
for i, (phi, zd, zf) in enumerate(zip(phase_angles, z_dft, z_fir)):
    ax[i].plot(t, util.db(zd), c='lightgray', label='DFT')
    ax[i].plot(t, util.db(zd - zf), label='diff')
    ax[i].set_xlabel('$t$ / ms')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylabel('Amplitude / dB')
ax[0].legend(loc='upper right')
fig.suptitle('[WAV] Waveforms - DFT vs FIR')

# Comapre spectra - DFT vs FIR methods
f = np.arange(len(Z_dft[0])) / len(z_dft[0]) * fs
fig, ax = plt.subplots(figsize=(12, 5), ncols=3, sharey=True)
for i, (phi, Zd, Zf) in enumerate(zip(phase_angles, Z_dft, Z_fir)):
    ax[i].plot(f, util.db(Zd), 'lightgray', label='DFT')
    ax[i].plot(f, util.db(Zd - Zf), label='diff')
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
ax[0].set_ylabel('Magnitdue / dB')
ax[0].legend(loc='upper right')
fig.suptitle('[WAV] Spectra - DFT vs FIR')

# Compare spectra for different phase angles - DFT
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Z) in enumerate(zip(phase_angles[1:], Z_dft[1:])):
    ax[i].plot(f, util.db(Z_dft[0]), c='lightgray')
    ax[i].plot(f, util.db(Z))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
    ax[i].set_xlim(-1, 15)
ax[0].set_ylabel('Magnitdue / dB')
fig.suptitle(r'[WAV] Spectra 0 vs $\varphi$ - DFT')

# Compare spectra for different phase angles - DFT
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Z) in enumerate(zip(phase_angles[1:], Z_fir[1:])):
    ax[i].plot(f, util.db(Z_fir[0]), c='lightgray')
    ax[i].plot(f, util.db(Z))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
    ax[i].set_xlim(-1, 15)
ax[0].set_ylabel('Magnitdue / dB')
fig.suptitle(r'[WAV] Spectra 0 vs $\varphi$ - FIR')

# Spectra for different phase angles - DFT
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Z) in enumerate(zip(phase_angles[1:], Z_dft[1:])):
    ax[i].plot(f[::repetition],
        util.db(np.abs(Z_dft[0, ::repetition]) / np.abs(Z[::repetition])))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
    ax[i].set_xlim(-1, 20)
ax[0].set_ylabel('Magnitdue Difference / dB')
fig.suptitle(r'[WAV] Spectra 0 vs $\varphi$ - DFT')

# Spectra for different phase angles - FIR
fig, ax = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
for i, (phi, Z) in enumerate(zip(phase_angles[1:], Z_fir[1:])):
    ax[i].plot(f[::repetition],
        util.db(np.abs(Z_fir[0, ::repetition]) / np.abs(Z[::repetition])))
    ax[i].set_xlabel('$f$ / Hz')
    ax[i].set_title('{:0.0f}'.format(np.rad2deg(phi)))
    ax[i].grid(True)
    ax[i].set_xlim(-1, 20)
ax[0].set_ylabel('Magnitdue Difference / dB')
fig.suptitle(r'[WAV] Spectra 0 vs $\varphi$ - FIR')