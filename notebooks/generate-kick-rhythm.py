import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample
from os.path import join

# Source signal
src_dir = '../data/source-signals'
suffix = '.wav'
filename = 'techno-kick'
sig, fs = sf.read(join(src_dir, filename + suffix))
n_sig = len(sig)

# Parameters
repetitions = 3
t_period, t_intro, t_outro = 250, 10, 10  # in milliseconds
peak = -6  # in dB
do_plot = True
do_save = True

# Normalize
oversample = 4
true_peak = np.max(np.abs(resample(sig, oversample * n_sig)))
sig *= 10**(peak / 20) / true_peak

# Convert time variables to samples
n_intro, n_outro, n_period = np.round(
        np.array([t_intro, t_outro, t_period]) * fs / 1000).astype('int')

# Output
n_out = n_intro + (repetitions - 1) * n_period + n_sig + n_outro
y = np.zeros(n_out)
for i in range(repetitions):
    idx = n_intro + i * n_period + np.arange(n_sig)
    y[idx] += sig

if do_plot:
    fig, ax = plt.subplots()
    plt.plot(y)
if do_save:
    sf.write(join(src_dir, 'techno-tick-rhythm' + suffix), y, fs)
