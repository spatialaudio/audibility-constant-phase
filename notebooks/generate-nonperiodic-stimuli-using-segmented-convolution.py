import numpy as np
import soundfile as sf
import util
import importlib
from os.path import join
from scipy.signal import butter, lfilter

importlib.reload(util)

# Directories
src_dir = '../data/source-signals'
out_dir = '../data/stimuli'

# Phase angles
phase_angles = np.linspace(0, 1 * np.pi, num=3, endpoint=True)  # in radian
filter_order = 3963530


# Pink Noise
songname = 'pnoise'
fs = 44100  # Hz
t = 6*60  # s
lpf_order = 4
f_cutoff = 300  # Hz
np.random.seed(1024)
b, a = butter(lpf_order, Wn=f_cutoff, btype='lowpass', fs=fs)
pn = util.pink_noise(int(t*fs))
pn = pn - np.mean(pn)
pn = lfilter(b, a, pn)
pn = pn - np.mean(pn)
pn = pn / np.max(np.abs(pn)) * 0.25  # peak -12 dB
x = pn
n_start, n_stop = 10789628, 10893326  # in samples
t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
util.make_aperiodic_stimuli_selection(
        x, fs, songname, n_start, n_stop, phase_angles,
        t_fadein, t_fadeout, t_intro, t_outro, channels=2,
        save_stimuli=True, crestfactor=False, out_dir=out_dir)
