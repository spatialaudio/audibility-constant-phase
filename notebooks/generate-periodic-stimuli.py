import numpy as np
import soundfile as sf
import util
import importlib
from scipy.signal import kaiser
from os.path import join

importlib.reload(util)

# Directories
src_dir = '../data/source-signals'
out_dir = '../data/stimuli'

# Phase angles
phase_angles = np.linspace(0, 2 * np.pi, num=int(360/15), endpoint=False)  # rad

# Techno Kick
filename = 'techno-kick'
x, fs = sf.read(join(src_dir, filename + '.wav'))

t_period, t_predelay, t_intro, t_outro = 600, 25, 15, 15  # in milliseconds
repetition = 3
peak_db = -12
util. make_periodic_stimuli(
        x, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)


# Castanets
filename = 'castanets'
x, fs = sf.read(join(src_dir, filename + '.wav'))

t_period, t_predelay, t_intro, t_outro = 2000, 0, 15, 15  # in milliseconds
repetition = 1
peak_db = -12
util. make_periodic_stimuli(
        x, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=True, save_crestfactor=True,
        out_dir=out_dir, db_lufs_ref=-35)


# Square Wave Bursts
fs = 44100
f0 = 50
amplitude = 0.99
duration = 10 / f0
num_partials = 10
modal_window = kaiser(2 * num_partials + 1, beta=4)[num_partials + 1:]
_, y, _ = util.square_wave(f0, num_partials, amplitude, duration,
                           fs, 0, modal_window)
n_taper = util.t2n(2 / f0, fs, ms=False)
y = util.fade(y, n_taper, n_taper, type='h')

t_period, t_predelay, t_intro, t_outro = 500, 20, 0, 0  # in milliseconds
repetition = 3
peak_db = -12

filename = 'square_a'
util. make_periodic_stimuli(
        y, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)
filename = 'square_b'
util. make_periodic_stimuli(
        y, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)


# AUTHOR https://soundcloud.com/8-bit-logic
# DOWNLOAD http://99sounds.org/kick-drum/
#filename = 'min_kick_22_G'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#x = x[0:int((0.6-0.025)*fs)]  # 80 bpm
#t_period, t_predelay, t_intro, t_outro = 600, 25, 15, 15  # in milliseconds
#repetition = 3
#peak_db = -12
#util. make_periodic_stimuli(
#        x, fs, filename, phase_angles, repetition, peak_db,
#        t_period, t_predelay, t_intro, t_outro, channels=2,
#        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)


# AUTHOR https://soundcloud.com/8-bit-logic
# DOWNLOAD http://99sounds.org/kick-drum/
#filename = 'sub_kick_23_G'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#x = x[0:int((0.6-0.025)*fs)]  # 80 bpm
#t_period, t_predelay, t_intro, t_outro = 600, 25, 15, 15  # in milliseconds
#repetition = 3
#peak_db = -12
#util. make_periodic_stimuli(
#        x, fs, filename, phase_angles, repetition, peak_db,
#        t_period, t_predelay, t_intro, t_outro, channels=2,
#        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)


# AUTHOR https://soundcloud.com/8-bit-logic
# DOWNLOAD http://99sounds.org/kick-drum/
#filename = 'trad_kick_12_D'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#x = x[0:int((0.6-0.025)*fs)]  # 80 bpm
#t_period, t_predelay, t_intro, t_outro = 600, 25, 15, 15  # in milliseconds
#repetition = 3
#peak_db = -12
#util. make_periodic_stimuli(
#        x, fs, filename, phase_angles, repetition, peak_db,
#        t_period, t_predelay, t_intro, t_outro, channels=2,
#        save_stimuli=True, save_crestfactor=True, out_dir=out_dir)
