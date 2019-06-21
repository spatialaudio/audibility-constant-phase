import numpy as np
import soundfile as sf
import util
import importlib
from os.path import join
from scipy.signal import butter, lfilter, kaiser

importlib.reload(util)

# Directories
dir_src = '../data/source-signals'
out_dir = '../data/crest-factor/'

phase_angles = np.linspace(0, 2 * np.pi, endpoint=True, num=360)

# Aperiodic stimuli
filter_order = 3963530

# Hotel California, Hell Freezes Over, Eagles, 1994
#filename = 'hotelcalifornia_mono_full'
#songname = 'hotelcalifornia'
#x, fs = sf.read(join(dir_src, filename + '.wav'))
#n_start, n_stop = 1981766, 2123080  # in samples
#t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
#util.make_aperiodic_stimuli(
#        x, fs, songname, n_start, n_stop, phase_angles,
#        filter_order, t_fadein, t_fadeout, t_intro, t_outro,
#        channels=2, save_stimuli=False, crestfactor=True,
#        crestfactor_full=False, out_dir=out_dir)

# Hotel California, Hell Freezes Over, Eagles, 1994
filename = 'hotelcalifornia_mono_fake'
songname = 'hotelcalifornia'
x, fs = sf.read(join(dir_src, filename + '.wav'))
n_start, n_stop = 1981766, 2123080  # in samples
t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
util.make_aperiodic_stimuli(
        x, fs, songname, n_start, n_stop, phase_angles,
        filter_order, t_fadein, t_fadeout, t_intro, t_outro,
        channels=2, save_stimuli=False, crestfactor=True,
        crestfactor_full=False, out_dir=out_dir)

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
hpre = util.constant_phase_shifter(filter_order, np.deg2rad(283), beta=8.6)[1]
pn = util.acausal_filter(pn, hpre)  # pre-shifting for decreasing crest factor
x = pn
n_start, n_stop = 9747840, 9851538
t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
util.make_aperiodic_stimuli(
        x, fs, songname, n_start, n_stop, phase_angles,
        filter_order, t_fadein, t_fadeout, t_intro, t_outro,
        channels=2, save_stimuli=False, crestfactor=True,
        crestfactor_full=False, out_dir=out_dir)


# Periodic Stimuli

# Castanets
filename = 'castanets'
x, fs = sf.read(join(dir_src, filename + '.wav'))

t_period, t_predelay, t_intro, t_outro = 2000, 0, 15, 15  # in milliseconds
repetition = 1
peak_db = -12
util.make_periodic_stimuli(
        x, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=False, save_crestfactor=True,
        db_lufs_ref=-35, out_dir=out_dir)

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

filename = 'square'
util.make_periodic_stimuli(
        y, fs, filename, phase_angles, repetition, peak_db,
        t_period, t_predelay, t_intro, t_outro, channels=2,
        save_stimuli=False, save_crestfactor=True, out_dir=out_dir)
