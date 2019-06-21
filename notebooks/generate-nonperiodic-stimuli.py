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
phase_angles = np.linspace(0, 2 * np.pi, endpoint=False, num=int(360/15))  # rad
filter_order = 3963530


# Hotel California, Hell Freezes Over, Eagles, 1994
#filename = 'hotelcalifornia_mono_full'
#songname = 'hotelcalifornia'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#n_start, n_stop = 1981766, 2123080  # in samples
#t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
#util.make_aperiodic_stimuli(x, fs, songname, n_start, n_stop, phase_angles,
#                            filter_order, t_fadein, t_fadeout, t_intro, t_outro,
#                            channels=2, save_stimuli=True, crestfactor=True,
#                            crestfactor_full=False, out_dir=out_dir)


# Tiesto & DallasK, Show Me (Original Mix), 2015
#filename = 'showme_mono_full'
#songname = 'showme'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#n_start, n_stop = 4630950, 4713552  # in samples
#t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
#util.make_aperiodic_stimuli(x, fs, songname, n_start, n_stop, phase_angles,
#                            filter_order, t_fadein, t_fadeout, t_intro, t_outro,
#                            channels=2, save_stimuli=True, crestfactor=True,
#                            crestfactor_full=False, out_dir=out_dir)


# AutoErotique, Asphyxiation, 2013
#filename = 'asphyxiation_mono_full'
#songname = 'asphyxiation'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#n_start, n_stop = 1734705, 1817408  # in samples
#t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
#util.make_aperiodic_stimuli(x, fs, songname, n_start, n_stop, phase_angles,
#                            filter_order, t_fadein, t_fadeout, t_intro, t_outro,
#                            channels=2, save_stimuli=True, crestfactor=True,
#                            crestfactor_full=False, out_dir=out_dir)


# Knife Party, 404, 2014
#filename = 'knifeparty404_mono_full'
#songname = 'knifeparty404'
#x, fs = sf.read(join(src_dir, filename + '.wav'))
#n_start, n_stop = 10789628, 10893326  # in samples
#t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
#util.make_aperiodic_stimuli(x, fs, songname, n_start, n_stop, phase_angles,
#                            filter_order, t_fadein, t_fadeout, t_intro, t_outro,
#                            channels=2, save_stimuli=True, crestfactor=True,
#                            crestfactor_full=False, out_dir=out_dir)


# Hotel California, Hell Freezes Over, Eagles, 1994
filename = 'hotelcalifornia_mono_fake'
songname = 'hotelcalifornia'
x, fs = sf.read(join(src_dir, filename + '.wav'))
n_start, n_stop = 1981766, 2123080  # in samples
t_fadein, t_fadeout, t_intro, t_outro = 10, 10, 10, 10  # in milliseconds
util.make_aperiodic_stimuli(x, fs, songname, n_start, n_stop, phase_angles,
                            filter_order, t_fadein, t_fadeout, t_intro, t_outro,
                            channels=2, save_stimuli=True, crestfactor=False,
                            crestfactor_full=False, out_dir=out_dir,
                            db_lufs_ref=-35)


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
        x, fs, songname, n_start, n_stop, phase_angles, filter_order,
        t_fadein, t_fadeout, t_intro, t_outro, channels=2, save_stimuli=True,
        crestfactor=True, crestfactor_full=False, out_dir=out_dir)
