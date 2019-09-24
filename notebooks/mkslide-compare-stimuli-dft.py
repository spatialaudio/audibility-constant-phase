import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from os import path, mkdir
from util import db

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

dir_stimuli = '../abx_software/webMUSHRA_c929877_20180814/'\
              + 'configs/resources/stimuli/'

stimuli = ['square_a', 'square_b', 'pnoise', 'castanets', 'hotelcalifornia']
a_label = ['square wave ($-90\\degree$)',
           'square wave ($-45\\degree$)',
           'pink noise ($-90\\degree$)',
           'castanets ($-90\\degree$)',
           'hotel california ($-90\\degree$)']

fs = 44100

fig, ax = plt.subplots(figsize=(15, 10), ncols=3, nrows=2,
                       sharex=True, sharey=True)

for i, stml in enumerate(stimuli):
    s0 = sf.read(dir_stimuli + stml + '_phi000.wav')[0][:, 0]
    try:
        sp = sf.read(dir_stimuli + stml + '_phi270.wav')[0][:, 0]
    except:
        sp = sf.read(dir_stimuli + stml + '_phi315.wav')[0][:, 0]
    S0 = np.fft.rfft(s0)
    Sp = np.fft.rfft(sp)
    Smax = np.max(np.abs(S0))
    N = len(s0)
    f = fs * np.arange(N // 2 + 1) / N
    axi = ax.flat[i]
    axi.plot(f, db(S0 / Smax), 'C0', lw=1, label='Original Signal')
    axi.plot(f, db((np.abs(S0) - np.abs(Sp)) / Smax), 'C3', lw=2,
             label='Magnitude Deviation')
    axi.set_xlim(-5, 120)
    axi.set_ylim(-150, 10)
    axi.set_title(a_label[i])
    axi.grid(True)
ax[1, 0].set_xlim()
ax[1, 0].legend(loc='upper left')
for axi in ax[1]:
    axi.set_xlabel('$f$ / Hz')
for axi in ax[:, 0]:
    axi.set_ylabel('Magnitude / dB')
ax.flat[-1].axis('off')

filename = 'stimuli-spectra.pdf'
plt.savefig(dir_fig + filename, bbox_inches='tight')
