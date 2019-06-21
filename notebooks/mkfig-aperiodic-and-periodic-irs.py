import numpy as np
import matplotlib.pyplot as plt
import util
import importlib
from matplotlib import rcParams

importlib.reload(util)
dir_fig = '../paper/graphics/'

rcParams['figure.figsize'] = [15, 3.5]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 13
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Phase angle
phi = -np.pi / 4

# Impulse response
nmin, nmax = -16, 16
n = np.arange(nmin, nmax + 1)
h = util.discrete_ir_constant_phase(n, phi)

# Periodic impulse responses
Neven, Nodd = 32, 31
hp_even = util.periodic_constant_phase_shifter_ir(Neven, phi)
hp_odd = util.periodic_constant_phase_shifter_ir(Nodd, phi)

# Set h[0] to None
h[n == 0] = None
hp_even[0] = None
hp_odd[0] = None

# h[0]
n0 = [0]
h0 = [np.cos(phi)]

# Colors
col0 = 'C8'  # color for h[n==0]
colnz = 'C0'  # color for h[n!=0]

fig, ax = plt.subplots(ncols=3, sharey=True, gridspec_kw={'wspace': 0.05})

# Impulse response
ax[0].stem(n0, h0, markerfmt=col0 + 'o', linefmt=col0, basefmt=col0)
ax[0].stem(n, h, markerfmt=colnz + 'o', linefmt=colnz, basefmt=colnz)
ax[0].set_xlim(nmin - 0.5, nmax + 0.5)
ax[0].set_title('$h[n]$')

# Periodic impulse response with even period
ax[1].stem(n0, h0, markerfmt=col0 + 'o', linefmt=col0, basefmt=col0)
ax[1].stem(hp_even, markerfmt=colnz + 'o', linefmt=colnz, basefmt=colnz)
ax[1].set_title(r'$\tilde{{h}}[n]$, $M={}$'.format(Neven))
ax[1].set_xlim(-0.5, Neven - 0.5)

# Periodic impulse response with odd period
ax[2].stem(n0, h0, markerfmt=col0 + 'o', linefmt=col0, basefmt=col0)
ax[2].stem(hp_odd, markerfmt=colnz + 'o', linefmt=colnz, basefmt=colnz)
ax[2].set_title(r'$\tilde{{h}}[n]$, $M={}$'.format(Nodd))
ax[2].set_xlim(-0.5, Nodd - 0.5)

for axi in ax:
    axi.set_ylim(-0.55, 0.8)
    axi.set_xlabel('$n$ / sample')
    axi.grid(color='lightgray')

filename = 'discrete-irs-phi{:03.0f}.pdf'.format(np.rad2deg(phi))
plt.savefig(dir_fig + filename, dpi=300, bbox_inches='tight')
