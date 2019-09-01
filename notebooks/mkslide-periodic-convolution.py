import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib import rcParams, font_manager
from os import path, mkdir

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

rcParams['figure.figsize'] = [5, 3]
rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.size'] = 13
rcParams['font.stretch'] = 'condensed'
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


N = 16  # period
num_period = 5
shifts = [-2, -1, 0, 1, 2]
labels = ['$h[n+2N]$', '$h[n+N]$', '$h[n]$', '$h[n-N]$', '$h[n-2N]$']

# Discrete-time impulse response
phi = -np.pi / 4
nmin, nmax = -2 * N, 2 * N
n = np.arange(nmin, nmax + 1)
irs = [util.discrete_ir_constant_phase(n - i * N, phi) for i in shifts]

# Periodic impulse resposne
nN = np.arange(N)
hp = util.periodic_constant_phase_shifter_ir(N, phi)

# Plot
ymin, ymax = -num_period - 0.75, 1
xpos_label = 8
voffset_label = 0.2
col0 = 'C0'
col1 = 'C3'

fig, ax = plt.subplots(figsize=(8, 6))

# Shade single period [0, ..., N-1]
plt.fill_between(x=[nN.min() - 0.5, nN.max() + 0.5],
                 y1=[ymin, ymin], y2=[ymax, ymax], color='c')

# Shifted impulse resposnes
for i, h in enumerate(irs):
    ax.stem(n, h - i, bottom=-i,
            markerfmt=col0 + 'o', linefmt=col0, basefmt=col0)
    ax.text(xpos_label, -i + voffset_label, labels[i],
            va='bottom', ha='center')

# Periodic impulse respose (DFT)
ax.stem(nN, hp - num_period, bottom=-num_period,
        markerfmt=col1 + 'o', linefmt=col1, basefmt=col1)
ax.text(xpos_label, -num_period + voffset_label, r'$\tilde{h}[n]$',
        ha='center')

ax.set_xlabel('$n$ / sample')
ax.set_yticklabels([])
ax.set_xlim(nmin - 0.5, nmax + 0.5)
ax.set_ylim(ymin, ymax)

filename = 'periodic-ir'
ext = '.pdf'
plt.savefig(dir_fig + filename + ext, dpi=300, bbox_inches='tight')
