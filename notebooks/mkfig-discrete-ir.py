import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib import rcParams

dir_fig = '../paper/graphics/'

rcParams['figure.figsize'] = [5, 3]
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

phi = -np.pi / 4
nmin, nmax = -16, 16
n = np.arange(nmin, nmax + 1)
h = util.discrete_ir_constant_phase(n, phi)
h[n == 0] = None

plt.figure()
plt.stem([0], [np.cos(phi)], markerfmt='C3o', linefmt='C3', basefmt='C3',
         label=r'$\cos\varphi\cdot\delta[n]$')
plt.stem(n, h, markerfmt='C0o', linefmt='C0', basefmt='')
plt.grid(color='lightgray')
plt.xlabel('$n$ / sample')
plt.ylabel('$h[n]$')
plt.xlim(nmin - 0.5, nmax + 0.5)
plt.ylim(-0.55, 0.8)

filename = 'discrete-ir-phi{:03.0f}.pdf'.format(np.rad2deg(phi))
plt.savefig(dir_fig + filename, dpi=300, bbox_inches='tight')
