import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import rcParams
from os import path, mkdir

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

rcParams['axes.linewidth'] = 0.5
rcParams['axes.edgecolor'] = 'gray'
rcParams['axes.facecolor'] = 'None'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'gray'
rcParams['ytick.color'] = 'gray'
rcParams['font.family'] = 'sans serif'
rcParams['font.size'] = 13
rcParams['text.usetex'] = False
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'


def get_time(data, prefix):
    """Decision time for a given stimuli"""
    d = [di for di in data[1:] if re.match(re.compile(prefix), di[4])]
    return np.stack([float(di[8]) for di in d])


# Load listening experiment result
csvfile = '../abx_software/webMUSHRA_c929877_20180814/results' + \
          '/abx_constant_phase_shift/paired_comparison.csv'
data = np.genfromtxt(csvfile, dtype=None, delimiter=',', encoding=None)

# Decision time for each stimuli
stimuli = ['square_a', 'square_b', 'pnoise', 'castanets', 'hotelcalifornia']
t = np.stack([get_time(data, st) for st in stimuli], axis=-1)

# labels
a_label = ['square wave\n($-90\\degree$)',
           'square wave\n($-45\\degree$)',
           'pink noise\n($-90\\degree$)',
           'castanets\n($-90\\degree$)',
           'hotel california\n($-90\\degree$)']

# colors for the elements in the box plot
col_med = 'C0'
col_box = 'Gray'
col_flr = 'C0'
col_wsk = 'Gray'
col_cap = col_wsk

# Box plot properties
medianprops = dict(linestyle='-', linewidth=3, color=col_med)
boxprops = dict(linestyle='-', linewidth=2, color=col_box)
flierprops = dict(marker='P', markerfacecolor=col_flr, markeredgecolor='none',
                  markersize=8, alpha=0.33)
whiskerprops = dict(linewidth=2, color=col_wsk)
capprops = dict(linewidth=2, color=col_cap)

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6.5))
for i, ti in enumerate(t.T):
    ax.plot((i + 1) * np.ones_like(ti), ti / 1000, linestyle='none',
            marker='P', markersize=8, markerfacecolor=col_flr,
            markeredgecolor='none', alpha=0.33)
ax.set_xlim(0.5, 5.7)
ax.set_ylim(0, 155)
ax.set_yticks(np.arange(0, 160, 30))
ax.set_xticks(np.arange(1, 6))
ax.set_xticklabels(a_label, fontsize=11.5)
ax.set_xlabel('Stimulus (phase shift)')
ax.set_ylabel('Trial Decision Time / s')
ax.yaxis.grid(color='gray', alpha=0.5)

plt.savefig(dir_fig + 'scatterplot_time.pdf')


# Box plot
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.boxplot(t / 1000, whis=[5, 95], widths=0.5,
           medianprops=medianprops, boxprops=boxprops, flierprops=flierprops,
           whiskerprops=whiskerprops, capprops=capprops)
ax.set_xlim(0.5, 5.7)
ax.set_ylim(0, 155)
ax.set_yticks(np.arange(0, 160, 30))
ax.set_xticklabels(a_label, fontsize=11.5)
ax.set_xlabel('Stimulus (phase shift)')
ax.set_ylabel('Trial Decision Time / s')
ax.yaxis.grid(color='gray', alpha=0.5)

# Legend
x_label = 5.3
t_label = [np.percentile(t[:, -1], p) / 1000 for p in [5, 25, 50, 75, 95]]
p_label = ['P$_{05}$', 'P$_{25}$', 'P$_{50}$', 'P$_{75}$', 'P$_{95}$']
colors = [col_cap, col_box, col_med, col_box, col_cap]
for tl, pl, col in zip(t_label, p_label, colors):
    ax.text(x_label, tl, pl, fontsize=15, color=col, va='center', ha='left')

plt.savefig(dir_fig + 'boxplot_time.pdf')
