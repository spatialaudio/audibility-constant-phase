import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import binom
from matplotlib import rcParams, cm
from os import path, mkdir

dir_fig = '../talk/graphics/'
if not path.exists(dir_fig):
    mkdir(dir_fig)

rcParams['figure.figsize'] = [6, 6]
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


def calc_p(correct_trials, total_trials):
    # return probability of at least 'correct_trials' out of 'total_trials'
    # with 50:50 chance coin flipping
    return 1-binom.cdf(correct_trials-1, total_trials, 0.5)


audio = ['square_a',
         'square_b',
         'pnoise',
         'castanets',
         'hotelcalifornia']
p_H0 = (19 / 25) * 100  # H0 rejection boarder in %

csvfile = '../abx_software/webMUSHRA_c929877_20180814/results' + \
          '/abx_constant_phase_shift/paired_comparison.csv'
data = np.genfromtxt(csvfile, dtype=None, delimiter=',', encoding=None)
vp_id = np.unique([di for di in data[1:, 1]]).astype('int')
num_vp = len(vp_id)

x_total = np.zeros((num_vp, len(audio)), dtype='int')
x_correct = np.zeros((num_vp, len(audio)), dtype='int')
for vp in range(num_vp):
    d = [di for di in data[1:] if int(di[1]) == vp]
    for i, a in enumerate(audio):
        answers = [di[7] for di in d if re.match(re.compile(a), di[4])]
        x_total[vp, i] = len(answers)
        x_correct[vp, i] = answers.count('correct')

p = calc_p(x_correct, x_total)
h = x_correct / x_total * 100  # detection frequency correct answers

voffset_count = -0.3
hoffset_count = 0.004
fill_color = cm.get_cmap('viridis')(0.60)
marker_size_min = 45
marker_size_exp = 1.75
marker_color = cm.get_cmap('viridis')(0.05)
marker_opacity = 0.65
alabel = ('square wave\n($-90\\degree$)',
          'square wave\n($-45\\degree$)',
          'pink noise\n($-90\\degree$)',
          'castanets\n($-90\\degree$)',
          'hotel california\n($-90\\degree$)')
pmin, pmax = 23, 105
cmin, cmax = [pi * 0.25 for pi in [pmin, pmax]]
second_yaxis = True
yticks = np.arange(0, 125, 25)


fig, ax = plt.subplots(figsize=(8, 6.5))

ax.fill_between([-1, len(audio)], [p_H0, p_H0], [pmax + 50, pmax + 50],
                color=fill_color, lw=0, alpha=0.15)
ax.text(4.35, 78,
        r'$\mathcal{H}_0$ rejection region' + '\n' + r'($p \geq 19/25 = 76%$)',
        fontsize=12, ha='right', va='bottom')
for i, a in enumerate(audio):
    elements, counts = np.unique(h[:, i], return_counts=True)
    xaxis = i * np.ones(len(elements))
    marker_size = marker_size_min * counts**marker_size_exp
    ax.scatter(xaxis, elements, s=marker_size,
               color=marker_color, alpha=marker_opacity, lw=0, zorder=2)
    idx_multiple = counts > 1
    for cnt, prb in zip(counts[idx_multiple], elements[idx_multiple]):
        ax.text(i + hoffset_count, prb + voffset_count, cnt, color='w',
                ha='center', va='center', fontsize=11, zorder=3)
if second_yaxis:
    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(0, 30, 5))
    ax2.set_yticks(np.arange(26), minor=True)
    ax2.set_ylim(cmin, cmax)
    ax2.set_ylabel('Number of Correct Answers')
plt.xticks(np.arange(len(audio)))
ax.set_xticklabels(alabel, fontsize=11.5)
ax.set_yticks(yticks)
ax.set_xlim(-0.45, len(audio) - 1 + 0.45)
ax.set_ylim(pmin, pmax)
ax.yaxis.grid(color='lightgray', alpha=0.5)
ax.set_xlabel('Stimulus (phase shift)')
ax.set_ylabel(r'Detection Rate / %')
plt.savefig(dir_fig + 'scatter.pdf', bbox_inches='tight')

# Post hoc Chi-square test
fig.set_size_inches(8, 7.2)
y1_chitest = 104  # square wave 90 vs pink noise, castanets, hotel california
y2_chitest = 110  # square wave 45 vs pink noise, castanets, hotel california
y3_chitest = 55  # square wave 90 vs 45
y4_chitest = 29
y1 = np.array([y1_chitest, y1_chitest])
y2 = np.array([y2_chitest, y2_chitest])
y3 = np.array([y3_chitest, y3_chitest])
y4 = np.array([y4_chitest, y4_chitest])
dy = 1.5
col_chitest = fill_color

ax.set_ylim(pmin, pmax + 12)
ax.plot([0, 2], y1 + 2 * dy, color=col_chitest, marker='|')
ax.plot([0, 3], y1 + dy, color=col_chitest, marker='|')
ax.plot([0, 4], y1, color=col_chitest, marker='|')
ax.text(0, y1_chitest + 2 * dy, r'$\ast$$\ast$$\ast$(post hoc)',
        fontsize=12, ha='left', va='bottom', color=col_chitest)

ax.plot([1, 2], y2 + 2 * dy, color=col_chitest, marker='|')
ax.plot([1, 3], y2 + dy, color=col_chitest, marker='|')
ax.plot([1, 4], y2, color=col_chitest, marker='|')
ax.text(1, y2_chitest + 2 * dy, r'$\ast$$\ast$$\ast$(post hoc)',
        fontsize=12, ha='left', va='bottom', color=col_chitest)

ax.plot([2, 3], y4 + 2 * dy, color=col_chitest, marker='|')
ax.plot([2, 4], y4 + dy, color=col_chitest, marker='|')
ax.plot([3, 4], y4, color=col_chitest, marker='|')

ax.plot([0, 1], y3, color=col_chitest, marker='|')
ax.text(0.5, y3_chitest, 'ns', ha='center', va='bottom', color=col_chitest)
ax.text(2.5, y4_chitest + 2 * dy, 'ns',
        ha='center', va='bottom', color=col_chitest)

ax.text(0, 30, r'$\ast$$\ast$$\ast$: $p<0.001$',
        ha='left', va='bottom', color=col_chitest)
ax.text(0, 26, 'ns: not significant',
        ha='left', va='bottom', color=col_chitest)

plt.savefig(dir_fig + 'scatter-with-chitest.pdf', bbox_inches='tight')
