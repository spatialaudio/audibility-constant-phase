import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import binom
from matplotlib import rcParams, cm

dir_fig = '../paper/graphics/'

rcParams['figure.figsize'] = [6, 6]
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

voffset_count = -1.5
fill_color = cm.get_cmap('viridis')(0.60)
marker_size_min = 35
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


fig, ax = plt.subplots()

ax.fill_between([-1, len(audio)], [p_H0, p_H0], [pmax, pmax],
                color=fill_color, lw=0, alpha=0.15)
ax.text(4.35, 78, '$\\mathcal{H}_0$ rejection region \n ($p \\ge 19/25 = 76\%$)',
        fontsize=12, ha='right', va='bottom')
for i, a in enumerate(audio):
    elements, counts = np.unique(h[:, i], return_counts=True)
    xaxis = i * np.ones(len(elements))
    marker_size = marker_size_min * counts**marker_size_exp
    ax.scatter(xaxis, elements, s=marker_size,
               color=marker_color, alpha=marker_opacity, lw=0, zorder=2)
    idx_multiple = counts > 1
    for cnt, prb in zip(counts[idx_multiple], elements[idx_multiple]):
        ax.text(i, prb + voffset_count, cnt, color='w',
                ha='center', va='bottom', fontsize=11, zorder=3)
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
ax.set_ylabel(r'Detection Rate / \%')
plt.savefig(dir_fig + 'scatter.pdf', bbox_inches='tight')
