"""Render the results comparison figure from the reported metrics.

Greyscale safe: the two bars in each pair differ by fill and hatch, not by
colour, so the figure survives black and white printing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARCH = ['Compact\nnetwork', 'ResNet18', 'MobileNetV2', 'DenseNet201']
ACC_OFF = [93.45, 95.76, 95.76, 96.15]
ACC_ON = [93.86, 96.53, 96.11, 96.09]
F1_OFF = [0.9508, 0.9693, 0.9686, 0.9719]
F1_ON = [0.9545, 0.9745, 0.9716, 0.9715]

fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
x = np.arange(len(ARCH))
w = 0.36

for k, (off, on, lab, lo, hi, fmt) in enumerate([
        (ACC_OFF, ACC_ON, 'Accuracy (\\%)', 92.5, 97.9, '%.2f'),
        (F1_OFF, F1_ON, '$F_1$ score', 0.940, 0.9835, '%.4f')]):
    a = ax[k]
    b1 = a.bar(x - w / 2, off, w, facecolor='white', edgecolor='black',
               linewidth=0.9, label='Without denoising')
    b2 = a.bar(x + w / 2, on, w, facecolor='0.62', edgecolor='black',
               linewidth=0.9, hatch='///', label='With wavelet denoising')
    for bars in (b1, b2):
        for r in bars:
            a.annotate(fmt % r.get_height(),
                       (r.get_x() + r.get_width() / 2, r.get_height()),
                       textcoords='offset points', xytext=(0, 2),
                       ha='center', fontsize=7.0)
    a.set_xticks(x)
    a.set_xticklabels(ARCH, fontsize=8.5)
    a.set_ylim(lo, hi)
    a.set_ylabel(lab, fontsize=9.5)
    a.tick_params(axis='y', labelsize=8)
    a.grid(axis='y', color='0.85', linewidth=0.6)
    a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)

ax[0].set_title('(a) Accuracy', fontsize=10, pad=8)
ax[1].set_title('(b) $F_1$ score', fontsize=10, pad=8)
ax[0].legend(fontsize=8, frameon=False, loc='upper left', ncol=1)

fig.tight_layout()
fig.savefig('figs/results_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white')
print('written figs/results_comparison.png')
