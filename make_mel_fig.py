"""Illustrate the perceptual frequency mapping and the filter bank built on it.

Both panels are the mapping of Equation (3) evaluated directly, so the figure
contains no measured data and depends on no recording.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FS, NFILT, FMAX = 16000, 128, 8000


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8))

# (a) the warping itself, against a linear reference
f = np.linspace(0, FMAX, 2000)
ax[0].plot(f, hz_to_mel(f), color='black', linewidth=1.6,
           label='Perceptual mapping')
ax[0].plot([0, FMAX], [0, hz_to_mel(FMAX)], color='0.55', linewidth=1.1,
           linestyle='--', label='Linear reference')
ax[0].axvline(700, color='0.3', linewidth=0.8, linestyle=':')
ax[0].annotate('700 Hz', (700, hz_to_mel(FMAX) * 0.07), xytext=(8, 0),
               textcoords='offset points', fontsize=8, color='0.25')
ax[0].set_xlabel('Frequency (Hz)', fontsize=9.5)
ax[0].set_ylabel('Perceptual units', fontsize=9.5)
ax[0].set_xlim(0, FMAX)
ax[0].set_ylim(0, hz_to_mel(FMAX) * 1.02)
ax[0].legend(fontsize=8, frameon=False, loc='lower right')
ax[0].set_title('(a) Frequency mapping', fontsize=10, pad=8)

# (b) a reduced bank, uniform on the warped axis, drawn against hertz
SHOW = 16
edges_mel = np.linspace(0, hz_to_mel(FMAX), SHOW + 2)
edges_hz = mel_to_hz(edges_mel)
for i in range(SHOW):
    lo, mid, hi = edges_hz[i], edges_hz[i + 1], edges_hz[i + 2]
    shade = '0.25' if i % 2 == 0 else '0.65'
    ax[1].plot([lo, mid, hi], [0, 1, 0], color=shade, linewidth=1.1)
ax[1].set_xlabel('Frequency (Hz)', fontsize=9.5)
ax[1].set_ylabel('Filter weight', fontsize=9.5)
ax[1].set_xlim(0, FMAX)
ax[1].set_ylim(0, 1.12)
ax[1].set_title('(b) Triangular filter bank', fontsize=10, pad=8)

for a in ax:
    a.tick_params(labelsize=8)
    a.grid(color='0.88', linewidth=0.6)
    a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig('figs/mel_mapping.png', dpi=300, bbox_inches='tight',
            facecolor='white')
print('written figs/mel_mapping.png')
print('first filter spans  %.0f Hz' % (mel_to_hz(edges_mel[2]) - mel_to_hz(edges_mel[0])))
print('last filter spans   %.0f Hz' % (mel_to_hz(edges_mel[-1]) - mel_to_hz(edges_mel[-3])))
