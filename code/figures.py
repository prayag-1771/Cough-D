"""Produce the manuscript figures from the corpora and the saved decisions.

    python -I figures.py --what all
    python -I figures.py --what denoise --ext png --outdir /tmp/look

Each figure is written as a PDF into the manuscript's figure directory. Nothing
here recomputes a result: the decision figures read the same per recording
files the tables are built from, so a figure and a table can never disagree.

The denoising figure is the exception, in that it has to decode audio. It takes
its examples from the classifier derived corpus, which is the only one whose
archive is kept locally after imaging, and the caption says so. What the figure
illustrates is the behaviour of the shrinkage rule on a waveform, which does
not depend on how the recording came to be labelled.
"""
import argparse
import csv
import glob
import io
import os
import re
import statistics as st
import sys
import zipfile

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import denoise
import spectro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'manuscript', 'figs')
MS = os.path.join(ROOT, 'manuscript')
ARCHIVE = os.path.join(ROOT, 'data', 'coughvid_v3', 'public_dataset_v3.zip')
META = os.path.join(ROOT, 'data', 'coughvid_v3', 'coughvid_20211012',
                    'metadata_compiled.csv')

# The manuscript takes vector figures; a raster copy is useful only for looking
# at one quickly, so the extension is settable and defaults to the real output.
EXT = 'pdf'
OUTDIR = None

# Black for the signal and one accent for what was removed. The figures are
# read in print as often as on a screen, so nothing depends on hue.
KEPT = '#1a1a1a'
CUT = '#c03028'

plt.rcParams.update({
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 200,
})


def shortpath(p):
    """Path relative to the project when it lies inside it, else as given."""
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def pick_examples(n_each=1):
    """Recordings the shipped detector scores as clearly one class or other."""
    rows = list(csv.DictReader(io.open(META, encoding='utf-8')))
    hi, lo = [], []
    for r in rows:
        v = r.get('cough_detected') or ''
        try:
            v = float(v)
        except ValueError:
            continue
        if v >= 0.9:
            hi.append((r['uuid'], v))
        elif v <= 0.1:
            lo.append((r['uuid'], v))
    hi.sort(key=lambda t: -t[1])
    lo.sort(key=lambda t: t[1])
    return hi[:n_each], lo[:n_each]


def decode_from_archive(uuid, work):
    """Decode one recording out of the archive, whatever container it uses."""
    with zipfile.ZipFile(ARCHIVE) as z:
        names = [n for n in z.namelist()
                 if os.path.basename(n).startswith(uuid)
                 and not n.endswith('.json')]
        if not names:
            return np.zeros(0, dtype=np.float32)
        member = names[0]
        os.makedirs(work, exist_ok=True)
        tmp = os.path.join(work, os.path.basename(member))
        with z.open(member) as src, open(tmp, 'wb') as dst:
            dst.write(src.read())
    x = spectro.decode(tmp)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return x


def usable(x):
    """Reject the truncated uploads, which carry no signal to shrink."""
    return (x.size >= spectro.SR // 2 and np.isfinite(x).all()
            and float(np.max(np.abs(x))) > 1e-4)


def window(x):
    """A window around the loudest point, so the event is on screen."""
    span = int(0.9 * spectro.SR)
    c = int(np.argmax(np.abs(x)))
    a = max(0, min(c - span // 2, x.size - span))
    return x[a:a + span]


def audit_shrinkage(work, n=200):
    """How much of each recording the shrinkage actually removes.

    The denoising results turn out to be null, and this measures the reason
    rather than leaving it to be inferred. The universal threshold is set from
    an estimate of the noise level, so on a recording that is already clean it
    falls far below the signal and almost nothing is shrunk. If that is what is
    happening then the two input conditions are very nearly the same images,
    and no difference between them was ever available to be found.
    """
    hi, lo = pick_examples(n)
    rows = []
    seen = {'visu': [], 'bayes': []}
    for (uuid, score), lab in ([(h, 'cough') for h in hi]
                               + [(l, 'non_cough') for l in lo]):
        x = decode_from_archive(uuid, work)
        if not usable(x):
            continue
        seg = window(x)
        power = float(np.sum(seg ** 2))
        row = {'uuid': uuid, 'label': lab, 'score': '%.4f' % score}
        for rule in ('visu', 'bayes'):
            cut = seg - denoise.denoise(seg, rule=rule)
            frac = float(np.sum(cut ** 2)) / power
            row[rule] = '%.3e' % frac
            seen[rule].append(frac)
        rows.append(row)

    out = os.path.join(ROOT, 'results', 'shrinkage.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['uuid', 'label', 'score', 'visu',
                                           'bayes'])
        w.writeheader()
        w.writerows(rows)
    print('  wrote %s  (%d recordings)' % (shortpath(out), len(rows)))

    # The manuscript quotes the summary, not the per recording column, so the
    # summary is written down too rather than left in a console log.
    summary = os.path.join(ROOT, 'results', 'shrinkage_summary.csv')
    with open(summary, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['rule', 'recordings', 'median_pct', 'q1_pct', 'q3_pct',
                    'max_pct'])
        for rule in ('visu', 'bayes'):
            v = sorted(seen[rule])
            if not v:
                continue
            q1, med, q3 = v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4]
            w.writerow([rule, len(v), '%.4f' % (100 * med), '%.4f' % (100 * q1),
                        '%.4f' % (100 * q3), '%.4f' % (100 * v[-1])])
            print('    %-6s removed power: median %.4f per cent, quartiles '
                  '%.4f and %.4f, max %.4f'
                  % (rule, 100 * med, 100 * q1, 100 * q3, 100 * v[-1]))
    print('  wrote %s' % shortpath(summary))
    return seen


def _mag(r):
    """A magnification factor a reader can hold in mind, grouped in threes."""
    s = '%d' % round(r)
    out = []
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return ','.join(out)


def fig_denoise(work):
    """What the shrinkage removes, for a cough and for a non-cough.

    The residual is drawn on its own scale. Plotted against the signal it is a
    flat line at zero, which is the finding but not a legible figure, so the
    magnification needed to see it is stated on the panel instead.
    """
    hi, lo = pick_examples(8)
    panels = []
    for cands, label in ((hi, 'cough'), (lo, 'non-cough')):
        for uuid, _score in cands:
            x = decode_from_archive(uuid, work)
            if usable(x):
                seg = window(x)
                panels.append((label, seg, denoise.denoise(seg, rule='visu')))
                break
        else:
            sys.exit('could not decode an example for %s' % label)

    fig, ax = plt.subplots(2, 2, figsize=(6.6, 3.4), sharex=True)
    t = np.arange(panels[0][1].size) / float(spectro.SR)
    for i, (label, seg, kept) in enumerate(panels):
        cut = seg - kept
        lim = 1.05 * float(np.max(np.abs(seg)))
        # Headroom above the residual so the annotation does not sit on it.
        clim = 1.8 * float(np.max(np.abs(cut))) or 1.0
        frac = float(np.sum(cut ** 2)) / float(np.sum(seg ** 2))

        ax[i][0].plot(t, seg, color=KEPT, lw=0.4)
        ax[i][0].set_ylim(-lim, lim)
        ax[i][0].set_ylabel(label + '\namplitude')

        ax[i][1].plot(t, cut, color=CUT, lw=0.4)
        ax[i][1].set_ylim(-clim, clim)
        note = ('magnified %s times, %.4f per cent of signal power'
                % (_mag(lim / clim), 100.0 * frac))
        ax[i][1].text(0.98, 0.93, note, transform=ax[i][1].transAxes,
                      ha='right', va='top', fontsize=6.5, color=CUT)

    ax[0][0].set_title('as recorded', fontsize=8)
    ax[0][1].set_title('what the shrinkage removed', fontsize=8)
    ax[1][0].set_xlabel('time (s)')
    ax[1][1].set_xlabel('time (s)')
    fig.tight_layout()
    out = os.path.join(OUTDIR or FIGS, 'denoise.' + EXT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % shortpath(out))


def read_runs(d):
    """{(arch, cond): [balanced accuracy per seed]} from decision files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
        bits = os.path.basename(f)[:-4].split('__')
        if len(bits) != 3:
            continue
        hit = {0: [0, 0], 1: [0, 0]}
        for r in csv.DictReader(io.open(f, encoding='utf-8')):
            k = 0 if r['true'] == 'cough' else 1
            hit[k][0] += 1
            hit[k][1] += (r['pred'] == r['true'])
        bal = 0.5 * sum(hit[k][1] / float(hit[k][0]) for k in (0, 1)
                        if hit[k][0])
        out.setdefault((bits[0], bits[1]), []).append(100.0 * bal)
    return out


def fig_transfer():
    """What each configuration scores in corpus, and what it keeps outside."""
    inc = read_runs(os.path.join(ROOT, 'runs_coswara'))
    ext = read_runs(os.path.join(ROOT, 'runs_fsd50k'))
    shared = sorted(set(inc) & set(ext))
    if not shared:
        print('  transfer: no overlapping conditions yet, skipped')
        return

    archs = sorted({a for a, _c in shared})
    tone = dict(zip(archs, ('#1a1a1a', '#c03028', '#2b5d8a', '#6a7f3a')))
    marks = dict(zip(archs, ('o', 's', '^', 'D')))

    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    for arch in archs:
        keys = [k for k in shared if k[0] == arch]
        col = tone.get(arch, KEPT)
        for n, k in enumerate(keys):
            a, b = st.mean(inc[k]), st.mean(ext[k])
            sa = st.stdev(inc[k]) if len(inc[k]) > 1 else 0.0
            sb = st.stdev(ext[k]) if len(ext[k]) > 1 else 0.0
            ax.plot([0, 1], [a, b], color=col, lw=0.7, alpha=0.75,
                    zorder=2, label=arch if n == 0 else None)
            ax.errorbar([0], [a], yerr=[sa], fmt=marks.get(arch, 'o'),
                        color=col, mfc='none', ms=3.6, lw=0.6, capsize=1.5,
                        zorder=3)
            ax.errorbar([1], [b], yerr=[sb], fmt=marks.get(arch, 'o'),
                        color=col, mfc='none', ms=3.6, lw=0.6, capsize=1.5,
                        zorder=3)

    ax.set_xlim(-0.26, 1.26)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['within the\nprompted corpus',
                        'human annotated\naudio'])
    ax.set_ylabel('balanced accuracy (per cent)')
    ax.tick_params(axis='x', length=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc='lower left',
              handlelength=1.4, borderaxespad=0.1)
    out = os.path.join(OUTDIR or FIGS, 'transfer.' + EXT)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print('  wrote %s' % shortpath(out))


def fig_shrinkage():
    """How much signal power each shrinkage rule removes, over the audit set.

    The null result on denoising is easier to accept once it is clear how
    little the rules take out, and the quantity is already measured, so it is
    worth a figure rather than a sentence.
    """
    path = os.path.join(ROOT, 'results', 'shrinkage.csv')
    if not os.path.exists(path):
        print('  shrinkage: results/shrinkage.csv absent, skipped')
        return
    uni, eb = [], []
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        try:
            uni.append(100.0 * float(r['visu']))
            eb.append(100.0 * float(r['bayes']))
        except (KeyError, ValueError):
            continue
    if not uni:
        print('  shrinkage: no usable rows, skipped')
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for vals, lab, col in ((uni, 'universal threshold', KEPT),
                           (eb, 'empirical Bayes', '#c03028')):
        v = sorted(x for x in vals if x > 0)
        if not v:
            continue
        y = [100.0 * (i + 1) / len(v) for i in range(len(v))]
        ax.plot(v, y, color=col, lw=0.9, label=lab)
    ax.set_xscale('log')
    ax.set_xlabel('signal power removed (per cent)')
    ax.set_ylabel('recordings below (per cent)')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    out = os.path.join(OUTDIR or FIGS, 'shrinkage.' + EXT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % shortpath(out))


def fig_falsepos():
    """The rate at which each annotated class is called a cough."""
    path = os.path.join(MS, 'tab_confusion.tex')
    if not os.path.exists(path):
        print('  falsepos: tab_confusion.tex absent, skipped')
        return
    rows = []
    for line in io.open(path, encoding='utf-8'):
        m = re.match(r'([A-Z][A-Za-z, ]+?) & (\d+) & (\d+) & \$([\d.]+)\$',
                     line.strip())
        if m:
            rows.append((m.group(1).strip(), float(m.group(4))))
    if not rows:
        print('  falsepos: no rows parsed, skipped')
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(3.3, 0.22 * len(rows) + 0.7))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cough = [i for i, n in enumerate(names) if n.lower().startswith('cough')]
    cols = ['#c03028' if i in cough else KEPT for i in range(len(rows))]
    ax.barh(range(len(rows)), vals, color=cols, height=0.65)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('decisions assigning the clip to cough (per cent)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    out = os.path.join(OUTDIR or FIGS, 'falsepos.' + EXT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % shortpath(out))


def fig_effect():
    """Each shrinkage effect against the seed spread it has to clear."""
    path = os.path.join(ROOT, 'results', 'paired.csv')
    if not os.path.exists(path):
        print('  effect: results/paired.csv absent, skipped')
        return
    pts = []
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        try:
            pts.append((abs(float(r['accuracy_change'])),
                        float(r['change_sd']),
                        float(r['p_adjusted'])))
        except (KeyError, ValueError):
            continue
    if not pts:
        print('  effect: no usable rows, skipped')
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    lo = min(min(p[0] for p in pts if p[0] > 0),
             min(p[1] for p in pts if p[1] > 0)) * 0.55
    hi = max(max(p[0] for p in pts), max(p[1] for p in pts)) * 1.8
    ax.plot([lo, hi], [lo, hi], ls=':', lw=0.7, color='#888888', zorder=1)
    seen = set()
    for eff, sd, padj in pts:
        sig = padj < 0.05
        lab = ('detectable effect' if sig else 'no detectable effect')
        ax.plot([max(eff, lo)], [max(sd, lo)], 's' if sig else 'o', ms=4.5,
                color='#c03028' if sig else KEPT, mfc='none', lw=0.8,
                ls='none', zorder=3,
                label=None if lab in seen else lab)
        seen.add(lab)
    # A decade either way of the data, so the nine comparisons separate
    # instead of piling into the corner behind the one large spread.
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(hi * 0.055, hi * 0.105, 'spread exceeds the effect', rotation=45,
            fontsize=5.5, color='#888888', ha='center', va='bottom')
    ax.legend(frameon=False, fontsize=6, loc='lower right',
              handlelength=1.2, borderaxespad=0.1)
    ax.set_xlabel('size of the effect (accuracy points)')
    ax.set_ylabel('spread across seeds (points)')
    out = os.path.join(OUTDIR or FIGS, 'effect.' + EXT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % shortpath(out))


def main():
    global EXT, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', default='all',
                    choices=('all', 'denoise', 'transfer', 'shrinkage',
                             'power', 'falsepos', 'effect'))
    ap.add_argument('--work', default=None)
    ap.add_argument('--ext', default='pdf')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()

    EXT, OUTDIR = a.ext, a.outdir
    os.makedirs(FIGS, exist_ok=True)
    if OUTDIR:
        os.makedirs(OUTDIR, exist_ok=True)
    work = a.work or os.path.join(ROOT, 'data', 'fig_work')

    if a.what in ('all', 'denoise'):
        fig_denoise(work)
    if a.what in ('all', 'transfer'):
        fig_transfer()
    if a.what in ('all', 'shrinkage'):
        audit_shrinkage(work, a.n)
    if a.what in ('all', 'power'):
        fig_shrinkage()
    if a.what in ('all', 'falsepos'):
        fig_falsepos()
    if a.what in ('all', 'effect'):
        fig_effect()


if __name__ == '__main__':
    main()
