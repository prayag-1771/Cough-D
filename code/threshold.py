"""Separate loss of discrimination from loss of calibration.

    python -I threshold.py

A fall in balanced accuracy has two quite different possible causes, and they
call for different responses. The model may no longer rank coughs above non
coughs, in which case the representation does not transfer and nothing applied
afterwards will recover it. Or it may still rank them correctly while the
decision threshold inherited from training now sits in the wrong place, in
which case the loss is largely a calibration problem and a practitioner can
address it with held out data from the target domain.

The area under the receiver operating characteristic separates these, because
it depends only on the ordering of the scores and not on where the threshold
falls. It is computed here from the ranks, which is equivalent to the
Mann Whitney statistic and needs no external dependency.

The best balanced accuracy attainable at any threshold is also reported. That
quantity is chosen on the very set it is evaluated on, so it is not an
achievable operating point and is not presented as one. It is an upper bound:
the most that moving the threshold could possibly recover, which is the figure
needed to decide whether moving it is worth attempting at all.
"""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, 'manuscript')

ARCH = {'resnet18': 'ResNet18', 'mobilenetv2': 'MobileNetV2',
        'compact': 'Compact', 'densenet201': 'DenseNet201'}
ORDER = ['resnet18', 'mobilenetv2', 'densenet201', 'compact']
CONDS = ['linear', 'wav_linear', 'mel', 'wav_mel']


def scored(path):
    """[(score for cough, is cough)] for one run."""
    out = []
    with io.open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            s = r.get('score_1')
            if s is None or s == '':
                return []
            out.append((float(s), r['true'].strip() == A.POS))
    return out


def auc(pairs):
    """Rank based area under the curve, ties averaged."""
    pos = sum(1 for _s, y in pairs if y)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float('nan')
    order = sorted(pairs, key=lambda t: t[0])
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    rsum = sum(ranks[k] for k, (_s, y) in enumerate(order) if y)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def balanced_at(pairs, t):
    tp = sum(1 for s, y in pairs if y and s >= t)
    fn = sum(1 for s, y in pairs if y and s < t)
    tn = sum(1 for s, y in pairs if not y and s < t)
    fp = sum(1 for s, y in pairs if not y and s >= t)
    rec = tp / float(tp + fn) if tp + fn else 0.0
    spec = tn / float(tn + fp) if tn + fp else 0.0
    return (rec + spec) / 2.0


def best_threshold(pairs):
    """The threshold maximising balanced accuracy, and the value attained."""
    best, at = -1.0, 0.5
    for t in sorted({s for s, _y in pairs} | {0.5}):
        v = balanced_at(pairs, t)
        if v > best:
            best, at = v, t
    return best, at


def collect(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.csv'):
            continue
        m = A.TAG.match(fn[:-4])
        if not m:
            continue
        pairs = scored(os.path.join(root, fn))
        if not pairs:
            continue
        hi, at = best_threshold(pairs)
        out.setdefault((m.group('arch'), m.group('img')), []).append(
            dict(auc=auc(pairs), default=balanced_at(pairs, 0.5),
                 best=hi, at=at))
    return out


def pm(xs, places=2, scale=100.0):
    m, s = A.mean_sd([scale * x for x in xs])
    return '$%.*f \\pm %.*f$' % (places, m, places, s)


def main():
    inc = collect(os.path.join(ROOT, 'runs_coswara'))
    ext = collect(os.path.join(ROOT, 'runs_fsd50k'))
    shared = [(a, c) for a in ORDER for c in CONDS
              if (a, c) in inc and (a, c) in ext]
    if not shared:
        print('  no overlapping conditions yet, skipped')
        return

    body = []
    last = None
    for arch, cond in shared:
        i, e = inc[(arch, cond)], ext[(arch, cond)]
        if last is not None and arch != last:
            body.append('\\addlinespace')
        body.append(
            '%s & %s & %s & %s & %s & %s \\\\'
            % (ARCH.get(arch, arch) if arch != last else '', A.short(cond),
               pm([x['auc'] for x in i], 3, 1.0),
               pm([x['auc'] for x in e], 3, 1.0),
               pm([x['default'] for x in e]),
               pm([x['best'] for x in e])))
        last = arch

    head = [
        # A two column class gives a single column about three and a
        # half inches wide, which takes four columns of this kind of
        # content and not six or more, so this one is a page wide
        # float. A single column class sets a starred float exactly
        # as it sets an ordinary one, so one source serves both.
        '\\begin{table*}[tbp]',
        '\\caption{Discrimination against calibration on the external set. The '
        'area under the curve depends only on the ordering of the scores, so a '
        'value that holds up while balanced accuracy falls indicates that the '
        'ranking survived and the inherited threshold did not. The final '
        'column is the balanced accuracy attainable at the best threshold for '
        'this set. That threshold is chosen on the same recordings it is '
        'scored on, so the column is an upper bound on what recalibration '
        'could recover and not an operating point anyone could reach in '
        'advance.}\\label{tab:threshold}',
        '\\small',
        '\\begin{tabular}{@{}llcccc@{}}',
        '\\toprule',
        '& & \\multicolumn{2}{c}{Area under the curve} & '
        '\\multicolumn{2}{c}{External balanced accuracy} \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
        'Architecture & Input & In corpus & External & At $0.5$ & '
        'Best possible \\\\',
        '\\midrule',
    ]
    tail = ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    text = '\n'.join(head + body + tail) + '\n'
    with io.open(os.path.join(MS, 'tab_threshold.tex'), 'w',
                 encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    print('  wrote tab_threshold.tex')

    for arch, cond in shared:
        i, e = inc[(arch, cond)], ext[(arch, cond)]
        print('  %-12s %-11s auc in %.3f  ext %.3f   bal at 0.5 %.2f  '
              'best %.2f at t=%.3f'
              % (arch, cond,
                 A.mean_sd([x['auc'] for x in i])[0],
                 A.mean_sd([x['auc'] for x in e])[0],
                 100 * A.mean_sd([x['default'] for x in e])[0],
                 100 * A.mean_sd([x['best'] for x in e])[0],
                 A.mean_sd([x['at'] for x in e])[0]))


if __name__ == '__main__':
    main()
