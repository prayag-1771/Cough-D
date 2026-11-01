"""Turn a directory of per recording decisions into the tables a reviewer wants.

    python analyse.py runs

Reads every CSV written by train.py, whose name encodes the condition as
arch__imageset__seedN, and reports

  * each condition's metrics as a mean and standard deviation across seeds,
    which is what makes a difference interpretable at all;
  * the pooled accuracy with its Wilson score interval;
  * McNemar's exact test on the paired decisions for each denoising
    comparison, per seed and pooled, with a Holm correction across the family
    of comparisons;
  * the spread across seeds, so the size of a claimed effect can be read
    against the variation between runs of the same condition.

Standard library only, so there is nothing to install. The positive class is
cough.
"""
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

POS = 'cough'


# ----------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """Wilson score interval: keeps nominal coverage near 1, unlike Wald."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def chi2_sf_1df(x):
    """Survival function of chi-square with one degree of freedom."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(b, c):
    """Two sided McNemar on the discordant counts.

    Exact binomial while the discordant total is small enough to enumerate,
    otherwise the continuity corrected chi-square. Returns (statistic, p).
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    if n <= 1000:
        lo = min(b, c)
        tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
        return (float(min(b, c)), min(1.0, 2.0 * tail))
    stat = (abs(b - c) - 1) ** 2 / n
    return (stat, chi2_sf_1df(stat))


def holm(pairs):
    """Holm-Bonferroni step down. pairs is [(label, p)]; returns adjusted."""
    order = sorted(pairs, key=lambda kv: kv[1])
    m = len(order)
    out, running = [], 0.0
    for i, (label, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # enforce monotonicity
        out.append((label, p, running))
    return out


def metrics(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal = (rec + spec) / 2
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / math.sqrt(d)) if d else 0.0
    return dict(n=n, acc=acc, prec=prec, rec=rec, spec=spec, f1=f1,
                bal=bal, mcc=mcc, tp=tp, fn=fn, fp=fp, tn=tn)


def mean_sd(xs):
    k = len(xs)
    if k == 0:
        return (float('nan'), float('nan'))
    m = sum(xs) / k
    if k == 1:
        return (m, 0.0)
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (k - 1)))


# ---------------------------------------------------------------------- input
TAG = re.compile(r'^(?P<arch>[^_]+(?:_[^_]+)*?)__(?P<img>.+)__seed(?P<seed>\d+)$')


def load(path):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError('%s is empty' % path)
    need = {'file', 'true', 'pred'}
    missing = need - set(rows[0])
    if missing:
        raise ValueError('%s lacks columns: %s' % (path, sorted(missing)))
    return {r['file']: (r['true'].strip(), r['pred'].strip()) for r in rows}


def confusion(d):
    tp = fn = fp = tn = 0
    for truth, pred in d.values():
        if truth == POS:
            tp += (pred == POS)
            fn += (pred != POS)
        else:
            fp += (pred == POS)
            tn += (pred != POS)
    return tp, fn, fp, tn


def short(img):
    s = img.replace('dataset_', '').replace('_images', '')
    return {'linear': 'linear', 'wav_linear': 'wav+linear',
            'mel': 'mel', 'wav_mel': 'wav+mel'}.get(s, s)


def main(root):
    paths = sorted(glob.glob(os.path.join(root, '*.csv')))
    if not paths:
        sys.exit('no CSV files in %s -- run run_grid.sh first' % root)

    runs = {}                                  # (arch, img, seed) -> decisions
    for p in paths:
        m = TAG.match(os.path.splitext(os.path.basename(p))[0])
        if not m:
            print('  skipping unrecognised name: %s' % os.path.basename(p))
            continue
        runs[(m['arch'], m['img'], int(m['seed']))] = load(p)

    if not runs:
        sys.exit('no file names matched arch__imageset__seedN')

    conds = defaultdict(list)
    for (arch, img, seed) in runs:
        conds[(arch, img)].append(seed)
    for k in conds:
        conds[k].sort()

    bar = '=' * 100

    # ------------------------------------------------- per condition summary
    print(bar)
    print('CONDITION SUMMARY, mean and standard deviation across seeds')
    print(bar)
    print('%-13s %-15s %-4s %-16s %-14s %-14s %-14s'
          % ('Architecture', 'Input', 'n', 'Accuracy', 'Specificity',
             'Balanced acc', 'MCC'))
    print('-' * 100)

    summary = {}
    for (arch, img) in sorted(conds):
        seeds = conds[(arch, img)]
        ms = [metrics(*confusion(runs[(arch, img, s)])) for s in seeds]
        summary[(arch, img)] = ms
        cells = []
        for key in ('acc', 'spec', 'bal', 'mcc'):
            mu, sd = mean_sd([m[key] for m in ms])
            scale = 1.0 if key == 'mcc' else 100.0
            fmt = '%6.3f +- %.3f' if key == 'mcc' else '%6.2f +- %.2f'
            cells.append(fmt % (mu * scale, sd * scale))
        print('%-13s %-15s %-4d %-16s %-14s %-14s %-14s'
              % (arch, short(img), len(seeds), cells[0], cells[1],
                 cells[2], cells[3]))

    # ---------------------------------------------------- pooled with Wilson
    print()
    print(bar)
    print('POOLED OVER SEEDS, with the 95 percent Wilson score interval')
    print(bar)
    print('%-13s %-15s %-9s %-18s %-8s' % ('Architecture', 'Input',
                                           'Accuracy', '95% interval', 'Seeds'))
    print('-' * 100)
    for (arch, img) in sorted(conds):
        ms = summary[(arch, img)]
        tp = sum(m['tp'] for m in ms); fn = sum(m['fn'] for m in ms)
        fp = sum(m['fp'] for m in ms); tn = sum(m['tn'] for m in ms)
        lo, hi = wilson(tp + tn, tp + fn + fp + tn)
        print('%-13s %-15s %7.2f   [%5.2f, %5.2f]      %d'
              % (arch, short(img), 100 * (tp + tn) / (tp + fn + fp + tn),
                 100 * lo, 100 * hi, len(ms)))

    # ------------------------------------------- paired denoising comparisons
    PAIRS = [('linear', 'wav_linear'), ('mel', 'wav_mel')]

    print()
    print(bar)
    print('PAIRED DENOISING COMPARISONS, McNemar on the per recording decisions')
    print(bar)

    family = []
    for arch in sorted({a for a, _ in conds}):
        for plain, dn in PAIRS:
            if (arch, plain) not in conds or (arch, dn) not in conds:
                continue
            shared = sorted(set(conds[(arch, plain)]) & set(conds[(arch, dn)]))
            if not shared:
                continue

            label = '%s, %s -> %s' % (arch, short(plain), short(dn))
            print()
            print(label)
            print('  %-6s %9s %9s %6s %6s %12s' %
                  ('seed', 'acc base', 'acc dn', 'b', 'c', 'p'))

            deltas, Bs, Cs = [], 0, 0
            for s in shared:
                a = runs[(arch, plain, s)]
                b_ = runs[(arch, dn, s)]
                common = set(a) & set(b_)
                if len(common) != len(a) or len(common) != len(b_):
                    print('  seed %d: %d shared of %d/%d recordings'
                          % (s, len(common), len(a), len(b_)))
                bb = cc = 0
                for f in common:
                    ok_a = a[f][0] == a[f][1]
                    ok_b = b_[f][0] == b_[f][1]
                    bb += (ok_a and not ok_b)      # denoising broke it
                    cc += (ok_b and not ok_a)      # denoising fixed it
                Bs += bb
                Cs += cc
                ma = metrics(*confusion(a))
                mb = metrics(*confusion(b_))
                deltas.append(mb['acc'] - ma['acc'])
                _, p = mcnemar(bb, cc)
                print('  %-6d %9.2f %9.2f %6d %6d %12.4g'
                      % (s, 100 * ma['acc'], 100 * mb['acc'], bb, cc, p))

            mu, sd = mean_sd([100 * d for d in deltas])
            stat, p_pool = mcnemar(Bs, Cs)
            print('  pooled discordance b=%d c=%d   McNemar p = %.4g' % (Bs, Cs, p_pool))
            print('  accuracy change across seeds: %+.2f +- %.2f points' % (mu, sd))
            if sd > 0 and abs(mu) < sd:
                print('  the change is smaller than the variation between seeds')
            family.append((label, p_pool))

    if family:
        print()
        print(bar)
        print('HOLM CORRECTION ACROSS THE %d DENOISING COMPARISONS' % len(family))
        print(bar)
        print('%-52s %12s %12s %s' % ('Comparison', 'p', 'p adjusted', 'at 0.05'))
        print('-' * 100)
        for label, p, adj in holm(family):
            print('%-52s %12.4g %12.4g %s'
                  % (label, p, adj, 'significant' if adj < 0.05 else 'not significant'))

    # ---------------------------------------------- seed spread vs effect size
    print()
    print(bar)
    print('VARIATION BETWEEN SEEDS WITHIN A CONDITION')
    print(bar)
    spreads = []
    for (arch, img) in sorted(conds):
        ms = summary[(arch, img)]
        if len(ms) < 2:
            continue
        accs = [100 * m['acc'] for m in ms]
        mu, sd = mean_sd(accs)
        spreads.append(sd)
        print('  %-13s %-15s range %.2f points (sd %.2f) over %d seeds'
              % (arch, short(img), max(accs) - min(accs), sd, len(accs)))
    if spreads:
        print()
        print('  mean within condition standard deviation: %.2f accuracy points'
              % (sum(spreads) / len(spreads)))
        print('  No effect smaller than roughly twice this value can be')
        print('  claimed from a single run per condition.')
    else:
        print('  only one seed per condition: the spread cannot be estimated,')
        print('  and no small difference can be interpreted. Add seeds.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs')
