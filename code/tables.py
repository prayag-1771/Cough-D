"""Emit the result tables as LaTeX, computed from the saved decision files.

    python -I tables.py

The setup section promises that every count in the results is derived from the
saved decision files by script rather than by arithmetic on the page. This is
that script. Each table is written as a complete float into the manuscript
directory and pulled into the document by assemble.py, so a table cannot drift
out of step with the runs it describes: regenerating is the only way to change
one.

The statistics come from analyse.py, which is the same code that produces the
figures the text quotes, so a table and a printed summary cannot disagree
either.
"""
import csv
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

# The denoising comparisons, each holding the frequency axis fixed so that the
# shrinkage is the only thing that differs between the two conditions.
PAIRS = [('linear', 'wav_linear'), ('mel', 'wav_mel')]


def collect(root):
    """{(arch, condition): [per seed confusion]} from one run directory."""
    out = {}
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.csv'):
            continue
        m = A.TAG.match(fn[:-4])
        if not m:
            continue
        d = A.load(os.path.join(root, fn))
        out.setdefault((m.group('arch'), m.group('img')), []).append(
            A.metrics(*A.confusion(d)))
    return out


def pm(xs, places=2):
    """Mean and standard deviation across seeds, as a per cent."""
    m, s = A.mean_sd([100.0 * x for x in xs])
    return '$%.*f \\pm %.*f$' % (places, m, places, s)


def pm_raw(xs, places=3):
    m, s = A.mean_sd(list(xs))
    return '$%.*f \\pm %.*f$' % (places, m, places, s)


def rows_for(data):
    """Table body lines, in a fixed architecture and condition order."""
    lines = []
    for arch in ORDER:
        present = [c for c in CONDS if (arch, c) in data]
        if not present:
            continue
        if lines:
            lines.append('\\addlinespace')
        for j, cond in enumerate(present):
            v = data[(arch, cond)]
            lines.append(
                '%s & %s & %d & %s & %s & %s & %s \\\\'
                % (ARCH.get(arch, arch) if j == 0 else '',
                   A.short(cond), len(v),
                   pm([x['acc'] for x in v]),
                   pm([x['rec'] for x in v]),
                   pm([x['spec'] for x in v]),
                   pm_raw([x['mcc'] for x in v])))
    return lines


def table(data, label, caption):
    if not data:
        return None
    body = rows_for(data)
    if not body:
        return None
    head = [
        '\\begin{table}[htbp]',
        '\\caption{%s}\\label{%s}' % (caption, label),
        '\\small',
        '\\begin{tabular}{@{}llrcccc@{}}',
        '\\toprule',
        'Architecture & Input & Seeds & Accuracy & Sensitivity & Specificity '
        '& MCC \\\\',
        '\\midrule',
    ]
    tail = ['\\botrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(head + body + tail) + '\n'


def paired_rows(root):
    """McNemar over the denoising pairs, pooled across seeds, Holm adjusted.

    Returned separately from the table it formats, because two corpora are
    tested this way and only one of them is tabulated. The other is quoted in
    the prose, and a quoted number has to come from a file like any other, so
    both sets of rows are written to a CSV as well.
    """
    data = {}
    if not os.path.isdir(root):
        return None, None
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.csv'):
            continue
        m = A.TAG.match(fn[:-4])
        if not m:
            continue
        data[(m.group('arch'), m.group('img'), m.group('seed'))] = \
            A.load(os.path.join(root, fn))

    raw = []
    for arch in ORDER:
        for base, dn in PAIRS:
            b_tot = c_tot = 0
            deltas = []
            seeds = sorted({s for (a, i, s) in data
                            if a == arch and i in (base, dn)})
            used = 0
            for s in seeds:
                p, q = data.get((arch, base, s)), data.get((arch, dn, s))
                if p is None or q is None:
                    continue
                shared = set(p) & set(q)
                if not shared:
                    continue
                used += 1
                b = sum(1 for k in shared
                        if p[k][1] == p[k][0] and q[k][1] != q[k][0])
                c = sum(1 for k in shared
                        if p[k][1] != p[k][0] and q[k][1] == q[k][0])
                b_tot += b
                c_tot += c
                ap = sum(1 for k in shared if p[k][1] == p[k][0]) / len(shared)
                aq = sum(1 for k in shared if q[k][1] == q[k][0]) / len(shared)
                deltas.append(100.0 * (aq - ap))
            if used < 2:
                continue
            raw.append((arch, base, dn, used, b_tot, c_tot,
                        A.mcnemar(b_tot, c_tot)[1], deltas))

    if not raw:
        return None, None
    # holm returns (label, raw p, adjusted p) ordered by raw p, so it is keyed
    # back onto the rows rather than zipped against them.
    adj = {lab: a for lab, _p, a in
           A.holm([(r[0] + r[1] + r[2], r[6]) for r in raw])}
    return raw, adj


def paired_table(root, label, caption):
    """Format one corpus' paired comparisons as a table."""
    raw, adj = paired_rows(root)
    if not raw:
        return None

    body = []
    for r in raw:
        arch, base, dn, used, b, c, p, deltas = r
        padj = adj[arch + base + dn]
        d_m, d_s = A.mean_sd(deltas)
        body.append(
            '%s & %s $\\rightarrow$ %s & %d & %d & %d & %s & %s & $%+.2f \\pm '
            '%.2f$ \\\\'
            % (ARCH.get(arch, arch), A.short(base), A.short(dn), used, b, c,
               fmt_p(p), fmt_p(padj), d_m, d_s))

    head = [
        '\\begin{table}[htbp]',
        '\\caption{%s}\\label{%s}' % (caption, label),
        '\\small',
        '\\begin{tabular}{@{}llrrrllc@{}}',
        '\\toprule',
        'Architecture & Comparison & Seeds & $b$ & $c$ & $p$ & $p$ adj.\\ & '
        'Accuracy change \\\\',
        '\\midrule',
    ]
    tail = ['\\botrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(head + body + tail) + '\n'


def fmt_p(p):
    if p != p:
        return '---'
    if p < 1e-4:
        return '$<10^{-4}$'
    return '$%.4f$' % p if p < 0.1 else '$%.3f$' % p


def write(name, text):
    if text is None:
        print('  %-18s skipped, no runs yet' % name)
        return False
    path = os.path.join(MS, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    print('  wrote %s' % name)
    return True


def write_summary():
    """Mean and spread per corpus, architecture and condition.

    The printed tables carry accuracy, sensitivity, specificity and the
    correlation coefficient, which is as many columns as the page width takes.
    Balanced accuracy is the quantity the text reasons with throughout, so it
    is written here with the rest, and every averaged figure the prose quotes
    has a file it can be checked against.
    """
    out = os.path.join(ROOT, 'results', 'summary.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for corpus in ('coswara', 'fsd50k', 'coughvid'):
        data = collect(os.path.join(ROOT, 'runs_' + corpus))
        for arch in ORDER:
            for cond in CONDS:
                v = data.get((arch, cond))
                if not v:
                    continue
                r = [corpus, arch, cond, len(v)]
                for key in ('acc', 'rec', 'spec', 'bal'):
                    m, s = A.mean_sd([100.0 * x[key] for x in v])
                    r += ['%.4f' % m, '%.4f' % s]
                m, s = A.mean_sd([x['mcc'] for x in v])
                r += ['%.4f' % m, '%.4f' % s]
                rows.append(r)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['corpus', 'arch', 'condition', 'seeds',
                    'accuracy', 'accuracy_sd', 'sensitivity', 'sensitivity_sd',
                    'specificity', 'specificity_sd', 'balanced', 'balanced_sd',
                    'mcc', 'mcc_sd'])
        w.writerows(rows)
    print('  wrote results/summary.csv (%d conditions)' % len(rows))


def write_per_seed():
    """Every run's metrics, one row each, so a single seed can be traced.

    The tables average over seeds, and the text occasionally needs to name an
    individual run, usually because it behaved unlike the others. Those
    figures would otherwise exist only in a console log.
    """
    out = os.path.join(ROOT, 'results', 'per_seed.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for corpus in ('runs_coswara', 'runs_fsd50k', 'runs_coughvid'):
        d = os.path.join(ROOT, corpus)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.csv'):
                continue
            m = A.TAG.match(fn[:-4])
            if not m:
                continue
            k = A.metrics(*A.confusion(A.load(os.path.join(d, fn))))
            rows.append([corpus.replace('runs_', ''), m.group('arch'),
                         m.group('img'), m.group('seed'), k['n'],
                         '%.4f' % (100 * k['acc']), '%.4f' % (100 * k['rec']),
                         '%.4f' % (100 * k['spec']),
                         '%.4f' % (100 * k['bal']), '%.4f' % k['mcc']])
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['corpus', 'arch', 'condition', 'seed', 'n', 'accuracy',
                    'sensitivity', 'specificity', 'balanced', 'mcc'])
        w.writerows(rows)
    print('  wrote results/per_seed.csv (%d runs)' % len(rows))


def write_corpora():
    """Counts and priors per corpus, read from the partition manifests."""
    out = os.path.join(ROOT, 'results', 'corpora.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for name, rel in (('coswara', 'coswara_img'),
                      ('coughvid', 'coughvid_img'),
                      ('fsd50k', 'fsd50k_img')):
        path = os.path.join(ROOT, 'data', rel, 'splits.csv')
        if not os.path.exists(path):
            continue
        with open(path, newline='', encoding='utf-8') as fh:
            recs = list(csv.DictReader(fh))
        n = len(recs)
        cough = sum(1 for r in recs if r['label'] == 'cough')
        people = len({r['participant'] for r in recs})
        by = {}
        for r in recs:
            s = by.setdefault(r['partition'], [0, 0])
            s[0] += 1
            s[1] += (r['label'] == 'cough')
        rows.append([name, n, cough, '%.4f' % (cough / float(n)), people]
                    + [by.get(p, [0, 0])[i] for p in ('train', 'val', 'test')
                       for i in (0, 1)])
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['corpus', 'recordings', 'cough', 'prior', 'participants',
                    'train', 'train_cough', 'val', 'val_cough', 'test',
                    'test_cough'])
        w.writerows(rows)
    print('  wrote results/corpora.csv (%d corpora)' % len(rows))


def write_paired():
    """The paired denoising tests for every corpus that has both conditions.

    The Holm correction is applied within a corpus and not across corpora,
    because the family of hypotheses being tested is the set of comparisons
    made on that corpus. Pooling the two would adjust each corpus' claims for
    tests performed on different data, which is not what the correction is for.
    """
    out = os.path.join(ROOT, 'results', 'paired.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for name, rel in (('coswara', 'runs_coswara'),
                      ('coughvid', 'runs_coughvid')):
        raw, adj = paired_rows(os.path.join(ROOT, rel))
        if not raw:
            continue
        for arch, base, dn, used, b, c, p, deltas in raw:
            d_m, d_s = A.mean_sd(deltas)
            rows.append([name, arch, base, dn, used, b, c,
                         '%.6g' % p, '%.6g' % adj[arch + base + dn],
                         '%+.2f' % d_m, '%.2f' % d_s])
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['corpus', 'arch', 'base', 'denoised', 'seeds', 'b', 'c',
                    'p', 'p_adjusted', 'accuracy_change', 'change_sd'])
        w.writerows(rows)
    print('  wrote results/paired.csv (%d comparisons)' % len(rows))


def main():
    os.makedirs(MS, exist_ok=True)
    write_per_seed()
    write_summary()
    write_corpora()
    write_paired()

    write('tab_grid.tex', table(
        collect(os.path.join(ROOT, 'runs_coswara')), 'tab:grid',
        'Detection on the prompt derived corpus, over held out speakers. Each '
        'row is one architecture under one input condition, averaged over the '
        'seeds in the third column, with the standard deviation across those '
        'seeds. Sensitivity is the proportion of coughs recovered and '
        'specificity the proportion of non-coughs rejected; both are reported '
        'because the class prior makes accuracy alone uninformative.'))

    write('tab_external.tex', table(
        collect(os.path.join(ROOT, 'runs_fsd50k')), 'tab:external',
        'The same saved weights applied to human annotated audio at the same '
        'class prior, with no retraining and no adaptation of any kind. The '
        'rows correspond one to one with those of Table~\\ref{tab:grid}. The '
        'fall is asymmetric: specificity is substantially retained while '
        'sensitivity is not, so the detectors continue to reject non-coughs '
        'and stop recovering coughs.'))

    write('tab_regime.tex', table(
        collect(os.path.join(ROOT, 'runs_coughvid')), 'tab:regime',
        'Detection on the classifier derived corpus at a matched class prior. '
        'What is measured here is agreement with the detector that produced '
        'the labels, not whether a cough is present, and the figures should '
        'be read as a property of that regime.'))

    write('tab_paired.tex', paired_table(
        os.path.join(ROOT, 'runs_coswara'), 'tab:paired',
        'Wavelet shrinkage tested on paired decisions over the prompt derived '
        'corpus. For each pair of conditions, $b$ counts the recordings the '
        'unprocessed condition classifies correctly and the shrunk condition '
        'does not, and $c$ counts the reverse; these are the only recordings '
        'McNemar\'s test uses. Adjusted $p$ values follow the step down '
        'procedure of Holm over the family of comparisons in this table. The '
        'final column gives the change in accuracy across seeds, which is the '
        'quantity that determines whether an effect of this size could be '
        'claimed from a single run.'))


if __name__ == '__main__':
    main()
