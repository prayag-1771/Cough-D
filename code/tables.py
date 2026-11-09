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


def paired_table(root, label, caption):
    """McNemar over the denoising pairs, pooled across seeds, Holm adjusted."""
    data = {}
    if not os.path.isdir(root):
        return None
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
        return None
    # holm returns (label, raw p, adjusted p) ordered by raw p, so it is keyed
    # back onto the rows rather than zipped against them.
    adj = {lab: a for lab, _p, a in
           A.holm([(r[0] + r[1] + r[2], r[6]) for r in raw])}

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


def main():
    os.makedirs(MS, exist_ok=True)

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
