"""Which sounds the detectors mistake for coughs, by annotated class.

    python -I error_analysis.py

The external evaluation reports that sensitivity collapses while specificity
is largely retained. That is an aggregate over a negative class assembled
deliberately from confusable material, and the aggregate hides the part a
practitioner needs: whether the false positives are spread evenly over that
material or concentrated in a few kinds of sound.

The human annotated corpus labels every clip against an ontology, so the
question is answerable directly. For each annotated class present among the
negatives, this counts how often a clip carrying that label is called a cough,
pooled over the seeds of one architecture. The cough clips are reported the
same way, as recall.

A clip may carry several labels, so the rows are not a partition of the
negatives and their counts do not sum to the total. A clip labelled both
Sneeze and Breathing appears under each.
"""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, 'manuscript')
GT = os.path.join(ROOT, 'data', 'fsd50k_meta', 'FSD50K.ground_truth',
                  'eval.csv')
RUNS = os.path.join(ROOT, 'runs_fsd50k')

ARCH = 'resnet18'

# As they read in the ontology, mapped to something a sentence can contain.
SHOWN = {
    'Breathing': 'Breathing',
    'Sneeze': 'Sneeze',
    'Gasp': 'Gasp',
    'Sigh': 'Sigh',
    'Laughter': 'Laughter',
    'Burping_and_eructation': 'Burping',
    'Chewing_and_mastication': 'Chewing',
    'Speech': 'Speech',
    'Male_speech_and_man_speaking': 'Male speech',
    'Female_speech_and_woman_speaking': 'Female speech',
    'Child_speech_and_kid_speaking': 'Child speech',
    'Whispering': 'Whispering',
    'Yell': 'Yell',
}


def labels_by_file():
    out = {}
    with io.open(GT, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            out[r['fname']] = set(r['labels'].split(','))
    return out


def main():
    if not os.path.exists(GT):
        sys.exit('no ground truth at %s' % GT)
    gt = labels_by_file()

    runs = [f for f in sorted(os.listdir(RUNS))
            if f.endswith('.csv') and f.startswith(ARCH + '__')] \
        if os.path.isdir(RUNS) else []
    if not runs:
        sys.exit('no %s runs under %s' % (ARCH, RUNS))

    # class -> [called cough, seen]; and the positives as recall.
    tally = {k: [0, 0] for k in SHOWN}
    pos = [0, 0]
    conds = set()
    for fn in runs:
        m = A.TAG.match(fn[:-4])
        if not m:
            continue
        conds.add(m.group('img'))
        for name, (truth, pred) in A.load(os.path.join(RUNS, fn)).items():
            base = name[:-4] if name.endswith('.png') else name
            said_cough = (pred == A.POS)
            if truth == A.POS:
                pos[1] += 1
                pos[0] += said_cough
                continue
            for lab in gt.get(base, ()):
                if lab in tally:
                    tally[lab][1] += 1
                    tally[lab][0] += said_cough

    rows = [(SHOWN[k], v[0], v[1], 100.0 * v[0] / v[1])
            for k, v in tally.items() if v[1]]
    rows.sort(key=lambda r: -r[3])

    n_seeds = len(runs)
    print('pooled over %d runs across %d input conditions'
          % (n_seeds, len(conds)))
    print('  %-14s %8s %8s %9s' % ('class', 'cough', 'clips', 'per cent'))
    for name, k, n, pct in rows:
        print('  %-14s %8d %8d %8.1f' % (name, k, n, pct))
    print('  %-14s %8d %8d %8.1f' % ('cough (recall)', pos[0], pos[1],
                                     100.0 * pos[0] / pos[1]))

    body = []
    for name, k, n, pct in rows:
        body.append('%s & %d & %d & $%.1f$ \\\\' % (name, n, k, pct))
    head = [
        '\\begin{table}[htbp]',
        '\\caption{What the detectors call a cough, by annotated class, '
        'pooled over every %s run on the external set. The third column '
        'counts the decisions in which a clip carrying that label was '
        'assigned to the cough class, and the fourth expresses it as a rate. '
        'A clip may carry several labels, so the rows overlap and their '
        'counts do not sum to the size of the negative class. The recall on '
        'clips actually labelled cough is given for comparison in the final '
        'row: several kinds of non-cough sound are assigned to the cough '
        'class more readily than coughs themselves '
        'are.}\\label{tab:confusion}' % ARCH.replace('resnet18', 'ResNet18'),
        '\\small',
        '\\begin{tabular}{@{}lrrc@{}}',
        '\\toprule',
        'Annotated class & Decisions & Called cough & Per cent \\\\',
        '\\midrule',
    ]
    body.append('\\addlinespace')
    body.append('Cough, as recall & %d & %d & $%.1f$ \\\\'
                % (pos[1], pos[0], 100.0 * pos[0] / pos[1]))
    tail = ['\\botrule', '\\end{tabular}', '\\end{table}', '']
    with io.open(os.path.join(MS, 'tab_confusion.tex'), 'w',
                 encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(head + body + tail) + '\n')
    print()
    print('wrote tab_confusion.tex')


if __name__ == '__main__':
    main()
