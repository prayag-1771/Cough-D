"""Report the counts that Section 6 states about the shrinkage comparisons.

    python -I paired_counts.py

Section 6 says how many paired comparisons were made and how many of them
show an effect. Both are counts over results/paired.csv, and both have gone
stale twice already, because a comparison enters the file as soon as the
runs behind it finish and the sentence was written when fewer had. Neither
trace_numbers.py nor check_paper.py can see it: the first checks that a
number in the prose occurs in a generated file, which "six" does, and the
second does not read results at all.

This prints the counts so the sentence can be checked against the data in
one command rather than by counting rows by eye.
"""
import csv
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05
WORDS = ('no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
         'eight', 'nine', 'ten', 'eleven', 'twelve')


def word(n):
    return WORDS[n] if n < len(WORDS) else str(n)


def main():
    path = os.path.join(ROOT, 'results', 'paired.csv')
    if not os.path.exists(path):
        print('no paired.csv yet')
        return 1
    rows = list(csv.DictReader(io.open(path, encoding='utf-8')))
    corpora = sorted({r['corpus'] for r in rows})
    sig = [r for r in rows if float(r['p_adjusted']) < ALPHA]
    losses = [r for r in sig if float(r['accuracy_change']) < 0]
    gains = [r for r in sig if float(r['accuracy_change']) > 0]

    print('paired comparisons: %d (%s), over %d corpora: %s'
          % (len(rows), word(len(rows)), len(corpora), ', '.join(corpora)))
    for c in corpora:
        n = sum(1 for r in rows if r['corpus'] == c)
        print('    %-10s %d' % (c, n))
    print()
    print('an effect at adjusted p < %.2f: %d (%s)'
          % (ALPHA, len(sig), word(len(sig))))
    print('    of those, losses %d, gains %d' % (len(losses), len(gains)))
    for r in sig:
        print('    %-9s %-12s %s -> %s   adj p %-10s  %s'
              % (r['corpus'], r['arch'], r['base'], r['denoised'],
                 r['p_adjusted'], r['accuracy_change']))
    print()
    improves = [r for r in rows if float(r['accuracy_change']) > 0]
    print('comparisons with any improvement in mean accuracy, significant '
          'or not: %d' % len(improves))
    for r in improves:
        print('    %-9s %-12s %s -> %s   %s'
              % (r['corpus'], r['arch'], r['base'], r['denoised'],
                 r['accuracy_change']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
