"""Work out where the cough and noncough labels came from.

    python -I check_labels.py <path to metadata_compiled.csv>

COUGHVID does not ship a human cough/noncough label for the whole corpus. It
ships cough_detected, a probability produced by a classifier the corpus
authors trained themselves, and expert annotations for a subset of roughly a
thousand recordings. The manuscript reports 15,619 cough and 7,261 noncough,
22,880 in total out of the corpus's 27,550.

If a threshold on cough_detected reproduces those two counts, the labels are
one classifier's output, and the study measures agreement with that classifier
rather than detection of coughing. That would be the single most consequential
thing about the design, so it is worth establishing before anything is rerun.

This script searches for the threshold pair that reproduces the published
counts, and reports how many recordings carry a genuine expert annotation.
Standard library only.
"""
import csv
import sys
from collections import Counter

TARGET_COUGH = 15619
TARGET_NON = 7261
TARGET_TOTAL = TARGET_COUGH + TARGET_NON


def main(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        rows = list(csv.DictReader(fh))

    print('recordings in metadata : %d' % len(rows))
    cols = list(rows[0].keys())
    print('columns (%d)           : %s' % (len(cols), ', '.join(cols)))
    print()

    key = next((c for c in cols if c.strip().lower() == 'cough_detected'), None)
    if key is None:
        print('no cough_detected column; cannot run the threshold test')
        return

    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (TypeError, ValueError):
            pass
    vals.sort()
    print('cough_detected present on %d of %d recordings' % (len(vals), len(rows)))
    print('  min %.4f  median %.4f  max %.4f'
          % (vals[0], vals[len(vals) // 2], vals[-1]))
    print()

    # ------------------------------------------------- how many are expert seen
    expert_cols = [c for c in cols
                   if any(w in c.lower() for w in ('expert', 'annotat', 'cough_type',
                                                   'quality', 'dyspnea', 'wheezing'))]
    if expert_cols:
        print('columns that look like expert annotation: %s' % ', '.join(expert_cols))
        seen = sum(1 for r in rows if any((r.get(c) or '').strip() for c in expert_cols))
        print('recordings with any expert annotation   : %d (%.1f%% of corpus)'
              % (seen, 100.0 * seen / len(rows)))
    else:
        print('no expert annotation columns found')
    print()

    # ----------------------------------------- does a threshold reproduce them?
    print('=' * 78)
    print('SEARCH: a threshold pair reproducing %d cough and %d noncough'
          % (TARGET_COUGH, TARGET_NON))
    print('=' * 78)

    grid = [i / 100.0 for i in range(1, 100)]
    exact, near = [], []
    for hi in grid:
        n_cough = sum(1 for v in vals if v >= hi)
        if abs(n_cough - TARGET_COUGH) > 400:
            continue
        for lo in grid:
            if lo > hi:
                continue
            n_non = sum(1 for v in vals if v < lo)
            err = abs(n_cough - TARGET_COUGH) + abs(n_non - TARGET_NON)
            if err == 0:
                exact.append((hi, lo, n_cough, n_non))
            elif err <= 60:
                near.append((err, hi, lo, n_cough, n_non))

    if exact:
        print('EXACT reproduction found:')
        for hi, lo, nc, nn in exact:
            print('  cough if cough_detected >= %.2f  -> %d' % (hi, nc))
            print('  noncough if cough_detected <  %.2f  -> %d' % (lo, nn))
            print('  discarded middle band: %d' % (len(vals) - nc - nn))
        print()
        print('  The labels are a threshold on another classifier\'s score.')
    elif near:
        near.sort()
        print('no exact pair; closest %d:' % min(6, len(near)))
        for err, hi, lo, nc, nn in near[:6]:
            print('  >=%.2f / <%.2f  -> cough %d (%+d), noncough %d (%+d)'
                  % (hi, lo, nc, nc - TARGET_COUGH, nn, nn - TARGET_NON))
    else:
        print('no threshold pair comes close to the published counts.')
        print('The labels were probably not derived from cough_detected alone.')

    # a single threshold splitting the whole corpus into exactly the two counts
    print()
    print('single threshold using every recording (no discarded band):')
    best = None
    for t in grid:
        nc = sum(1 for v in vals if v >= t)
        nn = len(vals) - nc
        e = abs(nc - TARGET_COUGH) + abs(nn - TARGET_NON)
        if best is None or e < best[0]:
            best = (e, t, nc, nn)
    e, t, nc, nn = best
    print('  best at %.2f -> cough %d (%+d), noncough %d (%+d)%s'
          % (t, nc, nc - TARGET_COUGH, nn, nn - TARGET_NON,
             '   EXACT' if e == 0 else ''))

    if 'status' in [c.strip().lower() for c in cols]:
        sk = next(c for c in cols if c.strip().lower() == 'status')
        print()
        print('status column (self reported, not a cough label): %s'
              % dict(Counter((r.get(sk) or 'blank').strip() for r in rows)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python -I check_labels.py <metadata_compiled.csv>')
    main(sys.argv[1])
