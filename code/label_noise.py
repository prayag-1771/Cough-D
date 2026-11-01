"""How far do COUGHVID's cough_detected scores agree with expert judgement?

    python -I label_noise.py <metadata_compiled.csv>

COUGHVID carries no human cough/noncough label for the corpus as a whole. It
carries cough_detected, a score produced by a classifier the corpus authors
trained, and expert annotations for a small subset. Among the expert fields,
quality_N takes the value no_cough when that expert judged no cough to be
present, which is the only human verdict on cough presence the corpus
contains.

Any study that labels the whole corpus cough against noncough must therefore
threshold cough_detected. This script measures what that costs, by treating
the expert verdict as the reference on the subset where one exists and
reporting, for each candidate threshold, how often the derived label disagrees
with it.

Two limits on what the numbers below can support, both measured rather than
assumed, and both printed with the results.

The expert no_cough group is small, so every rate carries a wide interval.
More importantly, the recordings sent for expert annotation were not drawn at
random: they score far higher on cough_detected than the rest of the corpus,
so the annotated subset over-represents exactly the recordings the score is
confident about. The disagreement rates below are therefore conditional on
that selection and are NOT an estimate of the corpus-wide label error rate.
They establish that expert-confirmed non-coughs enter the positive class, not
how many do.
"""
import csv
import math
import sys

EXPERTS = (1, 2, 3)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def verdict(row):
    """Majority expert verdict on cough presence, or None if unannotated.

    Returns False when the annotating experts say no cough, True when they say
    a cough is present. Ties are discarded rather than broken arbitrarily.
    """
    says = [(row.get('quality_%d' % i) or '').strip() for i in EXPERTS]
    says = [s for s in says if s]
    if not says:
        return None
    no = sum(s == 'no_cough' for s in says)
    yes = len(says) - no
    if no == yes:
        return None
    return yes > no


def main(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8', errors='replace')))

    labelled = []
    for r in rows:
        v = verdict(r)
        if v is None:
            continue
        try:
            labelled.append((float(r['cough_detected']), v))
        except (TypeError, ValueError, KeyError):
            pass

    n_yes = sum(1 for _, v in labelled if v)
    n_no = len(labelled) - n_yes
    print('corpus recordings                     : %d' % len(rows))
    print('with an unambiguous expert verdict    : %d' % len(labelled))
    print('  expert says a cough is present      : %d' % n_yes)
    print('  expert says no cough is present     : %d' % n_no)
    print()

    if n_no == 0:
        print('no expert no_cough examples: the test cannot be run')
        return

    print('=' * 92)
    print('DISAGREEMENT BETWEEN A THRESHOLD ON cough_detected AND THE EXPERT VERDICT')
    print('=' * 92)
    print('%-7s %-28s %-28s %s'
          % ('thresh', 'expert no_cough called cough', 'expert cough called noncough',
             'overall'))
    print('-' * 92)

    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        # false positives against the expert: no cough, but score says cough
        fp = sum(1 for s, v in labelled if not v and s >= t)
        fn = sum(1 for s, v in labelled if v and s < t)
        fpr, (fl, fh) = fp / n_no, wilson(fp, n_no)
        fnr, (nl, nh) = fn / n_yes, wilson(fn, n_yes)
        err = (fp + fn) / len(labelled)
        print('%-7.2f %3d/%-3d %5.1f%% [%4.1f,%5.1f]  %4d/%-5d %5.1f%% [%4.1f,%4.1f]  %5.1f%%'
              % (t, fp, n_no, 100 * fpr, 100 * fl, 100 * fh,
                 fn, n_yes, 100 * fnr, 100 * nl, 100 * nh, 100 * err))

    print()
    print('The left column is the quantity that matters. A recording an expert')
    print('judged to contain no cough, but which a threshold labels cough, enters')
    print('the positive class as a mislabelled example, and at test time it is')
    print('scored as a cough the detector is supposed to find.')
    print()

    print('The overall column is NOT a corpus error rate. It is dominated by the')
    print('%d expert-cough recordings against %d expert-no_cough ones, so it falls'
          % (n_yes, n_no))
    print('as the threshold drops simply because almost everything then counts as')
    print('cough. Read the two left columns, not the right one.')
    print()

    # ------------------------------------------- how selected was that subset?
    print('=' * 92)
    print('SELECTION OF THE ANNOTATED SUBSET')
    print('=' * 92)
    scored = []
    for r in rows:
        try:
            scored.append((float(r['cough_detected']), verdict(r) is not None))
        except (TypeError, ValueError, KeyError):
            pass
    ann = sorted(s for s, a in scored if a)
    non = sorted(s for s, a in scored if not a)

    def pct_at(v, t):
        return 100.0 * sum(1 for x in v if x >= t) / len(v) if v else 0.0

    print('%-22s %7s %9s %12s' % ('', 'n', 'median', 'share >= 0.8'))
    for nm, v in (('sent to an expert', ann), ('not sent', non)):
        if v:
            print('%-22s %7d %9.3f %11.1f%%'
                  % (nm, len(v), v[len(v) // 2], pct_at(v, 0.8)))
    print()
    print('The two distributions differ, so the annotated recordings are a')
    print('high scoring, non-random slice of the corpus. Every rate above is')
    print('conditional on that slice and cannot be extrapolated to the whole.')
    print()
    print('What the corpus does establish, without any of these caveats:')
    print('  * it carries no human cough/noncough verdict for %.0f%% of recordings;'
          % (100.0 * (len(rows) - len(labelled)) / len(rows)))
    print('  * the only corpus-wide label available is cough_detected, which is')
    print('    the output of a classifier trained by the corpus authors;')
    print('  * so a model trained and tested on thresholded labels is measured')
    print('    on its agreement with that classifier, not on cough detection.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python -I label_noise.py <metadata_compiled.csv>')
    main(sys.argv[1])
