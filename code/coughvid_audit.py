"""Audit two properties of the classifier derived corpus from its metadata.

    python -I coughvid_audit.py

Neither question needs a model, and both bear on what the corpus can be used
to establish, so they are settled from the published metadata alone.

The first is whether the recording selection used by the work this study
extends can be reproduced. That work reports a count of cough and non-cough
recordings obtained by thresholding the shipped detector score. One of the two
counts is reachable and the other is not, and the reason is arithmetic rather
than a matter of judgement, so it is demonstrated here rather than asserted.

The second is whether the subset carrying expert annotations can be used to
estimate the label error rate of the corpus as a whole. That inference is only
available if the annotated subset is a random sample of the corpus. If the
recordings sent for annotation were chosen by the very score whose reliability
is in question, then the agreement measured on them says nothing about the
recordings that were not sent, which are the ones the threshold is least sure
about.
"""
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, 'data', 'coughvid_v3', 'coughvid_20211012',
                    'metadata_compiled.csv')
OUT = os.path.join(ROOT, 'results', 'coughvid_audit.txt')

# Reported by the work being extended.
WANT_COUGH = 15619
WANT_NON = 7261

# A recording counts as expert annotated when the first annotator recorded a
# judgement of audio quality, which is the field every annotation carries.
EXPERT_FIELD = 'quality_1'


def scores(rows):
    out = []
    for r in rows:
        v = (r.get('cough_detected') or '').strip()
        if v:
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def reachable(vals, target, say):
    """Report whether any threshold yields exactly target recordings."""
    uniq = sorted(set(vals), reverse=True)
    lines = []
    found = None
    for t in uniq:
        n = sum(1 for v in vals if v >= t)
        if n == target:
            found = t
            break
        if n > target:
            # Stepped past it; report the two counts it jumped between.
            prev = [u for u in uniq if u > t]
            before = sum(1 for v in vals if v >= prev[-1]) if prev else 0
            lines.append('    the count steps from %d directly to %d at a '
                         'score of %.4f' % (before, n, t))
            ties = sum(1 for v in vals if v == t)
            lines.append('    %d recordings carry the value %.4f exactly'
                         % (ties, t))
            break
    if found is not None:
        lines.insert(0, '    reachable: a threshold of %.4f yields exactly %d'
                     % (found, target))
    else:
        lines.insert(0, '    not reachable by any threshold on this column')
    return ['  %s (target %d)' % (say, target)] + lines


def quartiles(v):
    v = sorted(v)
    if not v:
        return (float('nan'),) * 3
    return (v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4])


def main():
    if not os.path.exists(META):
        sys.exit('no metadata at %s' % META)
    rows = list(csv.DictReader(io.open(META, encoding='utf-8')))
    out = []
    out.append('recordings in the released metadata: %d' % len(rows))

    vals = scores(rows)
    out.append('recordings carrying a detector score: %d' % len(vals))
    out.append('')
    out.append('Reproducing the selection used by the work being extended')
    out += reachable(vals, WANT_COUGH, 'cough recordings')
    low = [-v for v in vals]
    out += reachable(low, WANT_NON, 'non-cough recordings, as a lower tail')
    out.append('')

    ann = [r for r in rows if (r.get(EXPERT_FIELD) or '').strip()]
    rest = [r for r in rows if not (r.get(EXPERT_FIELD) or '').strip()]
    a_s, r_s = scores(ann), scores(rest)
    out.append('Is the expert annotated subset a random sample')
    out.append('  annotated: %d recordings, %.1f per cent of the corpus'
               % (len(ann), 100.0 * len(ann) / len(rows)))
    if a_s and r_s:
        aq, rq = quartiles(a_s), quartiles(r_s)
        out.append('  detector score, annotated   quartiles %.4f %.4f %.4f, '
                   'mean %.4f' % (aq[0], aq[1], aq[2], sum(a_s) / len(a_s)))
        out.append('  detector score, unannotated quartiles %.4f %.4f %.4f, '
                   'mean %.4f' % (rq[0], rq[1], rq[2], sum(r_s) / len(r_s)))
        lo = min(a_s)
        below = sum(1 for v in r_s if v < lo)
        out.append('  lowest score among the annotated recordings: %.4f'
                   % lo)
        out.append('  unannotated recordings scoring below that: %d, '
                   '%.1f per cent of the unannotated'
                   % (below, 100.0 * below / len(r_s)))
        for cut in (0.8, 0.9):
            out.append('  at or above %.1f: %.1f per cent of annotated, '
                       '%.1f per cent of unannotated'
                       % (cut, 100.0 * sum(1 for v in a_s if v >= cut) / len(a_s),
                          100.0 * sum(1 for v in r_s if v >= cut) / len(r_s)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = '\n'.join(out) + '\n'
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    sys.stdout.write(text)
    print()
    print('wrote %s' % os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
