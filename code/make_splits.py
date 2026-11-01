"""Build one speaker-disjoint train/val/test split and apply it to every condition.

    python -I make_splits.py --img ../data/coswara_img --meta ../data/coswara_meta/combined_data.csv

Two properties matter and neither is optional.

The split is by participant, not by recording. Every participant contributes
two coughs and up to seven non-cough samples, so a split over recordings would
place the same person's cough-heavy sample in training and their cough-shallow
sample in test. The model would then be rewarded for recognising the speaker
and the microphone. Splitting by participant removes that, and it is the thing
COUGHVID cannot support at all: its recordings carry anonymous identifiers
with no participant grouping, so the leakage there is neither measurable nor
fixable.

The same split is applied to all four conditions. The conditions differ only
in the frequency axis and in whether denoising was applied, so comparing them
is only meaningful if they are trained and measured on identical partitions of
identical participants. The split is written once, here, and every run reads
it.

Output is splits.csv with one row per image: file, label, participant,
partition. Conditions share filenames, so one manifest serves all of them.
"""
import argparse
import collections
import csv
import os
import random

CONDITIONS = ('linear', 'wav_linear', 'mel', 'wav_mel')
FRACTIONS = (0.70, 0.15, 0.15)       # train, val, test


def collect(img_root):
    """Filenames present in every condition, so no run is missing an image."""
    per_cond = {}
    for cond in CONDITIONS:
        here = {}
        for label in ('cough', 'non_cough'):
            d = os.path.join(img_root, cond, label)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith('.png'):
                    here[fn] = label
        per_cond[cond] = here
        print('  %-11s %6d images' % (cond, len(here)))

    common = set.intersection(*(set(v) for v in per_cond.values())) if per_cond else set()
    print('  present in all %d conditions: %d' % (len(CONDITIONS), len(common)))

    # a filename must carry the same label everywhere, or something is wrong
    labels = {}
    for fn in sorted(common):
        ls = {per_cond[c][fn] for c in CONDITIONS}
        assert len(ls) == 1, 'label disagreement for %s: %s' % (fn, ls)
        labels[fn] = ls.pop()
    return labels


def participant_of(fn):
    """Filenames are <participant>__<prompt>.png."""
    return fn.split('__', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True)
    ap.add_argument('--meta')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.img, 'splits.csv')

    print('images found:')
    labels = collect(a.img)
    if not labels:
        raise SystemExit('no images under %s' % a.img)

    by_participant = collections.defaultdict(list)
    for fn, lab in labels.items():
        by_participant[participant_of(fn)].append((fn, lab))

    people = sorted(by_participant)
    print()
    print('participants: %d, recordings: %d' % (len(people), len(labels)))

    # Stratify the participant assignment by how many coughs each contributes,
    # so a short recording set cannot end up concentrated in one partition.
    status = {}
    if a.meta and os.path.exists(a.meta):
        for r in csv.DictReader(open(a.meta, encoding='utf-8', errors='replace')):
            status[r['id']] = (r.get('covid_status') or 'unknown').strip()
        known = sum(1 for p in people if p in status)
        print('participants matched to metadata: %d of %d' % (known, len(people)))

    strata = collections.defaultdict(list)
    for p in people:
        n_cough = sum(1 for _f, l in by_participant[p] if l == 'cough')
        strata[(status.get(p, 'unknown'), n_cough)].append(p)

    rng = random.Random(a.seed)
    assign = {}
    for key in sorted(strata):
        group = sorted(strata[key])
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(FRACTIONS[0] * n))
        n_va = int(round(FRACTIONS[1] * n))
        # with very small strata, guarantee training gets at least one
        if n and n_tr == 0:
            n_tr = 1
            n_va = 0
        for i, p in enumerate(group):
            assign[p] = ('train' if i < n_tr
                         else 'val' if i < n_tr + n_va
                         else 'test')

    rows = []
    for p in people:
        for fn, lab in sorted(by_participant[p]):
            rows.append({'file': fn, 'label': lab, 'participant': p,
                         'partition': assign[p]})

    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['file', 'label', 'participant', 'partition'])
        w.writeheader()
        w.writerows(rows)

    print()
    print('%-8s %10s %10s %10s %9s' % ('split', 'people', 'cough', 'non_cough', 'ratio'))
    print('-' * 52)
    for part in ('train', 'val', 'test'):
        sel = [r for r in rows if r['partition'] == part]
        c = sum(1 for r in sel if r['label'] == 'cough')
        n = len(sel) - c
        ppl = len({r['participant'] for r in sel})
        print('%-8s %10d %10d %10d %9s'
              % (part, ppl, c, n, ('%.2f' % (c / n)) if n else '-'))

    # the guarantee this script exists to provide
    sets = {part: {r['participant'] for r in rows if r['partition'] == part}
            for part in ('train', 'val', 'test')}
    overlap = ((sets['train'] & sets['val'])
               | (sets['train'] & sets['test'])
               | (sets['val'] & sets['test']))
    print()
    print('participant overlap between partitions: %d' % len(overlap))
    assert not overlap, 'speaker disjointness violated'
    print('wrote %s (%d rows)' % (out, len(rows)))


if __name__ == '__main__':
    main()
