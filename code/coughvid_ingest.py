"""Build the classifier-labelled arm of the study from COUGHVID v3.

    python -I coughvid_ingest.py --out ../data/coughvid_img --workers 6
    python -I coughvid_ingest.py --out ../data/coughvid_img --match none

This is the third label regime. Coswara's labels come from what a participant
was asked to do, FSD50K's from human annotators, and COUGHVID's from a
classifier: the corpus ships a cough_detected score produced by the corpus
authors' own detector, and no human cough or not cough judgement for the
corpus as a whole. A model trained and tested here is therefore measured on
its agreement with that detector, not on whether a cough is present, and that
is the point being made rather than a defect to be worked around.

Two further properties of this corpus are reported rather than repaired:

  * the recordings are identified by anonymous UUIDs with no participant
    field, so a speaker-disjoint split cannot be constructed. The split here
    is by recording, and any participant who contributed more than once can
    appear on both sides of it. This is a limitation of the corpus, and it is
    one of the reasons the main experiments use Coswara.
  * the expert-annotated subset was not drawn at random: recordings with high
    cough_detected scores are heavily over-represented in it, so agreement
    rates measured on that subset do not transfer to the whole corpus. See
    label_noise.py, which prints the same caveat beside its own numbers.

Selection, and a reproducibility finding. The study being extended reports
15,619 cough and 7,261 non-cough recordings, 22,880 of 34,434 retained. The
lower count is reproducible: cough_detected <= 0.1272 yields exactly 7,261.
The upper count is not reproducible by any threshold on that column. Six
recordings carry the value 0.8961 exactly, so the count of recordings at or
above a threshold near there steps from 15,617 straight to 15,623, and 15,619
lies strictly between the two. No single cut on cough_detected can produce it,
in either the compiled table shipped inside the archive or the extracted copy,
which are byte identical in the columns that matter. Something further was
therefore applied that the original does not record, and the selection cannot
be recovered from the corpus as published.

Rather than tune a near miss, this uses round thresholds and states them:
cough_detected >= 0.9 for cough and <= 0.1 for non-cough, discarding the
middle. Those give 15,448 and 6,555. The choice is reported in the paper
alongside the finding above, because an unreproducible selection in the work
being extended is worth recording in its own right.

Matching. With --match coswara, the default, the selection is subsampled to
the class prior of the Coswara arm, so that accuracies across the three
regimes are not simply reflecting different priors. The non-cough class is the
binding constraint, so this keeps every non-cough recording and samples the
cough class down to match. --match none images the full selection instead.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402

import denoise                                                 # noqa: E402
import spectro                                                 # noqa: E402

# Round, stated thresholds. See the note on selection above for why the
# original study's counts are not reproducible and were not chased.
LOW = 0.1
HIGH = 0.9

# The Coswara arm: 2,967 cough of 13,314 recordings.
COSWARA_COUGH_FRACTION = 2967.0 / 13314.0

CONDITIONS = [
    ('linear', None, 'linear'),
    ('linear', 'visu', 'wav_linear'),
    ('mel', None, 'mel'),
    ('mel', 'visu', 'wav_mel'),
]
AUDIO_EXT = ('.webm', '.ogg', '.wav')


def select(meta_path):
    """Threshold cough_detected into the two classes, reporting the counts."""
    cough, non, dropped, bad = [], [], 0, 0
    with open(meta_path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                s = float(r['cough_detected'])
            except (TypeError, ValueError):
                bad += 1
                continue
            if s >= HIGH:
                cough.append(r['uuid'])
            elif s <= LOW:
                non.append(r['uuid'])
            else:
                dropped += 1
    print('  cough_detected >= %.2f : %6d' % (HIGH, len(cough)))
    print('  cough_detected <= %.2f : %6d' % (LOW, len(non)))
    print('  between the thresholds : %6d  (discarded)' % dropped)
    print('  retained               : %6d of %6d'
          % (len(cough) + len(non), len(cough) + len(non) + dropped + bad))
    if bad:
        print('  unreadable score       : %6d' % bad)
    return cough, non


def match_prior(cough, non, how, seed):
    """Subsample so the class prior matches the Coswara arm."""
    if how == 'none':
        return cough, non
    rng = random.Random(seed)
    # non-cough is the smaller class here, so it is kept whole and the cough
    # class is sampled down to reach the target fraction.
    want_cough = int(round(len(non) * COSWARA_COUGH_FRACTION
                           / (1.0 - COSWARA_COUGH_FRACTION)))
    if want_cough < len(cough):
        cough = sorted(rng.sample(cough, want_cough))
    else:
        print('  cannot reach the target prior: only %d cough available, '
              'wanted %d' % (len(cough), want_cough))
    tot = len(cough) + len(non)
    print('  matched to the Coswara prior: %d cough of %d (%.1f%%, '
          'Coswara %.1f%%)' % (len(cough), tot, 100.0 * len(cough) / tot,
                               100.0 * COSWARA_COUGH_FRACTION))
    return cough, non


_ZIPS = {}


def _zip_for(path):
    """One open archive per worker process, not one per recording.

    This archive's central directory holds 68,869 entries, and opening it cost
    more than decoding the audio did: measured at 30 recordings a minute, the
    selection would have taken nearly five hours. A pool worker handles many
    recordings in its lifetime, so the handle is opened once and kept. The
    dictionary is per process, because pool workers are separate processes and
    a ZipFile cannot be shared across them.
    """
    z = _ZIPS.get(path)
    if z is None:
        z = _ZIPS[path] = zipfile.ZipFile(path)
    return z


def _one(args):
    """Pool entry point. Must stay module level to survive spawn on Windows."""
    zip_path, member, uuid, label, out_root = args
    tmp = os.path.join(out_root, '_tmp', uuid + os.path.splitext(member)[1])
    try:
        data = _zip_for(zip_path).read(member)
        with open(tmp, 'wb') as fh:
            fh.write(data)
        x = spectro.decode(tmp)
        if x.size < spectro.SR // 10 or not np.isfinite(x).all():
            return False
        if float(np.max(np.abs(x))) == 0.0:
            return False
        cache = {}
        for axis, rule, cond in CONDITIONS:
            if rule not in cache:
                cache[rule] = x if rule is None else denoise.denoise(x, rule=rule)
            d = os.path.join(out_root, cond, label)
            Image.fromarray(spectro.spectrogram_image(cache[rule], axis)).save(
                os.path.join(d, uuid + '.png'), optimize=True)
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def reconcile(out_root):
    """Restrict every condition to the recordings all of them have.

    A recording is imaged into the four conditions in one loop, so a failure
    part way through leaves it present in some conditions and absent from
    others. Two recordings did exactly that. The paired test of
    Section~4.4 compares decisions on the same recordings, so a condition
    holding an image the others lack would make the comparison silently
    uneven. The intersection is taken, the odd images are deleted, and the
    count removed is reported rather than passed over.
    """
    sets = {}
    for _axis, _rule, cond in CONDITIONS:
        s = set()
        for lab in ('cough', 'non_cough'):
            d = os.path.join(out_root, cond, lab)
            if os.path.isdir(d):
                s |= {(lab, f) for f in os.listdir(d) if f.endswith('.png')}
        sets[cond] = s

    common = set.intersection(*sets.values()) if sets else set()
    dropped = 0
    for cond, s in sets.items():
        for lab, fn in sorted(s - common):
            try:
                os.remove(os.path.join(out_root, cond, lab, fn))
                dropped += 1
            except OSError:
                pass
    if dropped:
        print('  reconciled the conditions: removed %d image(s) of %d '
              'recording(s) that failed part way through'
              % (dropped, len({f for _l, f in
                               set().union(*sets.values()) - common})))
    print('  every condition now holds the same %d recordings' % len(common))
    return sorted((f, lab) for lab, f in common)


def write_manifest(path, imaged, seed):
    """A recording-level split. This corpus cannot support a speaker split.

    Stratified by label so the three partitions carry the same prior, with the
    proportions used for the Coswara arm.
    """
    rng = random.Random(seed)
    rows = []
    for label in ('cough', 'non_cough'):
        got = sorted(f for f, lab in imaged if lab == label)
        rng.shuffle(got)
        n = len(got)
        n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
        for i, fn in enumerate(got):
            part = 'train' if i < n_tr else ('val' if i < n_tr + n_va else 'test')
            rows.append({'file': fn, 'label': label,
                         'participant': 'unknown', 'partition': part})
    rows.sort(key=lambda r: r['file'])
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['file', 'label', 'participant',
                                           'partition'])
        w.writeheader()
        w.writerows(rows)
    import collections
    c = collections.Counter((r['partition'], r['label']) for r in rows)
    print('  manifest %s' % path)
    for k in sorted(c):
        print('    %-6s %-10s %5d' % (k[0], k[1], c[k]))
    print('    participant is recorded as unknown throughout: this corpus')
    print('    carries no speaker identifier, so the split is by recording.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=os.path.join(
        '..', 'data', 'coughvid_v3', 'public_dataset_v3.zip'))
    ap.add_argument('--meta', default=os.path.join(
        '..', 'data', 'coughvid_v3', 'coughvid_20211012', 'metadata_compiled.csv'))
    ap.add_argument('--out', default=os.path.join('..', 'data', 'coughvid_img'))
    ap.add_argument('--match', default='coswara', choices=('coswara', 'none'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='0 means no limit')
    a = ap.parse_args()

    for p in (a.zip, a.meta):
        if not os.path.exists(p):
            sys.exit('not found: %s' % p)

    cough, non = select(a.meta)
    cough, non = match_prior(cough, non, a.match, a.seed)
    label_of = {u: 'cough' for u in cough}
    label_of.update({u: 'non_cough' for u in non})

    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            os.makedirs(os.path.join(a.out, cond, lab), exist_ok=True)
    os.makedirs(os.path.join(a.out, '_tmp'), exist_ok=True)

    with zipfile.ZipFile(a.zip) as z:
        audio = {}
        for n in z.namelist():
            stem, ext = os.path.splitext(os.path.basename(n))
            if ext.lower() in AUDIO_EXT and stem in label_of:
                audio[stem] = n

    missing = len(label_of) - len(audio)
    if missing:
        print('  %d selected recordings have no audio in the archive' % missing)

    jobs, imaged = [], []
    for uuid, member in sorted(audio.items()):
        label = label_of[uuid]
        done = os.path.join(a.out, CONDITIONS[0][2], label, uuid + '.png')
        if os.path.exists(done):
            imaged.append((uuid + '.png', label))
        else:
            jobs.append((a.zip, member, uuid, label, a.out))
    if a.limit:
        jobs = jobs[:a.limit]
    print('  to image: %d  (%d already present)' % (len(jobs), len(imaged)))

    if jobs:
        ok = bad = 0
        print('  imaging with %d workers' % a.workers, flush=True)
        with Pool(a.workers) as pool:
            for (zp, mem, uuid, label, _o), good in zip(
                    jobs, pool.imap(_one, jobs, chunksize=8)):
                if good:
                    ok += 1
                    imaged.append((uuid + '.png', label))
                else:
                    bad += 1
                if (ok + bad) % 500 == 0:
                    print('    %d/%d  (%d unusable)' % (ok + bad, len(jobs), bad),
                          flush=True)
        print('  imaged %d, %d unusable' % (ok, bad))

    shutil.rmtree(os.path.join(a.out, '_tmp'), ignore_errors=True)
    imaged = reconcile(a.out)
    write_manifest(os.path.join(a.out, 'splits.csv'), imaged, a.seed)

    print()
    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            d = os.path.join(a.out, cond, lab)
            print('  %-11s %-10s %5d' % (cond, lab, len(os.listdir(d))))


if __name__ == '__main__':
    main()
