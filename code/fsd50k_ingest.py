"""Build the external validation set from FSD50K's human-labelled eval split.

    python -I fsd50k_ingest.py --download --out ../data/fsd50k_img
    python -I fsd50k_ingest.py --out ../data/fsd50k_img      (archive already present)

Why FSD50K and why only the eval split. Its labels are human annotations, not
the output of a classifier, and the eval split specifically was exhaustively
verified by the corpus authors. It also supplies the negatives that matter for
this task: Breathing, Sneeze, Laughter and Speech are separate classes, so the
non-cough material can be made deliberately confusable instead of being
whatever happened to score low on a cough detector.

This set is never trained on. It exists to answer one question: does a model
selected on prompted Coswara recordings still separate cough from non-cough
when the audio, the microphones, the annotators and the label definition all
change. That is the external validation the original study lacked.

The eval audio is a zip split into two parts, which Python's zipfile cannot
read directly, so the parts are joined first. Only the clips actually needed
are extracted, and the audio is deleted once imaged, to keep peak disk use
well below what the full corpus would take.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

import denoise
import spectro

BASE = 'https://zenodo.org/records/4060432/files'
PARTS = ['FSD50K.eval_audio.z01', 'FSD50K.eval_audio.zip']   # order matters
GT = 'FSD50K.ground_truth/eval.csv'

# non-cough classes chosen to be acoustically confusable with coughing, plus a
# general pool; a detector that only beats silence is not worth reporting
HARD_NEGATIVES = ('Breathing', 'Sneeze', 'Gasp', 'Sigh', 'Laughter',
                  'Burping_and_eructation', 'Chewing_and_mastication',
                  'Speech', 'Male_speech_and_man_speaking',
                  'Female_speech_and_woman_speaking',
                  'Child_speech_and_kid_speaking', 'Whispering', 'Yell')

CONDITIONS = [
    ('linear', None, 'linear'),
    ('linear', 'visu', 'wav_linear'),
    ('mel', None, 'mel'),
    ('mel', 'visu', 'wav_mel'),
]


def fetch(url, dest):
    if os.path.exists(dest):
        print('  have %s' % os.path.basename(dest))
        return
    print('  fetching %s' % os.path.basename(dest), flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'cough-study'})
    tmp = dest + '.part'
    with urllib.request.urlopen(req, timeout=1200) as fh, open(tmp, 'wb') as out:
        shutil.copyfileobj(fh, out, 1 << 20)
    os.replace(tmp, dest)


def join_parts(raw, joined, drop_parts=True):
    """Concatenate the split archive, then release the parts.

    Python's zipfile cannot read a split archive, so the parts must be joined
    into one file, which doubles the space the corpus occupies while both
    exist. On this machine that is most of the free space, so the parts are
    deleted once the joined file is complete: it is self-sufficient, and the
    parts can be fetched again if needed.
    """
    if os.path.exists(joined):
        print('  joined archive present')
        return
    print('  joining %d parts' % len(PARTS), flush=True)
    tmp = joined + '.part'
    with open(tmp, 'wb') as out:
        for p in PARTS:
            with open(os.path.join(raw, p), 'rb') as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
    os.replace(tmp, joined)
    print('  joined: %.2f GB' % (os.path.getsize(joined) / 2 ** 30))
    if drop_parts:
        for p in PARTS:
            try:
                os.remove(os.path.join(raw, p))
            except OSError:
                pass
        print('  released the split parts')


def choose(gt_path, n_neg, seed):
    """Pick the cough clips and a matched sample of confusable negatives."""
    rows = list(csv.DictReader(open(gt_path, encoding='utf-8')))
    cough, neg = [], []
    for r in rows:
        labs = set(r['labels'].split(','))
        if 'Cough' in labs:
            cough.append(r['fname'])
        elif labs & set(HARD_NEGATIVES):
            neg.append(r['fname'])
    rng = random.Random(seed)
    rng.shuffle(neg)
    keep_neg = neg[:n_neg]
    print('  cough clips: %d' % len(cough))
    print('  confusable negatives available: %d, keeping %d' % (len(neg), len(keep_neg)))
    return cough, keep_neg


def image_one(wav_path, name, label, out_root):
    x = spectro.decode(wav_path)
    if x.size < spectro.SR // 10 or not np.isfinite(x).all():
        return False
    if float(np.max(np.abs(x))) == 0.0:
        return False
    cache = {}
    for axis, rule, cond in CONDITIONS:
        if rule not in cache:
            cache[rule] = x if rule is None else denoise.denoise(x, rule=rule)
        d = os.path.join(out_root, cond, label)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(spectro.spectrogram_image(cache[rule], axis)).save(
            os.path.join(d, name), optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    # The fetch is network bound and the imaging is processor bound, so they
    # are separable: the archive can be pulled while the machine is busy with
    # something else, and the imaging run once it is not.
    ap.add_argument('--download-only', action='store_true')
    ap.add_argument('--keep-parts', action='store_true',
                    help='keep the split parts after joining; they are '
                         'deleted by default because both copies together are '
                         'most of the free space on this machine')
    ap.add_argument('--raw', default=os.path.join('..', 'data', 'fsd50k_raw'))
    ap.add_argument('--gt', default=os.path.join('..', 'data', 'fsd50k_meta', GT))
    ap.add_argument('--out', default=os.path.join('..', 'data', 'fsd50k_img'))
    ap.add_argument('--negatives', type=int, default=1434)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-archive', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.raw, exist_ok=True)
    if a.download:
        for p in PARTS:
            fetch('%s/%s?download=1' % (BASE, p), os.path.join(a.raw, p))

    if a.download_only:
        have = [p for p in PARTS if os.path.exists(os.path.join(a.raw, p))]
        if len(have) != len(PARTS):
            sys.exit('download incomplete: have %d of %d parts'
                     % (len(have), len(PARTS)))
        print('  download only: stopping before the join and imaging stages')
        return

    joined = os.path.join(a.raw, 'eval_joined.zip')
    join_parts(a.raw, joined, drop_parts=not a.keep_parts)

    if not os.path.exists(a.gt):
        sys.exit('ground truth not found at %s' % a.gt)
    cough, neg = choose(a.gt, a.negatives, a.seed)
    wanted = {f: 'cough' for f in cough}
    wanted.update({f: 'non_cough' for f in neg})

    work = os.path.join(a.raw, 'work')
    os.makedirs(work, exist_ok=True)

    n_ok = n_bad = n_miss = 0
    with zipfile.ZipFile(joined) as z:
        inside = {os.path.basename(n): n for n in z.namelist()
                  if n.lower().endswith('.wav')}
        print('  wav entries in archive: %d' % len(inside))
        for i, (fname, label) in enumerate(sorted(wanted.items()), 1):
            member = inside.get(fname + '.wav')
            if member is None:
                n_miss += 1
                continue
            tmp = os.path.join(work, fname + '.wav')
            with z.open(member) as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            if image_one(tmp, fname + '.png', label, a.out):
                n_ok += 1
            else:
                n_bad += 1
            os.remove(tmp)
            if i % 250 == 0:
                print('    %d/%d imaged' % (i, len(wanted)), flush=True)

    shutil.rmtree(work, ignore_errors=True)
    if not a.keep_archive:
        os.remove(joined)
        print('  removed the joined archive')

    print()
    print('imaged %d clips, %d unusable, %d not in archive' % (n_ok, n_bad, n_miss))
    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            d = os.path.join(a.out, cond, lab)
            print('  %-11s %-10s %5d' % (cond, lab,
                                         len(os.listdir(d)) if os.path.isdir(d) else 0))


if __name__ == '__main__':
    main()
