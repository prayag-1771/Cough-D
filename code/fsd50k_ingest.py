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

The eval audio is a zip split into two parts. Joining them is necessary but
not sufficient, for the reason recorded in extract_members, so the entries are
pulled out with unzip rather than with Python's zipfile. Only the clips
actually needed are extracted, and each is deleted once imaged, to keep peak
disk use well below what the full corpus would take.

The split has 106 clips carrying the Cough label. That fixes the size of this
set: the negatives are sampled to put the cough prior where the other two
corpora already sit, so a drop in performance here cannot be attributed to a
change in class balance.
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
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


# Set in main(). The extraction shells out, so the executable has to be found
# rather than assumed: unzip is not on the PATH of a plain Windows process even
# when it is on the PATH of the shell this project is usually driven from.
UNZIP = None

CANDIDATES = (
    'C:/Program Files/Git/usr/bin/unzip.exe',
    'C:/Program Files (x86)/Git/usr/bin/unzip.exe',
    '/usr/bin/unzip',
)


def find_unzip(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            sys.exit('no unzip at %s' % explicit)
        return explicit
    found = shutil.which('unzip')
    if found:
        return found
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit('unzip was not found; pass its path with --unzip')


def write_test_manifest(out_root, cond='linear'):
    """List every imaged clip as a test clip.

    This corpus is never trained on, so there is nothing to partition: the
    manifest exists only because the loader reads one, and it records that the
    whole set is held out. The conditions all contain the same clips, so the
    listing is taken from one of them.
    """
    path = os.path.join(out_root, 'splits.csv')
    rows = []
    for lab in ('cough', 'non_cough'):
        d = os.path.join(out_root, cond, lab)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith('.png'):
                rows.append({'file': fn, 'label': lab,
                             'participant': 'unknown', 'partition': 'test'})
    if not rows:
        sys.exit('no images under %s' % out_root)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['file', 'label', 'participant',
                                           'partition'])
        w.writeheader()
        w.writerows(rows)
    n_cough = sum(1 for r in rows if r['label'] == 'cough')
    print('  manifest: %d clips, %d cough, prior %.4f'
          % (len(rows), n_cough, n_cough / float(len(rows))))
    return path


def member_index(joined):
    """Map each wav entry's bare name to its path inside the archive.

    The listing comes from unzip rather than from zipfile for the reason given
    in extract_members.
    """
    r = subprocess.run([UNZIP, '-l', joined], capture_output=True, text=True)
    idx = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].lower().endswith('.wav'):
            idx[os.path.basename(parts[-1])] = parts[-1]
    if not idx:
        sys.exit('no wav entries found in %s' % joined)
    return idx


def extract_members(joined, members, work, batch=120):
    """Extract the wanted entries with unzip, flattened into one directory.

    Concatenating the parts of a split archive does not produce a file that
    Python's zipfile will open. The end of central directory record still
    reports that the archive spans several disks, and zipfile refuses such an
    archive outright. Repairing the record is not enough either: every central
    directory entry stores its offset relative to the disk that holds it, so
    after concatenation the entries that came from the first part need their
    offsets shifted and the ones from the last part do not, and a single
    correction cannot serve both.

    unzip reads the concatenation correctly. It reports the leading part as
    prepended bytes and adjusts, which was verified by extracting entries from
    either side of the boundary and confirming both the stored sizes and that
    the audio decodes. The warnings it prints about the disk fields are
    expected and do not indicate a failed extraction, so its exit status is not
    treated as authoritative; the caller checks for the extracted file instead.

    Entries are requested in batches because the whole list exceeds the
    command line length the platform allows.
    """
    os.makedirs(work, exist_ok=True)
    for i in range(0, len(members), batch):
        chunk = members[i:i + batch]
        subprocess.run([UNZIP, '-j', '-o', '-qq', joined] + chunk
                       + ['-d', work], capture_output=True, text=True)
        print('    extracted %d/%d' % (min(i + batch, len(members)),
                                       len(members)), flush=True)


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
    ap.add_argument('--joined', default=None,
                    help='where to write the joined archive; defaults to --raw. '
                         'The join needs as much free space again as the parts '
                         'occupy, which the data drive may not have.')
    ap.add_argument('--gt', default=os.path.join('..', 'data', 'fsd50k_meta', GT))
    ap.add_argument('--out', default=os.path.join('..', 'data', 'fsd50k_img'))
    # 106 cough clips against 370 negatives puts the prior at 22.3 per cent,
    # which is where both of the other corpora sit.
    ap.add_argument('--negatives', type=int, default=370)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-archive', action='store_true')
    ap.add_argument('--unzip', default=None,
                    help='path to the unzip executable')
    ap.add_argument('--manifest-only', action='store_true',
                    help='write the manifest from images already on disk')
    a = ap.parse_args()

    if a.manifest_only:
        write_test_manifest(a.out)
        return

    global UNZIP
    UNZIP = find_unzip(a.unzip)

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

    jdir = a.joined or a.raw
    os.makedirs(jdir, exist_ok=True)
    joined = os.path.join(jdir, 'eval_joined.zip')
    join_parts(a.raw, joined, drop_parts=not a.keep_parts)

    if not os.path.exists(a.gt):
        sys.exit('ground truth not found at %s' % a.gt)
    cough, neg = choose(a.gt, a.negatives, a.seed)
    wanted = {f: 'cough' for f in cough}
    wanted.update({f: 'non_cough' for f in neg})

    work = os.path.join(a.raw, 'work')
    os.makedirs(work, exist_ok=True)

    inside = member_index(joined)
    print('  wav entries in archive: %d' % len(inside))

    members, labels = [], {}
    n_miss = 0
    for fname, label in sorted(wanted.items()):
        member = inside.get(fname + '.wav')
        if member is None:
            n_miss += 1
            continue
        members.append(member)
        labels[os.path.basename(member)] = (fname, label)
    extract_members(joined, members, work)

    n_ok = n_bad = 0
    for i, base in enumerate(sorted(labels), 1):
        fname, label = labels[base]
        tmp = os.path.join(work, base)
        if not os.path.exists(tmp):
            n_miss += 1
            continue
        if image_one(tmp, fname + '.png', label, a.out):
            n_ok += 1
        else:
            n_bad += 1
        os.remove(tmp)
        if i % 50 == 0:
            print('    %d/%d imaged' % (i, len(labels)), flush=True)

    shutil.rmtree(work, ignore_errors=True)
    if not a.keep_archive:
        os.remove(joined)
        print('  removed the joined archive')

    write_test_manifest(a.out)

    print()
    print('imaged %d clips, %d unusable, %d not in archive' % (n_ok, n_bad, n_miss))
    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            d = os.path.join(a.out, cond, lab)
            print('  %-11s %-10s %5d' % (cond, lab,
                                         len(os.listdir(d)) if os.path.isdir(d) else 0))


if __name__ == '__main__':
    main()
