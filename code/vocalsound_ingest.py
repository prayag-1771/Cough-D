"""Build an external validation set from VocalSound's held out test speakers.

    python -I vocalsound_ingest.py --download --out ../data/vocalsound_img
    python -I vocalsound_ingest.py --out ../data/vocalsound_img   (zip present)

Why this corpus. The Coswara negatives are prompted breathing, sustained
vowels and counting, which a detector might separate from coughing on gross
spectral shape alone. VocalSound's five non-cough classes are laughter, sighs,
sneezes, sniffs and throat clearing: all forced, transient, glottal sounds,
which is the confusion that matters. If the accuracy measured on Coswara is an
artefact of easy negatives, it should fall here, and that fall is a more
interesting result than the accuracy itself.

What this does and does not establish. VocalSound's labels are prompt derived,
as Coswara's are: a participant was asked to cough, and the recording is
labelled cough. So this set changes the speakers, the devices, the countries
and the difficulty of the negatives, but not the way the label was obtained.
It therefore measures cross corpus generalisation within one label regime, and
nothing about label provenance; FSD50K, whose eval split is human annotated,
is what the provenance comparison needs.

Only the corpus authors' own test split is imaged. Their splits are speaker
disjoint, so this set shares no speaker with their training material, and it
is the partition a reader can reproduce. This set is never trained on.

21,024 clips from 3,365 subjects, released at 16 kHz, which is the rate this
pipeline already works at, so no resampling is involved.
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402

import denoise                                                 # noqa: E402
import spectro                                                 # noqa: E402

URL = 'https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1'
ZIPNAME = 'vs_release_16k.zip'

POSITIVE = 'cough'
NEGATIVE = ('laughter', 'sigh', 'sneeze', 'sniff', 'throatclearing')
# f0003_0_cough.wav -> speaker f0003, take 0, label cough
NAME = re.compile(r'^([fm]\d+)_(\d+)_([a-z]+)\.wav$')

CONDITIONS = [
    ('linear', None, 'linear'),
    ('linear', 'visu', 'wav_linear'),
    ('mel', None, 'mel'),
    ('mel', 'visu', 'wav_mel'),
]


def fetch(url, dest):
    if os.path.exists(dest):
        print('  have %s (%.2f GB)' % (os.path.basename(dest),
                                       os.path.getsize(dest) / 2 ** 30))
        return
    print('  fetching %s' % os.path.basename(dest), flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'cough-study'})
    tmp = dest + '.part'
    with urllib.request.urlopen(req, timeout=1200) as fh, open(tmp, 'wb') as out:
        got = 0
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if got % (128 << 20) < (1 << 20):
                print('    %.2f GB' % (got / 2 ** 30), flush=True)
    os.replace(tmp, dest)
    print('  got %.2f GB' % (os.path.getsize(dest) / 2 ** 30))


def test_basenames(z):
    """The corpus authors' test split, by clip basename.

    Their datafiles carry the split; the names are read out of whatever
    structure the json uses rather than assuming one, and if no test
    datafile is present every clip is used instead, which is stated.
    """
    cand = [n for n in z.namelist()
            if os.path.basename(n) in ('te.json', 'test.json')]
    if not cand:
        return None
    raw = z.read(cand[0]).decode('utf-8', 'replace')
    names = set(re.findall(r'[fm]\d+_\d+_[a-z]+\.wav', raw))
    if not names:
        try:                            # a shape the regex would not catch
            obj = json.loads(raw)
            names = {os.path.basename(d['wav']) for d in obj.get('data', [])}
        except Exception:
            return None
    return names or None


def label_of(base):
    m = NAME.match(base)
    if not m:
        return None, None
    spk, _take, cls = m.groups()
    if cls == POSITIVE:
        return spk, 'cough'
    if cls in NEGATIVE:
        return spk, 'non_cough'
    return None, None


def _one(args):
    """Pool entry point. Must stay module level to survive spawn on Windows."""
    zip_path, member, base, label, out_root = args
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = z.read(member)
        tmp = os.path.join(out_root, '_tmp', base)
        with open(tmp, 'wb') as fh:
            fh.write(data)
        try:
            x = spectro.decode(tmp)
            if x.size < spectro.SR // 10 or not np.isfinite(x).all():
                return False
            if float(np.max(np.abs(x))) == 0.0:
                return False
            png = os.path.splitext(base)[0] + '.png'
            cache = {}
            for axis, rule, cond in CONDITIONS:
                if rule not in cache:
                    cache[rule] = x if rule is None else denoise.denoise(x, rule=rule)
                d = os.path.join(out_root, cond, label)
                Image.fromarray(spectro.spectrogram_image(cache[rule], axis)).save(
                    os.path.join(d, png), optimize=True)
            return True
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    # The download is network bound and the imaging is processor bound, so
    # they are separable: the archive can be fetched while the machine is
    # busy, and the imaging queued until it is not.
    ap.add_argument('--download-only', action='store_true')
    ap.add_argument('--raw', default=os.path.join('..', 'data', 'vocalsound_raw'))
    ap.add_argument('--out', default=os.path.join('..', 'data', 'vocalsound_img'))
    ap.add_argument('--split', default='te', choices=('te', 'all'))
    # This machine runs other work. Two workers leaves most of the processor
    # to it; raise it only when the machine is otherwise idle.
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--keep-archive', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.raw, exist_ok=True)
    zip_path = os.path.join(a.raw, ZIPNAME)
    if a.download:
        fetch(URL, zip_path)
    if a.download_only:
        if not os.path.exists(zip_path):
            sys.exit('download did not produce %s' % zip_path)
        print('  download only: stopping before the imaging stage')
        return
    if not os.path.exists(zip_path):
        sys.exit('archive not found at %s -- pass --download' % zip_path)

    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            os.makedirs(os.path.join(a.out, cond, lab), exist_ok=True)
    os.makedirs(os.path.join(a.out, '_tmp'), exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        wavs = [n for n in z.namelist() if n.lower().endswith('.wav')]
        keep = test_basenames(z) if a.split == 'te' else None
        if a.split == 'te' and keep is None:
            print('  no test datafile in the archive: using every clip')

        jobs, speakers, counts = [], set(), {'cough': 0, 'non_cough': 0}
        for n in wavs:
            base = os.path.basename(n)
            if keep is not None and base not in keep:
                continue
            spk, label = label_of(base)
            if label is None:
                continue
            png = os.path.join(a.out, CONDITIONS[0][2], label,
                               os.path.splitext(base)[0] + '.png')
            speakers.add(spk)
            counts[label] += 1
            if not os.path.exists(png):          # resumable
                jobs.append((zip_path, n, base, label, a.out))

    print('  wav entries in archive: %d' % len(wavs))
    print('  split %r: %d cough, %d non cough, %d speakers'
          % (a.split, counts['cough'], counts['non_cough'], len(speakers)))
    print('  to image: %d (%d already done)'
          % (len(jobs), counts['cough'] + counts['non_cough'] - len(jobs)))
    if not jobs:
        print('  nothing to do')
        return

    ok = bad = 0
    print('  imaging with %d workers' % a.workers, flush=True)
    with Pool(a.workers) as pool:
        for i, good in enumerate(pool.imap_unordered(_one, jobs, chunksize=8), 1):
            ok += bool(good)
            bad += (not good)
            if i % 250 == 0:
                print('    %d/%d  (%d unusable)' % (i, len(jobs), bad), flush=True)

    shutil.rmtree(os.path.join(a.out, '_tmp'), ignore_errors=True)
    if not a.keep_archive:
        os.remove(zip_path)
        print('  removed the archive')

    print()
    print('imaged %d clips, %d unusable' % (ok, bad))
    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            d = os.path.join(a.out, cond, lab)
            print('  %-11s %-10s %5d' % (cond, lab, len(os.listdir(d))))


if __name__ == '__main__':
    main()
