"""Ingest Coswara one date folder at a time, keeping peak disk use small.

    python -I coswara_ingest.py --list
    python -I coswara_ingest.py --folder 20200413 --out ../data/coswara_img
    python -I coswara_ingest.py --all --out ../data/coswara_img

Each folder on GitHub is a tar archive split into 100 MB parts. For each
folder this script downloads the parts, joins them, extracts, turns every
recording into the four spectrogram images the experiments need, and then
deletes the audio before moving to the next folder. Peak disk use is therefore
one folder, not the 16 GB the whole corpus occupies.

Why this corpus. Every participant records nine prompted samples: two coughs
(heavy and shallow), two breathing samples, three sustained vowels and two
counting passages. The label comes from which prompt the sample answers, so it
is not the output of another classifier, which is the defect in labelling
COUGHVID by thresholding its cough_detected score. The participant identifier
also allows the partitions to be made speaker disjoint, and both classes for a
given participant are recorded on the same device in the same session, so
channel and speaker effects cannot masquerade as the class difference.

Images are written as

    <out>/<condition>/<participant>__<prompt>.png

with the partition assigned later from the participant identifier, so that no
participant can appear in more than one partition.
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

import denoise
import spectro

RAW = 'https://raw.githubusercontent.com/iiscleap/Coswara-Data/master'
API = 'https://api.github.com/repos/iiscleap/Coswara-Data/contents'

COUGH_PROMPTS = ('cough-heavy', 'cough-shallow')
NONCOUGH_PROMPTS = ('breathing-deep', 'breathing-shallow', 'vowel-a', 'vowel-e',
                    'vowel-i', 'vowel-o', 'counting-normal', 'counting-fast')

# (image axis, denoise rule or None) -> directory name
CONDITIONS = [
    ('linear', None, 'linear'),
    ('linear', 'visu', 'wav_linear'),
    ('mel', None, 'mel'),
    ('mel', 'visu', 'wav_mel'),
]


WORKERS = max(1, min(7, (os.cpu_count() or 2) - 1))


def _one(args):
    """Pool entry point. Must stay module level to survive spawn on Windows."""
    try:
        return process_audio(*args)
    except Exception:
        return False


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'cough-study'})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


def list_folders():
    return sorted(x['name'] for x in get_json(API + '/')
                  if x['type'] == 'dir' and x['name'].isdigit())


def download_folder(folder, work):
    """Fetch and join the split parts. Returns the path to the tar.gz."""
    parts = sorted(x['name'] for x in get_json('%s/%s' % (API, folder))
                   if x['name'].startswith(folder + '.tar.gz.'))
    if not parts:
        return None
    joined = os.path.join(work, folder + '.tar.gz')
    with open(joined, 'wb') as out:
        for p in parts:
            url = '%s/%s/%s' % (RAW, folder, p)
            req = urllib.request.Request(url, headers={'User-Agent': 'cough-study'})
            with urllib.request.urlopen(req, timeout=600) as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
    return joined


def process_audio(path, participant, prompt, out_root):
    """Write the four condition images for one recording. True if written."""
    label = 'cough' if prompt in COUGH_PROMPTS else 'non_cough'
    name = '%s__%s.png' % (participant, prompt)
    # resumable: a recording already imaged in every condition is left alone,
    # so an interrupted run can be restarted without repeating the decode
    if all(os.path.exists(os.path.join(out_root, cond, label, name))
           for _a, _r, cond in CONDITIONS):
        return None

    x = spectro.decode(path)
    if x.size < spectro.SR // 10:          # under 100 ms is not a usable sample
        return False
    if not np.isfinite(x).all() or float(np.max(np.abs(x))) == 0.0:
        return False

    cache = {}
    for axis, rule, cond in CONDITIONS:
        if rule not in cache:
            cache[rule] = x if rule is None else denoise.denoise(x, rule=rule)
        d = os.path.join(out_root, cond, label)
        os.makedirs(d, exist_ok=True)
        img = spectro.spectrogram_image(cache[rule], axis)
        Image.fromarray(img).save(os.path.join(d, name), optimize=True)
    return True


def run_folder(folder, out_root, work):
    os.makedirs(work, exist_ok=True)
    tgz = download_folder(folder, work)
    if tgz is None:
        print('  %s: no parts found' % folder)
        return 0, 0

    ex = os.path.join(work, 'ex')
    if os.path.isdir(ex):
        shutil.rmtree(ex)
    os.makedirs(ex)
    with tarfile.open(tgz) as t:
        # filter='data' refuses absolute paths and links from the archive
        try:
            t.extractall(ex, filter='data')
        except TypeError:
            t.extractall(ex)
    os.remove(tgz)

    jobs = []
    for dirpath, _dirs, files in os.walk(ex):
        participant = os.path.basename(dirpath)
        for f in files:
            if not f.lower().endswith(('.wav', '.webm', '.ogg')):
                continue
            prompt = os.path.splitext(f)[0]
            if prompt not in COUGH_PROMPTS and prompt not in NONCOUGH_PROMPTS:
                continue
            jobs.append((os.path.join(dirpath, f), participant, prompt, out_root))

    # one ffmpeg process per recording dominates the cost, so the work is
    # spread over a pool; the jobs are independent and each writes its own
    # files, so there is nothing to coordinate
    n_ok = n_skip = n_done = 0
    if jobs:
        with multiprocessing.Pool(WORKERS) as pool:
            for r in pool.imap_unordered(_one, jobs, chunksize=8):
                if r is None:
                    n_done += 1
                elif r:
                    n_ok += 1
                else:
                    n_skip += 1

    shutil.rmtree(ex, ignore_errors=True)
    print('  %s: %d imaged, %d already done, %d unusable'
          % (folder, n_ok, n_done, n_skip), flush=True)
    return n_ok, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--folder')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default=os.path.join('..', 'data', 'coswara_img'))
    ap.add_argument('--work', default=os.path.join('..', 'data', 'coswara_work'))
    a = ap.parse_args()

    if a.list:
        fs = list_folders()
        print('%d date folders' % len(fs))
        print('\n'.join('  ' + f for f in fs))
        return

    todo = [a.folder] if a.folder else (list_folders() if a.all else [])
    if not todo:
        sys.exit('give --folder <name>, --all, or --list')

    os.makedirs(a.out, exist_ok=True)
    tot_ok = tot_skip = 0
    for i, f in enumerate(todo, 1):
        print('[%d/%d] %s' % (i, len(todo), f), flush=True)
        try:
            ok, sk = run_folder(f, a.out, a.work)
        except Exception as exc:                      # one bad folder must not stop the run
            print('  %s FAILED: %s: %s' % (f, type(exc).__name__, exc))
            continue
        tot_ok += ok
        tot_skip += sk

    shutil.rmtree(a.work, ignore_errors=True)
    print()
    print('total %d recordings imaged, %d unusable' % (tot_ok, tot_skip))
    for _axis, _rule, cond in CONDITIONS:
        for lab in ('cough', 'non_cough'):
            d = os.path.join(a.out, cond, lab)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print('  %-12s %-10s %6d images' % (cond, lab, n))


if __name__ == '__main__':
    main()
