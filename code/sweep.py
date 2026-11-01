"""Drive the whole grid in one process, loading each condition's images once.

    python -u -I sweep.py --images ../data/coswara_img --out ../runs_coswara

Reasons this exists rather than a shell loop over train.py. Each separate
process paid for a CUDA context, cuDNN autotuning and a 450 MB read of the
image cache before it trained anything, which came to roughly 40 seconds of
the 11 minutes a run took. Holding the partitions on the GPU across runs
removes all of it, and the images are uploaded four times instead of sixty six.

Order. The ResNet18 block over all four conditions comes first, because that
block alone answers both questions the study asks -- whether denoising changes
anything and whether the frequency axis does, with architecture and
initialisation held fixed. Everything after it broadens the claim, so an
interruption costs breadth rather than the result. DenseNet201 is last: 4 GB
of VRAM forces a batch of 12, making it by far the slowest per run.

A run whose CSV already exists is skipped, so this is resumable.
"""
import argparse
import os
import sys
import time

# Isolated mode (-I) leaves the script's own directory off sys.path, and this
# module is always run that way, so the import of its sibling needs the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                   # noqa: E402
import train as T                                              # noqa: E402

CONDITIONS = ['mel', 'wav_mel', 'linear', 'wav_linear']
SEEDS = [1, 2, 3, 4, 5]
DENSE_CONDITIONS = ['mel', 'wav_mel']
DENSE_SEEDS = [1, 2, 3]


def plan(which):
    """[(condition, arch, seed)] in the order they should run."""
    jobs = []
    if which in ('core', 'all'):
        for c in CONDITIONS:                      # core: one load per condition
            for s in SEEDS:
                jobs.append((c, 'resnet18', s))
    if which in ('rest', 'all'):
        for c in CONDITIONS:
            for arch in ('mobilenetv2', 'compact'):
                for s in SEEDS:
                    jobs.append((c, arch, s))
        for c in DENSE_CONDITIONS:
            for s in DENSE_SEEDS:
                jobs.append((c, 'densenet201', s))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join('..', 'data', 'coswara_img'))
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--out', default=os.path.join('..', 'runs_coswara'))
    ap.add_argument('--plan', default='all', choices=('core', 'rest', 'all'))
    T.add_args(ap)
    a = ap.parse_args()

    T.be_polite()      # this machine runs other work; see train.be_polite
    manifest = a.manifest or os.path.join(a.images, 'splits.csv')
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = T.config(a, dev)

    jobs = plan(a.plan)
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(
                a.out, '%s__%s__seed%d.csv' % (j[1], j[0], j[2])))]
    print('%d jobs in plan %r, %d already done, %d to run'
          % (len(jobs), a.plan, len(jobs) - len(todo), len(todo)))
    if dev == 'cuda':
        print('device: %s' % torch.cuda.get_device_name(0))
    print(flush=True)

    # Group by condition so the images are loaded once per condition, keeping
    # the plan's order otherwise.
    done = 0
    t_start = time.time()
    loaded_cond, parts = None, None
    for cond, arch, seed in todo:
        if cond != loaded_cond:
            parts = None                      # free the old one before loading
            if dev == 'cuda':
                torch.cuda.empty_cache()
            parts = T.load_partitions(os.path.join(a.images, cond), manifest, dev)
            loaded_cond = cond

        tag = '%s__%s__seed%d' % (arch, cond, seed)
        target = os.path.join(a.out, tag + '.csv')
        if os.path.exists(target):
            continue                          # another process got there first
        print('--- %s   %s' % (tag, time.strftime('%H:%M:%S')), flush=True)
        try:
            rows, s = T.train_one(arch, parts, seed, cfg, tag=tag)
        except torch.cuda.OutOfMemoryError:
            print('    out of memory, skipping this run', flush=True)
            torch.cuda.empty_cache()
            continue
        T.write_rows(target, rows)
        done += 1
        per = (time.time() - t_start) / done / 60
        left = per * (len(todo) - done)
        print('    kept epoch %d (val bal %.4f)  test acc %.4f  balanced %.4f'
              % (s['epoch'], s['val_bal'], s['test_acc'], s['test_bal']))
        print('    %d of %d done, %.1f min each, about %.1f h left'
              % (done, len(todo), per, left / 60), flush=True)

    print()
    print('SWEEP COMPLETE %s, %d runs in %.1f h'
          % (time.strftime('%H:%M:%S'), done, (time.time() - t_start) / 3600))


if __name__ == '__main__':
    main()
