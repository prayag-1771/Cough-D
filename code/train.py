"""Train one architecture on one image set with one seed, and save decisions.

    python -I train.py --arch resnet18 --images ../data/coswara_img/mel \
                       --manifest ../data/coswara_img/splits.csv --seed 1 --out runs

Writes <out>/<arch>__<images>__seed<N>.csv with one row per test recording, in
the format analyse.py reads, so McNemar's test on paired decisions is possible.

sweep.py imports train_one and load_partitions from here and drives the whole
grid in one process, so there is a single training implementation and no
question about which file produced a given result.

Differences from the original MATLAB scripts, each addressing a specific
weakness rather than being a matter of taste:

  * the validation partition is used for validation, never the test partition;
  * the split is speaker disjoint, taken from the manifest, so no participant
    appears in two partitions;
  * the seed is a parameter, so the spread across runs can be measured, which
    is what decides whether a difference under one point means anything;
  * the weights kept are those with the best validation balanced accuracy,
    selected without reference to the test partition;
  * training stops early once validation stops improving, under one protocol
    applied identically to every run;
  * gradients are clipped, because an unclipped run at the original learning
    rate diverges to a NaN loss on this hardware.

On throughput. The first version decoded PNGs inside a DataLoader and fed the
GPU through a per item Python loop, which left the card about half idle: a
10 epoch run took 11 minutes while the card reported under a gigabyte in use.
Every partition is small enough to sit on the GPU in its entirety as unsigned
bytes (about 670 MB for all three), so it is uploaded once and batches are
gathered on the device. Normalisation to float happens per batch, also on the
device, which keeps the resident cost at one byte per pixel instead of four.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CLASSES = ['cough', 'non_cough']        # index 0 is the positive class
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224

# Every batch has the same shape, so letting cuDNN benchmark its algorithms
# once and reuse the choice is free speed.
torch.backends.cudnn.benchmark = True

# Batch sizes chosen so that the peak reserved memory stays inside the card,
# measured per architecture by bench_arch.py and recorded in
# results/throughput.csv.
#
# The driver on this platform does not refuse an allocation that exceeds the
# four gigabytes of device memory; it satisfies it from host memory instead and
# the traffic crosses the bus on every access. Nothing fails, so the only
# symptom is speed. MobileNetV2 at a batch of 64 reserves 5450 MiB and trains
# at 23 images per second; the same network at 16 reserves 2228 MiB and trains
# at 148, a factor of 6.4 for a quantity that is invisible in any error
# message. DenseNet201 shows the same effect more sharply, 1.6 images per
# second at 32 against 29.8 at 8.
#
# Inverted residuals and dense concatenation are what make these two the
# binding cases: both hold high resolution feature maps alive across many
# layers, so their activation memory is far larger than their operation count
# suggests. The batch is set to leave about a gigabyte spare rather than to the
# largest value that happens to fit, because the figure being protected is a
# peak and the rest of the machine is in use. A batch of 32 for MobileNetV2
# measures the same speed as 16 but leaves only 69 MiB, which is not a margin.
BATCH = {'resnet18': 64, 'mobilenetv2': 16, 'compact': 128, 'densenet201': 8}


def be_polite():
    """Drop to below normal CPU priority.

    This machine runs other work, and a training process that holds the images
    on the GPU needs almost no CPU of its own, so yielding the processor costs
    this job nothing and keeps the rest of the machine responsive.
    """
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        k = ctypes.windll.kernel32
        return bool(k.SetPriorityClass(k.GetCurrentProcess(),
                                       BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:
        try:
            os.nice(10)
            return True
        except Exception:
            return False


class Partition(object):
    """One partition's images, resident on the GPU as unsigned bytes.

    Falls back to pinned host memory if the card cannot hold the images, which
    keeps the one-transfer-per-batch behaviour even when residency is not an
    option, rather than returning to a per item Python loop.
    """

    def __init__(self, root, split, manifest, dev):
        items = self._list(root, split, manifest)
        x, y, names = self._decode(items, root, split, manifest)
        self.names = names
        self.y_np = y

        want = x.nbytes
        resident = False
        if dev == 'cuda':
            free, _total = torch.cuda.mem_get_info()
            # Leave at least twice the images' size free for activations.
            resident = want * 3 < free
        if resident:
            try:
                self.x = torch.from_numpy(x).to(dev)
                self.y = torch.from_numpy(y).to(dev)
                self.where = 'gpu'
            except torch.cuda.OutOfMemoryError:
                resident = False
        if not resident:
            self.x = torch.from_numpy(x).pin_memory()
            self.y = torch.from_numpy(y).pin_memory()
            self.where = 'host'
        self.dev = dev

    @staticmethod
    def _list(root, split, manifest):
        items = []
        if manifest:
            want = {}
            with open(manifest, newline='', encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    if r['partition'] == split:
                        want[r['file']] = r['label']
            for fn, label in sorted(want.items()):
                p = os.path.join(root, label, fn)
                if os.path.exists(p):
                    items.append((p, CLASSES.index(label), fn))
            missing = len(want) - len(items)
            if missing:
                print('  %s/%s: %d of %d listed images absent'
                      % (os.path.basename(root), split, missing, len(want)))
        else:
            for label, name in enumerate(CLASSES):
                d = os.path.join(root, split, name)
                if not os.path.isdir(d):
                    continue
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith('.png'):
                        items.append((os.path.join(d, fn), label, fn))
        if not items:
            raise RuntimeError('no images for partition %r under %s' % (split, root))
        return items

    @staticmethod
    def _decode(items, root, split, manifest):
        # Decoding the PNGs takes minutes per partition, which would be
        # repeated for every run in the grid, so the decoded bytes are cached.
        # The cache is keyed on the manifest's modification time, so rebuilding
        # the split invalidates it automatically.
        stamp = os.path.getmtime(manifest) if manifest else 0.0
        cache = os.path.join(root, '_cache_%s.npz' % split)
        if os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=False)
                if float(z['stamp']) == stamp and int(z['n']) == len(items):
                    return (z['x'], z['y'], [str(s) for s in z['names']])
            except Exception:
                pass            # a damaged cache is rebuilt, never trusted

        x = np.zeros((len(items), SIZE, SIZE), dtype=np.uint8)
        y = np.zeros(len(items), dtype=np.int64)
        names = []
        for i, (path, label, fn) in enumerate(items):
            a = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
            if a.shape != (SIZE, SIZE):
                a = np.asarray(Image.fromarray(a).resize((SIZE, SIZE)), dtype=np.uint8)
            x[i] = a
            y[i] = label
            names.append(fn)
        try:
            np.savez(cache, x=x, y=y, names=np.array(names), stamp=stamp, n=len(items))
        except Exception as exc:
            print('  could not write cache: %s' % exc)
        return x, y, names

    def __len__(self):
        return self.x.shape[0]

    def batch(self, idx, mean, std):
        """Index tensor -> normalised float batch (B,3,H,W) on the device."""
        xb = self.x[idx]
        yb = self.y[idx]
        if self.where == 'host':
            xb = xb.to(self.dev, non_blocking=True)
            yb = yb.to(self.dev, non_blocking=True)
        xb = xb.unsqueeze(1).float().div_(255.0).expand(-1, 3, -1, -1)
        return (xb - mean) / std, yb


def load_partitions(images, manifest, dev):
    t0 = time.time()
    parts = tuple(Partition(images, s, manifest, dev)
                  for s in ('train', 'val', 'test'))
    print('  images %s | train %d  val %d  test %d  | held on %s (%.1f s)'
          % (os.path.basename(images.rstrip('/\\')), len(parts[0]), len(parts[1]),
             len(parts[2]), parts[0].where, time.time() - t0), flush=True)
    return parts


def build(arch):
    import torchvision.models as M
    if arch == 'compact':
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        return nn.Sequential(*blk(3, 8), *blk(8, 16), *blk(16, 32),
                             nn.Flatten(), nn.Linear(32 * 28 * 28, 128),
                             nn.ReLU(inplace=True), nn.Dropout(0.5),
                             nn.Linear(128, 2))
    if arch == 'resnet18':
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if arch == 'mobilenetv2':
        m = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    if arch == 'densenet201':
        m = M.densenet201(weights=M.DenseNet201_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m
    raise ValueError('unknown arch %r' % (arch,))


@torch.no_grad()
def evaluate(model, part, dev, amp, mean, std, batch):
    model.eval()
    n = len(part)
    preds, probs = [], []
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n), device=part.x.device)
        xb, _ = part.batch(idx, mean, std)
        with torch.autocast('cuda', enabled=amp):
            p = torch.softmax(model(xb).float(), dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def balanced_accuracy(truth, pred):
    out = []
    for c in (0, 1):
        m = truth == c
        out.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    return float(np.mean(out))


def train_one(arch, parts, seed, cfg, quiet=False, tag=None):
    """Train one (arch, seed) on already loaded partitions.

    Returns (rows, summary). rows is the per recording decision table;
    summary carries the kept epoch and the accuracies.

    With cfg['weights'] set to a directory, and a tag given, the kept weights
    are written there as <tag>.pt. external_eval.py needs them: applying a
    model to a corpus it was not trained on is inference only, so it must not
    require retraining, and the alternative of evaluating every external set
    inside this function would force the external images to exist before
    training could start.
    """
    tr, va, te = parts
    dev = cfg['dev']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    batch = cfg.get('batch') or BATCH.get(arch, 32)
    amp = cfg['amp'] and dev == 'cuda'

    model = build(arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    n = len(tr)
    best_bal, best_state, best_epoch = -1.0, None, -1
    skipped = 0
    stale = 0
    t0 = time.time()

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(n, device=tr.x.device)
        tot = seen = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = tr.batch(idx, mean, std)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=amp):
                loss = lossfn(model(xb), yb)
            scaler.scale(loss).backward()
            if cfg['clip']:
                if amp:
                    scaler.unscale_(opt)
                # Clipping an infinite gradient norm produces NaN, so the step
                # is abandoned instead. With mixed precision off this branch
                # should never be taken; the count is reported so a silent
                # failure cannot pass for a successful run.
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
                if not torch.isfinite(norm):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * yb.size(0)
            seen += yb.size(0)

        vp, _ = evaluate(model, va, dev, amp, mean, std, batch * 2)
        bal = balanced_accuracy(va.y_np, vp)
        if not quiet:
            print('    epoch %2d  loss %.4f  val bal acc %.4f  (%.1f min)%s'
                  % (ep, tot / max(seen, 1), bal, (time.time() - t0) / 60,
                     '  [%d steps skipped]' % skipped if skipped else ''),
                  flush=True)

        if bal > best_bal:
            best_bal, best_epoch, stale = bal, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg['patience']:
                if not quiet:
                    print('    stopping early: no improvement for %d epochs'
                          % stale, flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        wdir = cfg.get('weights')
        if wdir and tag:
            os.makedirs(wdir, exist_ok=True)
            torch.save({'arch': arch, 'seed': seed, 'tag': tag,
                        'epoch': best_epoch, 'val_bal': best_bal,
                        'classes': CLASSES, 'state': best_state},
                       os.path.join(wdir, tag + '.pt'))

    tp, prob = evaluate(model, te, dev, amp, mean, std, batch * 2)
    truth = te.y_np
    rows = [[te.names[k], CLASSES[truth[k]], CLASSES[tp[k]],
             '%.6f' % prob[k, 0], '%.6f' % prob[k, 1]] for k in range(len(truth))]
    summary = {'epoch': best_epoch, 'val_bal': best_bal,
               'test_acc': float((tp == truth).mean()),
               'test_bal': balanced_accuracy(truth, tp),
               'skipped': skipped, 'minutes': (time.time() - t0) / 60}

    del model, opt, scaler, best_state
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return rows, summary


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'true', 'pred', 'score_1', 'score_2'])
        w.writerows(rows)


def add_args(ap):
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch', type=int, default=0, help='0 selects per architecture')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weights', default=None,
                    help='directory to keep the selected weights in, for '
                         'external_eval.py; omitted, they are discarded')
    # Mixed precision is off by default and deliberately so. On a GTX 1650,
    # which has compute capability 7.5 but no tensor cores, it buys very little
    # speed, and it cost correctness here: at the scaler's initial scale of
    # 65536 the gradients of the first convolution overflow the half precision
    # range, so clip_grad_norm_ divides by an infinite norm and writes NaN into
    # the gradients after the scaler's own overflow check has already run. The
    # loss is NaN from the first step and the network never leaves the majority
    # class. Pass --amp on to re-enable it.
    ap.add_argument('--amp', default='off', choices=('on', 'off'))
    return ap


def config(a, dev):
    return {'dev': dev, 'epochs': a.epochs, 'patience': a.patience,
            'batch': a.batch, 'lr': a.lr, 'clip': a.clip,
            'amp': a.amp == 'on', 'weights': getattr(a, 'weights', None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='runs')
    add_args(ap)
    a = ap.parse_args()

    cond = os.path.basename(a.images.rstrip('/\\'))
    tag = '%s__%s__seed%d' % (a.arch, cond, a.seed)
    os.makedirs(a.out, exist_ok=True)
    target = os.path.join(a.out, tag + '.csv')
    if os.path.exists(target):
        print('%s exists, skipping' % tag)
        return

    be_polite()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    parts = load_partitions(a.images, a.manifest, dev)
    print('%s' % tag, flush=True)
    rows, s = train_one(a.arch, parts, a.seed, config(a, dev), tag=tag)
    write_rows(target, rows)
    print('  kept epoch %d (val bal %.4f)  test acc %.4f  balanced %.4f  -> %s'
          % (s['epoch'], s['val_bal'], s['test_acc'], s['test_bal'], target),
          flush=True)


if __name__ == '__main__':
    main()
