"""Find the first NaN in a real training loop and report what produced it.

Checks, at every step, the inputs, the logits, the loss, and the gradients
before and after unscaling and clipping, and stops at the first non-finite
value with enough context to name the cause. Uses a subset of the real
training partition so the data is the same data that fails.
"""
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG = os.path.join('..', 'data', 'coswara_img', 'mel')
MAN = os.path.join('..', 'data', 'coswara_img', 'splits.csv')
N = 1600
BATCH = 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load():
    want = []
    with open(MAN, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['partition'] == 'train':
                want.append((r['file'], r['label']))
    want.sort()
    # interleave the classes so early batches contain both
    cough = [w for w in want if w[1] == 'cough']
    non = [w for w in want if w[1] == 'non_cough']
    mixed = []
    for i in range(max(len(cough), len(non))):
        if i < len(cough):
            mixed.append(cough[i])
        if i < len(non):
            mixed.append(non[i])
    mixed = mixed[:N]

    xs = np.zeros((len(mixed), 224, 224), dtype=np.uint8)
    ys = np.zeros(len(mixed), dtype=np.int64)
    for i, (fn, lab) in enumerate(mixed):
        p = os.path.join(IMG, lab, fn)
        xs[i] = np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
        ys[i] = 0 if lab == 'cough' else 1
    return xs, ys


def main(amp=True):
    print('loading %d real training images' % N, flush=True)
    xs, ys = load()
    print('  uint8 range %d..%d, class counts %s'
          % (xs.min(), xs.max(), np.bincount(ys)))

    dev = 'cuda'
    import torchvision.models as M
    torch.manual_seed(0)
    m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    n_inf=[0]
    n_steps = len(ys) // BATCH
    for step in range(n_steps):
        sl = slice(step * BATCH, (step + 1) * BATCH)
        xb = torch.from_numpy(xs[sl]).to(dev)
        yb = torch.from_numpy(ys[sl]).to(dev)

        xin = xb.unsqueeze(1).float().div_(255.0).expand(-1, 3, -1, -1)
        xin = (xin - mean) / std

        if not torch.isfinite(xin).all():
            print('STEP %d: INPUT is not finite' % step)
            return

        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=amp):
            out = m(xin)
            loss = lossfn(out, yb)

        if not torch.isfinite(out).all():
            print('STEP %d: LOGITS not finite (dtype %s, absmax %s)'
                  % (step, out.dtype, out.abs().max().item()))
            return
        if not torch.isfinite(loss):
            print('STEP %d: LOSS not finite, logits absmax %.3f'
                  % (step, out.abs().max().item()))
            return

        scaler.scale(loss).backward()

        scaler.unscale_(opt)
        bad = [n for n, p in m.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            n_inf[0] += 1
            if n_inf[0] <= 2:
                print('  step %3d: %d tensors non-finite after unscale_, scale=%.0f'
                      % (step, len(bad), scaler.get_scale()))
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue
        if False:
            print('x')
            print('   e.g. %s   scale=%.1f  loss=%.4f'
                  % (bad[:3], scaler.get_scale(), float(loss)))
            # this is the state in which clip_grad_norm_ turns inf into NaN
            tot = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            print('   grad norm reported by clip_grad_norm_: %s' % tot.item())
            after = [n for n, p in m.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
            print('   non-finite tensors after clipping: %d' % len(after))
            print()
            print('   GradScaler would normally SKIP this step, but clipping')
            print('   has already written NaN into the gradients, and on the')
            print('   next step those NaN enter the parameters.')
            return

        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if not all(torch.isfinite(p).all() for p in m.parameters()):
            print('STEP %d: PARAMETERS became non-finite after the step'
                  % step)
            print('   scale=%.1f loss=%.4f' % (scaler.get_scale(), float(loss)))
            return

        if step % 10 == 0:
            print('  step %3d  loss %.4f  scale %.0f'
                  % (step, float(loss), scaler.get_scale()), flush=True)

    print()
    print('%d steps attempted, %d skipped for non-finite grads' % (n_steps, n_inf[0]))
    print('final loss %.4f' % float(loss))


if __name__ == '__main__':
    import sys
    amp = not (len(sys.argv) > 1 and sys.argv[1] == 'off')
    print('AMP %s' % ('on' if amp else 'off'))
    main(amp)
