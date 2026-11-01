"""Find why the loss went to NaN, by changing one thing at a time.

Runs a few dozen optimiser steps per configuration and reports the loss, so
the cause is identified rather than guessed at. The GTX 1650 has no tensor
cores, so mixed precision is the first suspect; the learning rate inherited
from the original MATLAB scripts is the second.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ROOT = os.path.join('..', 'data', 'coswara_img', 'mel')
STEPS = 40


def batch(n=32):
    xs, ys = [], []
    for label, cls in enumerate(('cough', 'non_cough')):
        d = os.path.join(ROOT, cls)
        fs = sorted(os.listdir(d))[:n // 2]
        for f in fs:
            a = np.asarray(Image.open(os.path.join(d, f)).convert('L'),
                           dtype=np.float32) / 255.0
            t = torch.from_numpy(a).unsqueeze(0).repeat(3, 1, 1)
            xs.append((t - MEAN) / STD)
            ys.append(label)
    return torch.stack(xs), torch.tensor(ys)


def run(amp, lr, clip=None):
    import torchvision.models as M
    torch.manual_seed(0)
    dev = 'cuda'
    m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    lossfn = nn.CrossEntropyLoss()

    x, y = batch()
    x, y = x.to(dev), y.to(dev)

    losses = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=amp):
            out = m(x)
            loss = lossfn(out, y)
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))

    bad = next((i for i, v in enumerate(losses) if not np.isfinite(v)), None)
    print('  amp=%-5s lr=%-7s clip=%-5s first %.4f  last %.4f  %s'
          % (amp, lr, clip, losses[0], losses[-1],
             'NaN at step %d' % bad if bad is not None else 'finite throughout'))
    return bad is None


if __name__ == '__main__':
    print('torch %s on %s' % (torch.__version__, torch.cuda.get_device_name(0)))
    print('%d steps per configuration, same batch each step' % STEPS)
    print()
    for amp in (True, False):
        for lr in (1e-3, 1e-4, 3e-5):
            run(amp, lr)
    print()
    print('with gradient clipping:')
    run(True, 1e-3, clip=1.0)
    run(False, 1e-3, clip=1.0)
