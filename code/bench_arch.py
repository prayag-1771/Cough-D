"""Measure training throughput and peak memory for one architecture.

    python -I bench_arch.py mobilenetv2
    python -I bench_arch.py mobilenetv2 resnet18 --batches 64 32

Why this exists. MobileNetV2 trains an order of magnitude slower per epoch
here than ResNet18, although it performs roughly six times fewer operations
per image. A gap in that direction is not explained by arithmetic, so it has
to be measured rather than reasoned about: either the depthwise separable
convolutions are reaching a slow kernel, or the activations do not fit beside
everything else on a four gigabyte card and the allocator is spending its time
releasing and re-reserving blocks.

The two candidate remedies are distinguished here. A smaller batch reduces the
activation footprint and so tests the memory explanation. The channels last
memory format changes which convolution kernels are eligible and so tests the
kernel explanation. Both are reported with the peak reserved bytes beside the
rate, because a remedy that raises throughput without lowering the peak was
not fixing a memory problem.

Synthetic images of the true shape are used. The quantity being measured is
the cost of a forward and backward pass at a given shape, which does not
depend on what the pixels are.
"""
import argparse
import sys
import time

import torch
import torch.nn as nn

SIZE = 224
STEPS = 12
WARMUP = 4


def build(arch):
    """The same constructors train.py uses, without its pretrained downloads."""
    import torchvision.models as tvm
    if arch == 'resnet18':
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == 'mobilenetv2':
        m = tvm.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif arch == 'densenet201':
        m = tvm.densenet201(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif arch == 'compact':
        m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2),
                          nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                          nn.MaxPool2d(4),
                          nn.Flatten(), nn.Linear(32 * 28 * 28, 128),
                          nn.ReLU(), nn.Linear(128, 2))
    else:
        sys.exit('unknown architecture: %s' % arch)
    return m


def run(arch, batch, last, dev):
    """Steps per second and peak reserved megabytes for one configuration."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(arch).to(dev)
    if last:
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch, 3, SIZE, SIZE, device=dev)
    if last:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, 2, (batch,), device=dev)

    m.train()
    t0 = None
    try:
        for i in range(WARMUP + STEPS):
            opt.zero_grad(set_to_none=True)
            loss_fn(m(x), y).backward()
            opt.step()
            if i == WARMUP - 1:
                torch.cuda.synchronize()
                t0 = time.time()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        del m, opt, x, y
        torch.cuda.empty_cache()
        return None, None
    dt = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 2 ** 20
    del m, opt, x, y
    torch.cuda.empty_cache()
    return batch * STEPS / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archs', nargs='+')
    ap.add_argument('--batches', type=int, nargs='+', default=None)
    ap.add_argument('--layouts', nargs='+', default=['contiguous', 'channels_last'],
                    choices=['contiguous', 'channels_last'])
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit('no cuda device')
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True

    free, total = torch.cuda.mem_get_info()
    print('card: %s   %.0f MiB free of %.0f MiB'
          % (torch.cuda.get_device_name(0), free / 2 ** 20, total / 2 ** 20))
    print('%-13s %6s %-9s %10s %12s' %
          ('arch', 'batch', 'layout', 'images/s', 'peak MiB'))

    defaults = {'resnet18': [64, 32], 'mobilenetv2': [64, 32],
                'compact': [128, 64], 'densenet201': [12, 8]}
    for arch in a.archs:
        for batch in (a.batches or defaults.get(arch, [64, 32])):
            for last in [l == 'channels_last' for l in a.layouts]:
                rate, peak = run(arch, batch, last, dev)
                layout = 'channels_last' if last else 'contiguous'
                if rate is None:
                    print('%-13s %6d %-9s %10s %12s'
                          % (arch, batch, layout[:9], 'OOM', '-'))
                else:
                    print('%-13s %6d %-9s %10.1f %12.0f'
                          % (arch, batch, layout[:9], rate, peak))
                sys.stdout.flush()


if __name__ == '__main__':
    main()
