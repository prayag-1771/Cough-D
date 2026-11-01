"""Statistics recoverable from the reported results without rerunning anything.

Inputs are the eight published rate rows and the composition of the test
partition. Everything below follows from those by arithmetic: no value is
assumed, and nothing here substitutes for the experiments the manuscript
still needs.
"""
import math

P, N = 3124, 1453          # cough, noncough in the test partition
TOT = P + N

ROWS = [
    ('Compact CNN', 'Linear',            93.45, 0.9754, 0.9273),
    ('Compact CNN', 'Wavelet + linear',  93.86, 0.9665, 0.9427),
    ('ResNet18',    'Mel',               95.76, 0.9584, 0.9805),
    ('ResNet18',    'Wavelet + Mel',     96.53, 0.9759, 0.9731),
    ('MobileNetV2', 'Mel',               95.76, 0.9797, 0.9577),
    ('MobileNetV2', 'Wavelet + Mel',     96.11, 0.9700, 0.9731),
    ('DenseNet201', 'Mel',               96.15, 0.9694, 0.9744),
    ('DenseNet201', 'Wavelet + Mel',     96.09, 0.9652, 0.9779),
]


def wilson(k, n, z=1.96):
    """Wilson score interval: behaves properly near 0 and 1, unlike Wald."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcc(tp, tn, fp, fn):
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return ((tp * tn - fp * fn) / math.sqrt(d)) if d else 0.0


print('=' * 94)
print('RECOVERED CONFUSION STRUCTURE AND INTERVAL ESTIMATES  (test n = %d)' % TOT)
print('=' * 94)
hdr = ('%-13s %-17s %5s %4s %4s %5s | %-18s %-16s %6s %6s'
       % ('Model', 'Input', 'TP', 'FN', 'FP', 'TN',
          'Accuracy 95% CI', 'Specificity', 'BalAcc', 'MCC'))
print(hdr)
print('-' * 94)

out = []
for m, inp, acc, pr, rc in ROWS:
    tp = round(rc * P)
    fn = P - tp
    fp = round(tp * (1 - pr) / pr)
    tn = N - fp
    a = (tp + tn) / TOT
    lo, hi = wilson(tp + tn, TOT)
    spec = tn / N
    bal = (rc + spec) / 2
    out.append((m, inp, tp, fn, fp, tn, a, lo, hi, spec, bal, mcc(tp, tn, fp, fn)))
    print('%-13s %-17s %5d %4d %4d %5d | %6.2f [%5.2f, %5.2f] %6.2f%%          %6.2f %6.3f'
          % (m, inp, tp, fn, fp, tn, a * 100, lo * 100, hi * 100,
             spec * 100, bal * 100, mcc(tp, tn, fp, fn)))

print()
print('=' * 94)
print('WIDTH OF THE ACCURACY INTERVAL vs THE DIFFERENCES BEING CLAIMED')
print('=' * 94)
widths = [(hi - lo) * 100 for *_, lo, hi, _, _, _ in
          [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]) for r in out]]
w = sum((r[8] - r[7]) * 100 for r in out) / len(out)
print('mean 95%% CI width on accuracy      : %.2f points' % w)
print('largest denoising effect claimed    : 0.77 points  (ResNet18)')
print('smallest denoising effect claimed   : 0.06 points  (DenseNet201)')
print()
print('Every effect the manuscript reports is smaller than the width of the')
print('interval around a single accuracy estimate. With one run and one split,')
print('none of the eight configurations is distinguishable from its pair.')

print()
print('=' * 94)
print('PAIRED COMPARISON, ResNet18 with and without denoising')
print('=' * 94)
a0 = next(r for r in out if r[0] == 'ResNet18' and r[1] == 'Mel')
a1 = next(r for r in out if r[0] == 'ResNet18' and r[1] == 'Wavelet + Mel')
n01 = a0[4] - a1[4]          # false positives removed
n10 = a1[3] - a0[3]          # false negatives introduced
print('false positives  %d -> %d   (%+d)' % (a0[4], a1[4], a1[4] - a0[4]))
print('false negatives  %d -> %d   (%+d)' % (a0[3], a1[3], a1[3] - a0[3]))
print()
print('McNemar needs the two discordant counts b and c, which are properties of')
print('the paired per sample predictions. A confusion matrix gives only margins,')
print('so b and c are NOT recoverable here. The test requires the prediction')
print('files. Lower bound on discordant pairs from the margins: %d.' % (abs(n01) + abs(n10)))
