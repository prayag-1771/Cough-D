"""Apply saved models to a corpus they were never trained on.

    python -I external_eval.py --weights ../weights --images ../data/fsd50k_img \
                               --out ../runs_fsd50k

Every model in the grid was selected on a validation split drawn from the same
corpus as its training data, which measures how well it fits that corpus and
nothing more. The question this script answers is the one the original study
left open: whether a detector chosen that way still separates cough from non
cough when the recordings, the microphones, the annotators and the definition
of the label all change at once.

Nothing is trained here. Each saved model is loaded, run once over the external
set under the condition it was trained on, and its per recording decisions are
written out in the same format the training runs produce, so the same analysis
code computes intervals and paired tests over either corpus without knowing
which it is looking at.

A model is only ever shown the external images for its own condition. Comparing
a model trained on wavelet shrunk inputs against unprocessed ones would confound
the thing being measured with a change of input, and the condition is recorded
in each checkpoint, so there is no need to infer it from the file name.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T


def load_checkpoints(wdir):
    """Checkpoints grouped by the condition they were trained on."""
    by_cond = {}
    for fn in sorted(os.listdir(wdir)):
        if not fn.endswith('.pt'):
            continue
        path = os.path.join(wdir, fn)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        tag = ck.get('tag') or os.path.splitext(fn)[0]
        parts = tag.split('__')
        if len(parts) != 3:
            print('  skipping %s: cannot read a condition from %r' % (fn, tag))
            continue
        cond = parts[1]
        if ck.get('classes') and list(ck['classes']) != list(T.CLASSES):
            sys.exit('%s was trained with a different class order' % fn)
        by_cond.setdefault(cond, []).append((tag, ck))
    return by_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join('..', 'weights'))
    ap.add_argument('--images', default=os.path.join('..', 'data', 'fsd50k_img'))
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--out', default=os.path.join('..', 'runs_fsd50k'))
    ap.add_argument('--split', default='test')
    # Inference needs no stored activations, so it fits in far less memory than
    # training. The cap keeps this usable while the grid has the card.
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest = a.manifest or os.path.join(a.images, 'splits.csv')
    if not os.path.exists(manifest):
        sys.exit('no manifest at %s' % manifest)
    os.makedirs(a.out, exist_ok=True)

    by_cond = load_checkpoints(a.weights)
    if not by_cond:
        sys.exit('no checkpoints under %s' % a.weights)
    print('device: %s' % (torch.cuda.get_device_name(0)
                          if dev == 'cuda' else 'cpu'))
    print('conditions: %s'
          % ', '.join('%s (%d)' % (c, len(v)) for c, v in sorted(by_cond.items())))

    mean = torch.tensor(T.MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(T.STD, device=dev).view(1, 3, 1, 1)

    n_written = 0
    for cond in sorted(by_cond):
        root = os.path.join(a.images, cond)
        if not os.path.isdir(root):
            print('  %s: no images, skipped' % cond)
            continue
        # One decode for every model sharing this condition.
        part = T.Partition(root, a.split, manifest, dev)
        truth = part.y_np
        print('  %-11s %d recordings, %d cough'
              % (cond, len(part), int((truth == 0).sum())), flush=True)

        for tag, ck in by_cond[cond]:
            out_path = os.path.join(a.out, tag + '.csv')
            if os.path.exists(out_path):
                continue
            model = T.build(ck['arch']).to(dev)
            model.load_state_dict(ck['state'])
            batch = min(a.batch, T.BATCH.get(ck['arch'], a.batch))
            pred, prob = T.evaluate(model, part, dev, False, mean, std, batch)
            del model
            if dev == 'cuda':
                torch.cuda.empty_cache()

            rows = [(part.names[i], T.CLASSES[truth[i]], T.CLASSES[pred[i]],
                     '%.6f' % prob[i][0], '%.6f' % prob[i][1])
                    for i in range(len(part))]
            T.write_rows(out_path, rows)
            n_written += 1
            print('    %-28s acc %.4f  balanced %.4f'
                  % (tag, float((pred == truth).mean()),
                     T.balanced_accuracy(truth, pred)), flush=True)

    print()
    print('wrote %d decision files to %s' % (n_written, a.out))


if __name__ == '__main__':
    main()
