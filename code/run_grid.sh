#!/usr/bin/env bash
# The experimental grid, in priority order.
#
# train.py skips a run whose CSV already exists, so this is resumable and can
# be interrupted and relaunched without losing work.
#
# Order matters. ResNet18 across all four conditions comes first because that
# block alone answers both questions the study is about: whether denoising
# changes anything, and whether the frequency axis does, with the architecture
# and the initialisation held fixed. Everything after it broadens the claim.
# DenseNet201 is last: 4 GB of VRAM forces a batch of 12, so it is by far the
# slowest per run.
set -u

IMG=../data/coswara_img
MAN=$IMG/splits.csv
OUT=../runs_coswara
SEEDS="1 2 3 4 5"

mkdir -p "$OUT"

run () {  # arch, condition, seed
  echo "--- $1 $2 seed$3  $(date +%H:%M:%S)"
  python -u -I train.py --arch "$1" --images "$IMG/$2" --manifest "$MAN" \
    --seed "$3" --out "$OUT" 2>/dev/null
}

# block 1: the core result
for s in $SEEDS; do
  for c in mel wav_mel linear wav_linear; do
    run resnet18 "$c" "$s"
  done
done

# block 2: does the pattern hold for a second transferred architecture
for s in $SEEDS; do
  for c in mel wav_mel linear wav_linear; do
    run mobilenetv2 "$c" "$s"
  done
done

# block 3: the from-scratch baseline, now on matched inputs, which is what
# the original design could not separate from architecture and axis
for s in $SEEDS; do
  for c in mel wav_mel linear wav_linear; do
    run compact "$c" "$s"
  done
done

# block 4: the expensive one, three seeds
for s in 1 2 3; do
  for c in mel wav_mel; do
    run densenet201 "$c" "$s"
  done
done

echo "GRID COMPLETE $(date +%H:%M:%S)"
ls -1 "$OUT" | wc -l
