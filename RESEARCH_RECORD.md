# Cough Detection: Research Record

A complete account of the study, its numbers and where each number comes from.
Everything below is reproduced from a file in `results/` or `manuscript/`; nothing
is quoted from memory. Where a figure is derived rather than stored, the
derivation is stated.

Last updated 13 November 2026.

---

## 1. What the study asks

Reported accuracies for cough detection sit in the high nineties, yet such models
are rarely validated outside the corpus they were developed on. Three factors
ordinarily move together in these studies and so cannot be told apart:

- **label provenance** — who or what decided a recording contains a cough,
- **negative class difficulty** — what the model must reject,
- **seed variation** — how much a result moves between identical runs.

This study holds the preprocessing, the architecture and the training protocol
fixed and varies **only the provenance of the label**, across three corpora whose
class priors are matched by construction.

It also tests the specific claim of the work it extends: that wavelet denoising
applied before spectrogram extraction improves detection.

### Two manuscripts come out of this

| | File | Claim |
|---|---|---|
| **New study** | `manuscript/paper_ieee.tex` (IEEE Access), `manuscript/paper.tex` (Springer) | Accuracy is set by label provenance; denoising does not help |
| **Conference conversion** | `overleaf/springer/sn-article_old.tex` | The original conference work in journal form, plus three side findings |

The two reach **opposite conclusions on denoising**. The conversion reports the
conference result as published; the new study refutes it under a controlled
protocol. They should not be submitted as if they agreed.

---

## 2. Corpora

Source: `results/corpora.csv`, `manuscript/04_setup.tex`.

| Corpus | Label from | Positive class | Negative class | Recordings | Prior | Speaker ids |
|---|---|---|---|---|---|---|
| **Coswara** | a prompt | the two cough tasks, heavy and shallow | breathing, sustained vowel, counting | 13,314 | 0.2228 | yes |
| **COUGHVID v3** | a detector | `cough_detected` ≥ 0.9 | `cough_detected` ≤ 0.1 | 8,398 | 0.2239 | no |
| **FSD50K eval** | a person | the `Cough` label | breathing, sneeze, gasp, sigh, laughter, speech | 476 | 0.2227 | no |

Partitions (`results/corpora.csv`):

| Corpus | Participants | Train | Train cough | Val | Val cough | Test | Test cough |
|---|---|---|---|---|---|---|---|
| Coswara | 1,495 | 9,301 | 2,074 | 1,998 | 445 | 2,015 | 448 |
| COUGHVID | — | 5,879 | 1,316 | 1,260 | 282 | 1,259 | 282 |
| FSD50K | — | 0 | 0 | 0 | 0 | 476 | 106 |

Coswara partitions are **speaker disjoint** — no participant appears on both sides
of a split. FSD50K is never trained on; it is the external set only.

---

## 3. Pipeline

Three choices are varied; everything else is held fixed.

### 3.1 Wavelet denoising

Level `J = 6`, `sym4` wavelet, discrete wavelet transform by Mallat's pyramidal
algorithm. Shrinkage applied to detail coefficients only. Three rules are
distinguished throughout, because the reported effect of denoising depends
entirely on which is used:

- **Universal threshold** (Donoho & Johnstone), soft thresholding:
  `λ = σ̂ √(2 ln n)`, with `σ̂` from the median absolute deviation of the finest scale.
- **Level adaptive** (Chang, Yu, Vetterli): `λ_j = σ̂² / σ̂_{x,j}` per scale.
- **Empirical Bayes** (Johnstone & Silverman): mixture prior of a point mass at
  zero and a heavy tailed density, mixing weight by marginal maximum likelihood,
  coefficient replaced by its posterior median.

**This distinction is the crux of the replication.** The work being extended calls
a denoising routine without naming a rule, and that routine's default is the
empirical Bayes procedure with a Cauchy prior and posterior median — *not* the soft
thresholding usually implied. This study therefore implements the rules directly
and states which is in use.

**How much each rule actually removes** (`results/shrinkage_summary.csv`,
259 recordings):

| Rule | Median % of signal power removed | Q1 | Q3 | Max |
|---|---|---|---|---|
| Universal threshold | **0.6709** | 0.0816 | 2.2759 | 49.1273 |
| Empirical Bayes | **0.0014** | 0.0000 | 0.0246 | 6.3976 |

This bounds what the comparison can show: the two input conditions are very nearly
the same images.

### 3.2 Spectrogram and frequency axis

Resample to 16 kHz; STFT with a 512-point Hamming window, hop 256 → **257 bins**.
Magnitudes to decibels.

- **linear** — the 257 bins as they are.
- **mel** — mapped to **128 bands**, `m(f) = 2595 log₁₀(1 + f/700)`.

Both resized to 224 × 224 and replicated to three channels. Decibel-to-pixel
conversion is per recording, so absolute loudness is discarded.

**Filterbank occupancy** (computed from `code/spectro.py` at these settings): of the
128 triangular filters, **1 receives no frequency bin at all** and is identically
zero, **25 are fed by exactly one bin**, and **53 are fed by two or fewer**. The
first filter fed by more than one bin is the 13th, centre 202 Hz; the median filter
is fed by 3 bins and the widest by 11. Bin spacing is 31.2 Hz. At `n_fft` 1024 the
empty filter disappears; 2048 is the shortest length at which every filter receives
at least three bins.

### 3.3 Architectures

| Network | Initialisation | Role |
|---|---|---|
| ResNet18 | ImageNet | residual connections at modest depth |
| MobileNetV2 | ImageNet | inverted residuals, ~10× fewer multiply-accumulates |
| DenseNet201 | ImageNet | far deeper, concatenates features across layers |
| Compact (4-layer) | random | isolates pretraining from input |

The compact network exists to separate what the prior work could not: its baseline
differed in architecture, pretraining, frequency axis **and** sample rate at once,
so no single cause could be assigned.

### 3.4 Training protocol

Identical for every run (`manuscript/04_setup.tex`):

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 1e-4 |
| Gradient clipping | unit norm |
| Epochs | at most 8 |
| Early stopping | no gain in validation balanced accuracy for 3 epochs |
| Weights retained | best validation balanced accuracy epoch |
| Test reads per run | one, after selection |
| Precision | single |
| Input | 224 × 224, ImageNet channel statistics |
| Batch size | 64 ResNet18, 16 MobileNetV2, 128 compact, 8 DenseNet201 |
| Seeds | five; three for DenseNet201 |

Batch size varies for a hardware reason, not a modelling one — see §7.3.

---

## 4. Experiment grid

| Grid | Runs | Composition |
|---|---|---|
| Coswara (prompted) | **66** | ResNet18, MobileNetV2, compact × 4 conditions × 5 seeds = 60; DenseNet201 × 2 mel conditions × 3 seeds = 6 |
| COUGHVID (classifier labelled) | **20** | ResNet18 × 4 conditions × 5 seeds |
| FSD50K (external) | **66** | every retained Coswara weight set, inference only |

The four conditions are the two frequency axes crossed with denoising applied or
withheld: `linear`, `wav_linear`, `mel`, `wav_mel`.

Every run writes one row per test recording carrying the file identifier, true
label, predicted label and both class scores. **Those rows, not any summary, are
what the analysis reads** — which is what makes the paired tests possible.

---

## 5. Results

### 5.1 Within the prompted corpus

Source: `results/summary.csv`.

| Architecture | Condition | Seeds | Accuracy | Sensitivity | Specificity | Balanced | MCC |
|---|---|---|---|---|---|---|---|
| ResNet18 | linear | 5 | 99.09 ± 0.13 | 97.72 ± 0.62 | 99.48 ± 0.26 | **98.60 ± 0.22** | 0.974 ± 0.004 |
| ResNet18 | wav+linear | 5 | 98.80 ± 0.24 | 97.14 ± 0.53 | 99.27 ± 0.30 | 98.21 ± 0.28 | 0.965 ± 0.007 |
| ResNet18 | mel | 5 | 98.71 ± 0.20 | 97.19 ± 0.81 | 99.14 ± 0.10 | 98.17 ± 0.41 | 0.963 ± 0.006 |
| ResNet18 | wav+mel | 5 | 98.65 ± 0.49 | 97.37 ± 0.53 | 99.02 ± 0.74 | 98.19 ± 0.25 | 0.961 ± 0.014 |
| MobileNetV2 | linear | 5 | 99.10 ± 0.12 | 97.68 ± 0.97 | 99.50 ± 0.28 | 98.59 ± 0.37 | 0.974 ± 0.003 |
| MobileNetV2 | wav+linear | 5 | 98.89 ± 0.16 | 97.01 ± 1.08 | 99.43 ± 0.42 | 98.22 ± 0.36 | 0.968 ± 0.004 |
| MobileNetV2 | mel | 5 | 98.86 ± 0.33 | 98.35 ± 0.34 | 99.00 ± 0.49 | **98.68 ± 0.17** | 0.967 ± 0.009 |
| MobileNetV2 | wav+mel | 5 | 98.91 ± 0.23 | 97.50 ± 0.91 | 99.31 ± 0.30 | 98.41 ± 0.41 | 0.969 ± 0.007 |
| DenseNet201 | mel | 3 | 99.07 ± 0.12 | 97.92 ± 0.68 | 99.40 ± 0.13 | **98.66 ± 0.31** | 0.973 ± 0.004 |
| DenseNet201 | wav+mel | 3 | 98.73 ± 0.50 | 97.54 ± 0.59 | 99.06 ± 0.78 | 98.30 ± 0.20 | 0.964 ± 0.014 |
| Compact | linear | 5 | 96.43 ± 0.38 | 93.35 ± 1.60 | 97.31 ± 0.74 | 95.33 ± 0.57 | 0.898 ± 0.010 |
| Compact | wav+linear | 5 | 95.62 ± 0.30 | 92.37 ± 1.71 | 96.55 ± 0.47 | 94.46 ± 0.72 | 0.876 ± 0.009 |
| Compact | mel | 5 | 96.46 ± 0.34 | 92.46 ± 2.24 | 97.60 ± 0.83 | 95.03 ± 0.79 | 0.898 ± 0.009 |
| Compact | wav+mel | 5 | 92.57 ± 8.29 | **73.93 ± 41.40** | 97.89 ± 1.55 | 85.91 ± 20.09 | 0.715 ± 0.400 |

**Two things to read here.**

*Capacity buys nothing.* DenseNet201 (98.66 ± 0.31 balanced, mel) is indistinguishable
from MobileNetV2 (98.68 ± 0.17), and its half-point margin over ResNet18 (98.17 ± 0.41)
is inside the bound this work sets for a claimable effect. The separation that does
appear is the **3 to 4 points** between each pretrained network and the one trained
from random initialisation. It is pretrained initialisation, not capacity, that
distinguishes the architectures.

*One row is a failure, not a measurement.* Compact on wav+mel averages 73.93 ± 41.40
sensitivity. A standard deviation that size on a bounded quantity is not a spread
about a mean — one seed converged to the majority class and stayed there. The
trainer reports any optimiser step abandoned for a non-finite gradient and none
were, so this is an optimisation outcome. It happened under that one condition and
to no other, and only to the network with the least capacity.

### 5.2 Does wavelet shrinkage help?

Source: `results/paired.csv`. McNemar's test on paired per-recording decisions;
`b` counts recordings the unprocessed condition gets right and the shrunk one does
not, `c` the reverse. Holm step-down correction, family per corpus.

| Corpus | Architecture | Comparison | Seeds | b | c | p | p adj. | Accuracy change |
|---|---|---|---|---|---|---|---|---|
| coswara | ResNet18 | linear → wav+linear | 5 | 66 | 37 | 0.0055 | **0.0277** | −0.29 ± 0.19 |
| coswara | ResNet18 | mel → wav+mel | 5 | 71 | 65 | 0.668 | 1.000 | −0.06 ± 0.39 |
| coswara | MobileNetV2 | linear → wav+linear | 5 | 71 | 50 | 0.0686 | 0.206 | −0.21 ± 0.24 |
| coswara | MobileNetV2 | mel → wav+mel | 5 | 53 | 58 | 0.704 | 1.000 | +0.05 ± 0.40 |
| coswara | DenseNet201 | mel → wav+mel | 3 | 50 | 29 | **0.0238** | 0.0953 | −0.35 ± 0.52 |
| coswara | Compact | linear → wav+linear | 5 | 196 | 115 | 5.1e−06 | **3.1e−05** | −0.80 ± 0.65 |
| coswara | Compact | mel → wav+mel | 5 | 525 | 133 | 5.9e−56 | **4.2e−55** | −3.89 ± 8.25 |
| coughvid | ResNet18 | linear → wav+linear | 5 | 79 | 36 | 7.5e−05 | **1.5e−04** | −0.68 ± 0.38 |
| coughvid | ResNet18 | mel → wav+mel | 5 | 51 | 53 | 0.922 | 0.922 | +0.03 ± 0.51 |

**Nine comparisons. Zero detectable improvements. Four reach significance and all
four are losses.** Two move a mean upward — by 0.05 and 0.03 points — and neither
approaches significance.

The largest improvement found anywhere is **+2.73 accuracy points** (MobileNetV2,
mel → wav+mel, on the external set) against seed spreads of **3.98 and 3.00** — the
effect is smaller than the variation it would have to clear.

**DenseNet201 is the case that shows what the correction is for.** Its single test
gives p = 0.0238, which any convention ignoring the family would report as an
effect, and 0.0953 after the step down, which would not. Run alone, those exact
decisions would have been published as a significant result about shrinkage.

### 5.3 Transfer to human annotated audio

Same saved weights, no retraining, matched prior. Source: `results/summary.csv`.

| Architecture | Condition | Accuracy | Sensitivity | Specificity | Balanced | MCC |
|---|---|---|---|---|---|---|
| ResNet18 | linear | 78.03 ± 1.62 | **56.04 ± 4.04** | 84.32 ± 3.24 | **70.18 ± 0.44** | 0.391 ± 0.012 |
| ResNet18 | wav+linear | 79.33 ± 1.19 | 53.21 ± 1.27 | 86.81 ± 1.65 | 70.01 ± 0.81 | 0.402 ± 0.022 |
| ResNet18 | mel | 78.70 ± 2.31 | 57.55 ± 3.89 | 84.76 ± 3.88 | 71.15 ± 1.01 | 0.411 ± 0.031 |
| ResNet18 | wav+mel | 77.56 ± 3.18 | 59.62 ± 2.78 | 82.70 ± 4.34 | 71.16 ± 2.08 | 0.401 ± 0.056 |
| MobileNetV2 | linear | 78.15 ± 2.28 | 60.57 ± 5.48 | 83.19 ± 4.26 | 71.88 ± 1.25 | 0.415 ± 0.024 |
| MobileNetV2 | mel | 74.92 ± 3.98 | 66.04 ± 4.11 | 77.46 ± 5.97 | 71.75 ± 1.93 | 0.392 ± 0.054 |
| DenseNet201 | mel | 81.09 ± 1.31 | 54.40 ± 2.37 | 88.74 ± 2.03 | 71.57 ± 0.96 | 0.443 ± 0.025 |
| Compact | linear | 68.11 ± 1.55 | 45.66 ± 4.88 | 74.54 ± 3.26 | **60.10 ± 1.06** | 0.184 ± 0.015 |

(Abbreviated; the full 14 rows are in `results/summary.csv`.)

**The headline: 98.60 → 70.18 balanced accuracy for identical weights.**

The fall is **asymmetric** — specificity 99.48 → 84.32, sensitivity **97.72 → 56.04**.
The detector does not stop working so much as stop finding coughs, and a single
accuracy figure hides which of the two has happened.

**It is discrimination, not calibration** (`manuscript/tab_threshold.tex`):

| Architecture | Condition | AUC in corpus | AUC external | Balanced at inherited threshold | Best attainable |
|---|---|---|---|---|---|
| ResNet18 | linear | 0.998 ± 0.002 | **0.749 ± 0.018** | 70.18 ± 0.44 | 71.90 ± 0.91 |
| ResNet18 | mel | 0.999 ± 0.000 | 0.768 ± 0.016 | 71.15 ± 1.01 | 72.34 ± 1.68 |
| MobileNetV2 | linear | 0.998 ± 0.000 | 0.768 ± 0.007 | 71.88 ± 1.25 | 73.30 ± 0.73 |
| DenseNet201 | mel | 0.998 ± 0.001 | 0.767 ± 0.022 | 71.57 ± 0.96 | 72.96 ± 0.93 |
| Compact | linear | 0.992 ± 0.001 | 0.648 ± 0.012 | 60.10 ± 1.06 | 62.23 ± 1.59 |

The best threshold obtainable **on the external set itself** recovers only about
**1.7 points**, and even that is optimistic because the threshold was chosen on the
same recordings it is scored on. Recalibration — the cheapest remedy — does not work.

*Which separation survives:* the three pretrained networks sit inside one another's
spread both inside the corpus and outside it (71.15 ± 1.01, 71.75 ± 1.93,
71.57 ± 0.96 on mel). The compact network, roughly three points behind ResNet18 in
corpus, is **roughly ten points behind out of corpus**.

### 5.4 What the detectors confuse

Source: `manuscript/tab_confusion.tex`, pooled over every ResNet18 run on the
external set. Rows overlap, since a clip may carry several labels.

| Annotated class | Decisions | Called cough | Per cent |
|---|---|---|---|
| **Sneeze** | 360 | 244 | **67.8** |
| **Cough (as recall)** | 2120 | 1200 | **56.6** |
| Burping | 440 | 149 | 33.9 |
| Whispering | 520 | 139 | 26.7 |
| Gasp | 200 | 41 | 20.5 |
| Sigh | 280 | 56 | 20.0 |
| Breathing | 800 | 132 | 16.5 |
| Yell | 280 | 40 | 14.3 |
| Male speech | 2320 | 316 | 13.6 |
| Speech | 4180 | 492 | 11.8 |
| Laughter | 1400 | 161 | 11.5 |
| Female speech | 1460 | 143 | 9.8 |
| Child speech | 440 | 34 | 7.7 |
| Chewing | 360 | 22 | 6.1 |

**Sneezes are assigned to the cough class more often than coughs are.** The prompted
corpus holds no forced expiration other than coughing, so that distinction was never
one the networks were required to learn. The composition of the negative class, not
the capacity of the architecture, set the figure the prompted corpus reports.

**This is the most novel result in the study.** The literature reports that these
models fail to generalise; this says *what they confuse and why it was structurally
guaranteed*.

### 5.5 The classifier derived regime

ResNet18 on COUGHVID reaches **98.41 ± 0.24** balanced accuracy (linear). What is
measured there is agreement with the detector that produced the labels, not whether
a cough is present.

**Two limits on what that corpus can establish** (`results/coughvid_audit.txt`):

1. **The selection is not reproducible.** Of 34,434 scored recordings, the cough
   class of the work being extended contains 15,619. **No threshold on
   `cough_detected` yields 15,619.** The count steps from 15,623 at a score of
   0.8961 directly to 15,617, because six recordings carry 0.8961 exactly and enter
   or leave together. The target lies strictly between two consecutive attainable
   counts.
2. **The expert subset is not a random sample.** 2,890 recordings (8.4%) carry a
   judgement from at least one of the four annotators. Their detector scores average
   **0.8954** against **0.6179** for the remainder; **85.8%** score at or above 0.8
   against **50.9%** of the unannotated. Agreement measured there cannot be
   extrapolated to the recordings nearest the threshold, which are exactly the ones
   whose labels are least secure.

---

## 6. The headline, in one place

One pipeline, one architecture, one input condition, three label regimes:

| Where the label came from | Corpus | Balanced accuracy |
|---|---|---|
| A participant was **told to cough** | Coswara | **98.60 ± 0.22** |
| **Another classifier's** score | COUGHVID | **98.41 ± 0.24** |
| A **human listened** and judged | FSD50K | **70.18 ± 0.44** |

All three would be reported as "cough detection accuracy". The first two are
indistinguishable at the precision anyone reports; the third sits almost **thirty
points** below either.

Set against that, every modelling choice measured here is small:

| Factor | Effect |
|---|---|
| Label provenance | **~28 points** |
| Architecture (on fixed input, linear) | ~3 points (95.33 vs 98.60) |
| Frequency axis (architecture fixed) | under half a point, direction inconsistent |
| Wavelet shrinkage | 0 of 9 improvements; 4 significant, all losses |

---

## 7. Side findings

Three results that bear on how such a study should be built rather than on its
outcome. All three were added to the conference conversion.

### 7.1 The filterbank is oversampled relative to its transform

See §3.2. One of 128 filters receives no bin, 25 receive exactly one, 53 receive two
or fewer. **Below roughly 200 Hz the mel axis expands a region in which the transform
holds no detail to expand.** Whatever the perceptual axis contributes, it is not
additional low-frequency resolution — it is the compression of the band above 4 kHz.

### 7.2 A selection that cannot be reproduced

See §5.5 item 1. A study deriving labels from a score should report **the threshold
and the metadata version**, not the resulting class sizes, which do not identify the
selection even when the corpus is public.

### 7.3 A memory condition that costs throughput silently

Source: `results/throughput.csv`. The card holds 4096 MiB.

| Architecture | Batch | Layout | Images/s | Peak MiB |
|---|---|---|---|---|
| ResNet18 | 64 | contiguous | 208.1 | 2898 |
| MobileNetV2 | 16 | contiguous | **147.6** | 2228 |
| MobileNetV2 | 32 | contiguous | 148.4 | 3222 |
| MobileNetV2 | 64 | contiguous | **23.2** | **5450** |
| MobileNetV2 | 16 | channels_last | 14.8 | 2238 |
| MobileNetV2 | 32 | channels_last | 13.2 | 3254 |
| DenseNet201 | 8 | contiguous | 29.8 | 2284 |
| DenseNet201 | 12 | contiguous | 33.6 | 3070 |
| DenseNet201 | 32 | contiguous | **1.6** | **7372** |
| Compact | 64 | contiguous | 912.6 | 1608 |
| Compact | 128 | contiguous | 967.1 | 3098 |

When activations exceed the card, **the allocation does not fail on this platform** —
the excess is served from host memory and the only indication is the rate. MobileNetV2
at batch 64 reserves 5450 MiB and runs at 23.2 images/s; at batch 16 it reserves
2228 MiB and runs at 147.6, **a factor of 6.4** between two runs of one network on one
corpus. DenseNet201 shows it more sharply: 1.6 against 29.8 images/s.

The competing explanation — that depthwise separable convolutions were hitting an
inefficient kernel — was tested by repeating every measurement in channels-last
layout. That layout leaves the peak essentially unchanged (2238 and 3254 MiB against
2228 and 3222) and runs an order of magnitude slower again, so it is not addressing a
memory problem.

The behaviour depends on the driver and operating system; elsewhere the same
configuration may fail outright with an allocation error.

**Practical consequence:** choose a batch size that leaves part of the card unused,
not the largest value that appears to fit.

---

## 8. Statistical method

- **Balanced accuracy and MCC** rather than accuracy alone — both weight the
  confusion matrix in proportion to class sizes, which accuracy and F1 do not.
- **Wilson score intervals**, not Wald, whose coverage is erratic near proportions
  of one — the region every condition here occupies.
- **McNemar's exact test** on paired per-recording decisions, exact while the
  discordant total is small enough to enumerate and continuity-corrected otherwise.
  Of five tests compared for this purpose by Dietterich, McNemar's was the only one
  with acceptable Type I error for a learner trained once.
- **Holm step-down correction** over the family of comparisons, per corpus.
- **Five seeds per condition** (three for DenseNet201), with the spread reported
  beside every effect. *An effect smaller than roughly twice that spread cannot be
  claimed from single runs, however large the test set is.* The DenseNet201 spread
  rests on three runs and is accordingly coarser.

---

## 9. Reproducing this

```
code/coswara_ingest.py      build the prompted corpus and its speaker-disjoint splits
code/coughvid_ingest.py     build the classifier-labelled corpus
code/fsd50k_ingest.py       build the external evaluation set
code/spectro.py             STFT, mel filterbank (built here, not imported)
code/denoise.py             the three shrinkage rules
code/sweep.py               the training grid
code/external_eval.py       saved weights against the external set
code/tables.py              summary, corpora, paired tables
code/threshold.py           discrimination against calibration
code/error_analysis.py      false positives by annotated class
code/figures.py             all five figures
code/trace_numbers.py       every number in the prose must occur in a generated file
code/check_tex.py           token-level LaTeX damage
check_paper.py              structure, citations, floats, cleanliness
assemble.py --venue {ieee,springer} --check
code/overleaf_bundle.py     the upload folder per target
```

Pipeline after a grid finishes: `external_eval → tables → threshold →
error_analysis → figures → trace_numbers → assemble (both venues)`.

---

## 10. Limitations

Stated plainly, because they bound what the study supports.

1. **The external set is small and single-source.** 476 recordings, 106 coughs. The
   fall is better established in direction than in size.
2. **The three regimes are three different corpora.** Microphones, speakers,
   annotators and the definition of the negative class change together with label
   provenance. The attribution to provenance is argued, not isolated. *Closing this
   needs new annotation — the same audio labelled two ways.*
3. **DenseNet201 is a reduced cell** — two conditions, three seeds — so its spread is
   a coarser estimate than the others'.
4. **One corpus family for training.** Coswara and COUGHVID only.
5. **The paper has never been compiled by its authors on this machine** — there is no
   TeX installation here. Page counts are estimates from word counts and float sizes.

---

## 11. Positioning

What the literature already establishes, and what this adds.

**Already known.** Cough models do not generalise across datasets — Zhang et al. 2026
report external AUC frequently below 0.6 with representations organising by recording
device rather than condition. Audio classifiers do not beat symptom checkers
(Coppock et al. 2024, AUC 0.846 unadjusted falling to 0.619 after matching on
confounders, 67,842 participants). ML COVID studies are riddled with methodological
pitfalls (Roberts et al. 2021, 2,212 studies screened, 62 included, none clinically
usable). Zimmer et al. 2026 externally validated eleven cough-based TB models:
**0.689–0.743 AUC on the challenge data falling to 0.480–0.615 on an independent
cohort.**

**What this study adds.**

1. The **attribution** — what the detectors confuse (sneezes at 67.8%, above coughs
   at 56.6%) and the structural reason the prompted negative class made it inevitable.
2. The **three-regime decomposition** under one fixed pipeline, including the
   classifier-labelled regime, which scores like the prompted one.
3. That the loss is **discrimination, not calibration**, closing off recalibration as
   a remedy — at most 1.7 points recoverable.
4. A denoising comparison **under a named shrinkage rule** with a protocol able to
   detect an effect of the size being claimed.
5. The three side findings of §7.

**Honest assessment.** This is a measurement and replication contribution, not a
method contribution. It is above the median published paper in this subfield for
rigour — five seeds, paired tests, multiple-comparison correction, external
validation, per-recording decisions retained — and below it for novelty of method,
since it proposes nothing new. Limitation 2 is the objection a careful reviewer will
raise first.
