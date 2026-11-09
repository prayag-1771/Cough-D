# Cough detection: denoising, frequency axis, and label provenance

Code and manuscript sources for a study of the cough versus non-cough decision in
crowdsourced respiratory audio. The question is narrow on purpose: does wavelet
denoising applied before the spectrogram is formed change how well a convolutional
network separates cough from non-cough, does the choice of frequency axis change it, and
does either answer depend on how the labels were obtained in the first place.

## Why the design looks like this

Three properties of the usual setup make a reported accuracy hard to interpret, and each
is addressed by a specific part of the pipeline rather than argued away.

**The label has a provenance.** A cough label can come from what a participant was asked
to do, from a human listening to the recording, or from another classifier's output. The
three are not interchangeable, so three corpora are used and the regime is stated for
each: Coswara is prompt derived, FSD50K is human annotated, and COUGHVID carries a
`cough_detected` score produced by the corpus authors' own detector. A model measured on
the third is measured on its agreement with that detector, not on whether a cough is
present.

**A split by recording is not a split by speaker.** If one participant's recordings land
on both sides of a split, the test set is partly memorised. Coswara carries a participant
identifier, so the split here is speaker disjoint and is asserted to be so at build time.
COUGHVID carries only anonymous identifiers and cannot support such a split, which is
reported rather than hidden.

**A single run is not a measurement.** Differences of under one accuracy point are
routinely reported in this area without any estimate of run to run variation. Every
condition here is run across five seeds, the spread between them is reported beside every
effect, and denoising is compared on paired per recording decisions with McNemar's test
and a Holm correction rather than by comparing two accuracies.

Coswara additionally lets both classes come from the same participant, device and
session, so a difference between classes cannot be explained by channel or speaker.

## Layout

    code/                      ingest, denoising, spectrograms, splits, training, analysis
    manuscript/                manuscript sources, generated tables, figures, sn-jnl.cls
    runs_coswara/              per recording decisions from the prompt derived grid
    runs_fsd50k/               the same weights applied to human annotated audio
    runs_coughvid/             the classifier derived grid
    results/                   measurements the manuscript quotes that are not runs
    references.bib             bibliography, the single source of truth for citations
    references_provenance.md   the page each bibliography field was read from
    PAPER_SPEC.md              formatting and integrity rules the manuscript follows
    assemble.py                joins the sections into one .tex and checks it
    check_paper.py             structural checks on the manuscript

Corpora, trained weights and decoded image caches are not in the repository. All of them
are rebuilt by the ingest scripts from the original sources. One CSV per architecture,
condition and seed is kept, because those rows and not any summary of them are what the
tables and figures are computed from.

## Pipeline

    python -I coswara_ingest.py   --out ../data/coswara_img
    python -I make_splits.py      ../data/coswara_img
    python -I sweep.py            --images ../data/coswara_img --out ../runs_coswara
    python -I analyse.py          ../runs_coswara

`fsd50k_ingest.py` builds the external validation set and `coughvid_ingest.py` the
classifier labelled arm. `external_eval.py` applies saved weights to a corpus they were
never trained on. `finish.py` chains whatever remains and skips any stage whose output
already exists, which is what makes an interrupted run cheap to resume.

`vocalsound_ingest.py` is retained but does not work: every form of its distribution link
returned a page rather than the audio at the time of writing. FSD50K replaced it, and is
the better choice anyway, since its evaluation split was exhaustively verified rather
than sampled for verification.

## Building the manuscript

    python -I code/tables.py          # result tables, from the decision files
    python -I code/threshold.py       # discrimination against calibration
    python -I code/error_analysis.py  # what the false positives actually are
    python -I code/figures.py         # both figures, and the shrinkage audit
    python -I code/trace_numbers.py   # every quoted number back to a file
    python -I assemble.py --check     # one .tex, then the structural checks

`assemble.py` writes `manuscript/paper.tex`, which is the single file the publisher asks
for, expanding each generated table where its `\input` stands. The bibliography is
generated from `references.bib` in citation order, so a reference is written down in
exactly one place. `code/pagecount.py` estimates the built length, which is an estimate
and says so: there is no TeX installation on the machine this was produced on.

Every number in the manuscript comes from a file in `runs_*/` or `results/` by way of one
of these scripts, rather than from arithmetic done while writing. Regenerating is the
only way to change a table.

## What the study found

A ResNet18 reaches 98.6 per cent balanced accuracy on held out speakers within the
prompted corpus and 70.2 per cent on human annotated audio at the same class prior, with
no retraining in between. The loss falls almost entirely on sensitivity, which drops from
97.7 to 56.0 per cent. It is a loss of discrimination rather than of calibration, so
moving the decision threshold recovers little of it, and the false positives concentrate
on the other forced expirations: clips annotated as sneezes are called coughs more often
than coughs themselves are. Wavelet shrinkage changes none of this, and removes a median
of 0.67 per cent of signal power from recordings of this quality.

Each module's docstring states what it does and, where a choice was made that could have
gone another way, why it went the way it did. Two are worth reading before trusting any
number out of this repository: `denoise.py`, which states exactly which thresholding rule
is implemented and that it is not bit identical to the reference implementation it
replicates, and `coughvid_ingest.py`, which records that the selection used by the work
being extended is not reproducible from the published corpus.

## Reproducing a single run

A run loads one condition's images for the three partitions, trains one architecture from
one seed, selects the epoch with the best validation balanced accuracy without reference
to the test partition, and writes one row per test recording with the true label, the
predicted label and both class scores. `analyse.py` reads a directory of those files and
produces the condition summaries, the pooled intervals and the paired tests.
