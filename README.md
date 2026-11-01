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

    code/      pipeline: ingest, denoising, spectrograms, splits, training, analysis
    figs/      figures used in the manuscript
    runs_coswara/   per recording decisions, one CSV per architecture, condition and seed
    *.tex      manuscript sources
    references.bib             bibliography, the single source of truth for citations
    references_provenance.md   the page each bibliography field was read from
    PAPER_SPEC.md              formatting and integrity rules the manuscript follows
    check_paper.py             structural checks on the manuscript

Corpora, trained weights and decoded image caches are not in the repository. All of them
are rebuilt by the ingest scripts from the original sources.

## Pipeline

    python -I coswara_ingest.py   --out ../data/coswara_img
    python -I make_splits.py      ../data/coswara_img
    python -I sweep.py            --images ../data/coswara_img --out ../runs_coswara
    python -I analyse.py          ../runs_coswara

`vocalsound_ingest.py` and `fsd50k_ingest.py` build the external validation sets, and
`coughvid_ingest.py` builds the classifier-labelled arm. `queue_runner.py` holds the
heavy jobs until the machine is free and splits them into a graphics card lane and a
processor lane.

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
