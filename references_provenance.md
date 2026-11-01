# Reference provenance

Every entry in `references.bib` is listed here with the page its fields were read from.
An entry may not enter the bibliography until it appears in this file. A field that could
not be confirmed is omitted from the entry rather than guessed, and that omission is
recorded here.

This file is working material and is not part of the manuscript.

| Key | Checked against | Fields confirmed | Omitted, and why |
|---|---|---|---|
| `coswara2020` | ISCA Archive, `isca-archive.org/interspeech_2020/sharma20d_interspeech.html` | all 8 authors in printed order, title, proceedings, pages 4811–4815, year | — |
| `coughvid2021` | Nature `s41597-021-00937-4`; PMC8222356; EPFL Infoscience 286867 | authors, title, journal, volume 8, article number 156, year | issue; records disagree between "8:156" and "8(1):1–10" |
| `fsd50k2022` | UPF repository record; Freesound annotator FSD50K release page | all 5 authors, title, journal, volume 30, pages 829–852, year | — |
| `vocalsound2022` | arXiv 2205.03433; MIT CSAIL SLS publication record | authors Yuan Gong, Jin Yu, James Glass, title, venue ICASSP, year | **pages**: not found on any source consulted, so the field is absent |
| `coppock2024` | arXiv 2212.08570 abstract page for the 25-author list; Nature Machine Intelligence record for volume and pages | all 25 authors in order, title, journal, volume 6, pages 229–242, year | — |
| `donoho1994` | Biometrika bibliographic records, vol 81 iss 3 | authors, title, journal, volume 81, number 3, pages 425–455, year | — |
| `chang2000` | EPFL Infoscience 33854 | authors, title, journal, volume 9, number 9, pages 1532–1546, year | — |
| `johnstone2005` | Project Euclid `euclid.aos/1123250227`; arXiv math/0508281 | authors, title, journal, volume 33, number 4, pages 1700–1752, year | — |
| `dietterich1998` | Author's copy at Oregon State, `web.engr.oregonstate.edu/~tgd/publications/`; MIT Press record | author, title, journal, volume 10, number 7, pages 1895–1923, year | — |

## Corrections made during verification

- `chang2000`: one scholarly index lists the first author as "Shih-Fu Chang". The IEEE
  record and the EPFL repository both give **S. Grace Chang**, which is what the entry
  uses.
- No digital object identifiers appear in any entry, by requirement.

## Claims taken from these sources, and where each is used

- `coppock2024` reports that once confounders including self-reported symptoms are
  matched, the area under the curve falls from 0.846 to 0.619. Used to motivate why
  label provenance and matched comparison are treated as the central problem here.
- `vocalsound2022` states that its recordings are acted rather than spontaneous. Used to
  justify placing VocalSound in the prompt-derived label regime rather than treating it
  as an independent-annotation corpus.
- `dietterich1998` finds McNemar's test to be the only one of five with acceptable Type I
  error for learners that can be trained once. Used to justify testing paired per
  recording decisions rather than comparing accuracies between runs.
- `donoho1994` sets the universal threshold and the soft-thresholding rule. `chang2000`
  gives the level-adaptive BayesShrink rule. `johnstone2005` gives the empirical Bayes
  posterior-median rule that is the default of the denoiser used in the work being
  extended. All three are implemented or named in the denoising comparison.
