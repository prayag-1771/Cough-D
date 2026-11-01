# Reference provenance

Every entry in `references.bib` is listed here with the page its fields were read from.
An entry may not enter the bibliography until it appears in this file. A field that could
not be confirmed is omitted from the entry rather than guessed, and that omission is
recorded here.

This file is working material and is not part of the manuscript. No entry carries a
digital object identifier, by requirement.

## Corpora and prior cough work

| Key | Checked against | Confirmed | Omitted, and why |
|---|---|---|---|
| `coswara2020` | ISCA Archive `interspeech_2020/sharma20d_interspeech.html` | all 8 authors in printed order, title, proceedings, pages 4811–4815, year | — |
| `coughvid2021` | Nature `s41597-021-00937-4`; PMC8222356; EPFL Infoscience 286867 | authors, title, journal, volume 8, article 156, year | issue; records disagree between "8:156" and "8(1):1–10" |
| `fsd50k2022` | UPF repository record; Freesound FSD50K release page | all 5 authors, title, journal, volume 30, pages 829–852, year | — |
| `vocalsound2022` | arXiv 2205.03433; MIT CSAIL SLS publication record | authors Yuan Gong, Jin Yu, James Glass, title, venue, year | **pages**: not found on any source consulted |
| `xia2021` | NeurIPS Datasets and Benchmarks proceedings PDF; Southampton repository | all 12 authors in order, title, venue, year | pages; the track does not paginate |
| `han2022` | PMC8799654; Southampton ePrints `s41746-021-00553-x` | all 12 authors, title, journal, volume 5, article 16, year | — |
| `coppock2024` | arXiv 2212.08570 abstract page for the 25-author list; Nature Machine Intelligence record for volume and pages | all 25 authors in order, title, journal, volume 6, pages 229–242, year | — |
| `roberts2021` | Dundee Discovery `s42256-021-00307-0` PDF; journal header | first 8 authors, title, journal, volume 3, number 3, pages 199–217, year | authors beyond the eighth, rendered as "and others"; it is a consortium list |
| `pahar2021` | arXiv 2012.01926; British Library catalogue record | all 4 authors, title, journal, volume 135, article 104572, year | — |
| `laguarta2020` | PMC8545024; MIT DSpace 1721.1/128954 | all 3 authors, title, journal, volume 1, pages 275–281, year | — |
| `imran2020` | DOAJ record; MEDLINE affiliation listing | all 9 authors, title, journal, volume 20, article 100378, year | — |
| `zhou2021` | Frontiers in Robotics and AI article page, `frobt.2021.580080` | all 8 authors in order, title, journal, volume 8, article identifier, year | — |
| `zhang2026` | arXiv 2608.25846 abstract page | all 4 authors, title, submission date 26 August 2026 | venue; it is a preprint and is cited as one |

## Methods, architectures and tools

| Key | Checked against | Confirmed | Omitted, and why |
|---|---|---|---|
| `donoho1994` | Biometrika records, volume 81 issue 3 | authors, title, journal, volume, number, pages 425–455, year | — |
| `chang2000` | EPFL Infoscience 33854 | authors, title, journal, volume 9, number 9, pages 1532–1546, year | — |
| `johnstone2005` | Project Euclid `euclid.aos/1123250227`; arXiv math/0508281 | authors, title, journal, volume 33, number 4, pages 1700–1752, year | — |
| `mallat1989` | Author's copy at `di.ens.fr/~mallat`; dblp `journals/pami/Mallat89` | author, title, journal, volume 11, number 7, pages 674–693, year | — |
| `stevens1937` | JASA bibliographic records, volume 8 issue 3 | all 3 authors, title, journal, volume, number, pages 185–190, year | — |
| `gemmeke2017` | Google Research publication page | all 8 authors, title, venue, pages 776–780, year | — |
| `he2016` | CVF Open Access `He_Deep_Residual_Learning_CVPR_2016_paper.html` | all 4 authors in byline order, title, venue, pages 770–778, year | — |
| `sandler2018` | CVF Open Access `Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html` | all 5 authors, title, venue, pages 4510–4520, year | — |
| `huang2017` | IEEE Xplore metadata via library catalogue; Princeton research portal | all 4 authors, title, venue, pages 2261–2269, year | — |
| `russakovsky2015` | SpringerLink `s11263-015-0816-y`; Princeton research portal; MIT DSpace | first 4 authors, title, journal, volume 115, number 3, pages 211–252, year | authors beyond the fourth, rendered as "and others" |
| `kingma2015` | UvA repository record; ICLR 2015 proceedings mirror | both authors, title, venue, year | pages; ICLR conference track proceedings are not paginated |
| `paszke2019` | NeurIPS proceedings page `paper/9015-pytorch-an-imperative`; arXiv 1912.01703 | first 3 of 21 authors, title, venue, year | **pages**: only one aggregator gave 8026–8037 and it could not be corroborated, so the field is absent |
| `mcnemar1947` | SpringerLink `BF02295996`; RePEc `psycho:v:12:y:1947:i:2` | author, title, journal, volume 12, number 2, pages 153–157, year | — |
| `holm1979` | JSTOR record via bibliographic indexes | author, title, journal, volume 6, number 2, pages 65–70, year | — |
| `brown2001` | Project Euclid `euclid.ss/1009213286` PDF header | all 3 authors, title, journal, volume 16, number 2, pages 101–133, year | — |
| `wilson1927` | JSTOR listing; MacTutor biography of Wilson | author, title, journal, volume 22, number 158, pages 209–212, year | — |
| `matthews1975` | PubMed 1180967 | author, title, journal, volume 405, number 2, pages 442–451, year | — |
| `chicco2020` | BMC Genomics `s12864-019-6413-7`; PMC6941312 | both authors, title, journal, volume 21, article 6, year | — |
| `dietterich1998` | Author's copy at Oregon State; MIT Press record | author, title, journal, volume 10, number 7, pages 1895–1923, year | — |

## Corrections caught during verification

These are cases where the value that would have been written from memory was wrong.

- `huang2017`: DenseNet is **pages 2261–2269**, not 4700–4708. Three independent records agree.
- `chang2000`: one index lists the first author as "Shih-Fu Chang". The IEEE record and the
  EPFL repository both give **S. Grace Chang**.
- `vocalsound2022`: the first author is **Yuan** Gong, and the paper has three authors, not
  two.
- `paszke2019` and `kingma2015`: page ranges could not be corroborated and are absent
  rather than filled with a plausible value.

## Claims taken from these sources, and where each is used

- `coppock2024`: once confounders including self-reported symptoms are matched, the area
  under the curve falls from 0.846 to 0.619. Motivates treating label provenance and
  matched comparison as the central problem.
- `zhang2026`: cough models fall below 0.6 ROC-AUC on external data, and the audio
  representations cluster by recording device and dataset rather than by disease status.
  Supports the argument that channel effects, not pathology, can drive apparent accuracy.
  Two of its authors are also authors of `coughvid2021`.
- `zhou2021`: reports 98.18% on a random split against 95.18% on a split where test
  coughs come from recordings unseen in training, and does not discuss the gap. Used as
  the example of prior work that measured the split effect without interrogating it.
- `han2022`: an unbiased model reaches 0.71 area under the curve, while less careful
  evaluation reaches up to 0.90 on the same material.
- `vocalsound2022`: states its recordings are acted rather than spontaneous. Used to place
  VocalSound in the prompt-derived regime rather than treating it as independently
  annotated.
- `dietterich1998`: of five tests compared, McNemar's is the only one with acceptable Type
  I error for a learner that is trained once. Justifies testing paired per recording
  decisions instead of comparing accuracies between runs.
- `donoho1994` gives the universal threshold and soft-thresholding rule, `chang2000` the
  level-adaptive BayesShrink rule, and `johnstone2005` the empirical Bayes posterior
  median rule that is the default of the denoiser used in the work being extended.
- `fsd50k2022`: labels are human annotations and the evaluation split was exhaustively
  verified by the corpus authors. This is why it supplies the human-annotated arm.
- `brown2001` and `wilson1927` justify the Wilson score interval over the Wald interval.
  `matthews1975` and `chicco2020` justify reporting the correlation coefficient alongside
  accuracy on unbalanced classes.
