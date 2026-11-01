# Writing a research paper in Springer Nature format

Reusable specification. Where a rule conflicts with a default, the rule wins.

## 1. Document class and preamble

Use the Springer Nature journal class. The project must contain `sn-jnl.cls` and the
matching `.bst`, both from the official Springer Nature LaTeX template package
(Overleaf's template gallery, or Springer Nature's LaTeX author support page). These are
not standard LaTeX files; without them the document will not compile.

```latex
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{listings}

\raggedbottom
```

Reference style is chosen by the class option: `sn-mathphys-num` for numbered math and
physical sciences, `sn-nature` for Nature Portfolio, `sn-basic` for general Springer,
`sn-vancouver-num`, `sn-apa`, `sn-chicago`. Never add `\bibliographystyle` — the class
sets it, and an explicit one silently overrides it.

### Four class-specific problems to pre-empt

These are not obvious and each produces a confusing symptom:

1. **Do not load `mathrsfs` unless you actually use `\mathscr`.** It appears in the
   template's default preamble, but merely declaring the alphabet makes LaTeX load the
   rsfs font at every math size in the document. If any table sets math inside `\tiny`,
   you get a stream of ``Font shape `U/rsfs/m/n' in size <4.015> not available''.

2. **Kill the alternating margins.** The class does `\LoadClass[twoside,fleqn]{article}`,
   so odd and even pages get mirrored margins for binding. On screen the text block
   visibly steps left and right. Passing `oneside` to `\documentclass` does not fix it,
   because the class applies its own `twoside` afterwards and the later declaration wins.
   Average the two margins instead:

   ```latex
   \AtBeginDocument{%
     \addtolength{\oddsidemargin}{\evensidemargin}%
     \setlength{\oddsidemargin}{0.5\oddsidemargin}%
     \setlength{\evensidemargin}{\oddsidemargin}%
   }
   ```

3. **Cap the bookmark depth.** `\bmhead` is defined as
   `\@startsection{subparagraph}{5}` while the class sets `bookmarksdepth=5`, so the back
   matter jumps from a level-1 bookmark straight to level 5 and hyperref reports
   ``Difference (4) between bookmark levels is greater than one'' once per heading. Add
   `\hypersetup{bookmarksdepth=3}`.

4. **Handle loose bibliography lines correctly.** DOIs and URLs cannot hyphenate, so TeX
   sets the surrounding lines loose and reports underfull boxes. `\sloppy` alone does not
   silence these — it permits the loose line but leaves `\hbadness` at 1000. You need
   both, scoped to the bibliography and applied at `\begin{document}` so it lands after
   natbib's own hooks:

   ```latex
   \AtBeginDocument{%
     \let\snOrigThebibliography\thebibliography
     \renewcommand{\thebibliography}[1]{%
       \snOrigThebibliography{#1}\sloppy\hbadness=10000\relax}%
   }
   ```

**Single file requirement.** Springer requires the submitted manuscript as one `.tex`
document. If the working copy uses `\input{}` for tables or the bibliography, expand
every one into a self-contained file before uploading.

## 2. Front matter

```latex
\title[SHORT RUNNING HEAD]{FULL TITLE}

\author[1]{\fnm{First} \sur{Author}}\email{...}
\author*[2]{\fnm{Corresponding} \sur{Author}}\email{...}
\author[2]{\fnm{Third} \sur{Author}}\email{...}

\affil[1]{\orgdiv{School Name (ABBREVIATION)}, \orgname{University},
  \orgaddress{\city{City}, \state{State}, \country{Country}}}
\affil*[2]{\orgdiv{...}, \orgname{...}, \orgaddress{...}}

\abstract{...}
\keywords{Term One, Term Two, Term Three, Term Four, Term Five, Term Six}
\maketitle
```

The star on `\author*` marks the corresponding author; the star on `\affil*` marks the
affiliation carrying the correspondence address. They need not be the first author. Name
every school or department consistently — if one carries its abbreviation in brackets,
all of them must.

**Title.** Name the method, the problem and the finding. A colon between "what we did"
and "what we found" works well. Keep it under about eighteen words. Avoid a verb stronger
than the evidence supports; if the limitations section concedes something cannot be
calibrated, do not claim it in the title. Whatever verb is chosen, use the same one in the
body — a title that says "predicts" over a body that says "indicates" is its own
inconsistency.

**Keywords.** Four to six terms. Do not repeat words already in the title — search
indexes the title anyway, so a duplicated keyword buys nothing. Choose the terms a reader
would search on that the title omits. Editors pick reviewers from keywords, so include
the sub-field the method belongs to.

**Abstract.** One unbroken paragraph. No citations, no display equations, no subheadings.
No dashes. No parenthetical statistics such as `(p = 0.002, d = 4.68)` — those belong in
the results table. State the problem, the method, the headline quantitative result, the
scope condition, and the limits, in that order.

## 3. Section structure

Keep the number of major sections small. If content does not need to be a section, fold
it inside one. A workable six-section shape:

1. Introduction
2. Related work and its limitations
3. System design
4. Experimental setup
5. Results and discussion
6. Conclusion

Rules that apply regardless of the shape chosen:

- Springer forbids subheadings in the Introduction. Write it as continuous prose.
- Put the contributions inside the Introduction as an enumerated list of four or five
  items, not as a section of their own. More than five and none of them registers.
- Every section that has subsections opens with a paragraph naming what follows and how
  many subsections come, so the reader knows the shape before the headings start.
- No heading may have exactly one child. A section with one subsection, or a subsection
  with one subsubsection, must have that child folded into the parent. Numbering
  something 3.1 implies a 3.2 that never arrives.
- Include a reproducibility subsection in the experimental setup: what a single run
  consists of end to end, the experimental grid stated in numbers, and a list of every
  secondary experiment with the question it settles and a pointer to where its result
  appears. Derive every count from saved result files, never by multiplying in your head.
- Do not include a Future Work section unless the target journal asks for one. Fold
  forward-looking statements into the limitations.

## 4. Writing style

**Abbreviations.** Expand every one at first use with the short form in brackets, then
use the short form alone. This includes ones that seem universal. Abbreviations appearing
only inside a figure must also be defined in the text, or spelled out in the figure — a
reader meeting "ROS" for the first time in a plot has nothing to expand it against.

**Dashes.** Remove every em dash and every compound hyphen from body text. Replace with a
colon where a colon works, otherwise a comma, parentheses, or a new sentence. Keep a dash
only where it belongs to a proper name (Holm–Bonferroni, Kolmogorov–Smirnov, dataset
names, hyphenated surnames) because removing those reads as a typo rather than as style.
Where dropping a hyphen would produce a non-word, use the closed form: subtype, subclass,
multiclass, nonconvex, precomputed, standalone.

**Paragraph length.** Merge adjacent short paragraphs that carry one continuous argument;
two of roughly seventy-five words each should become one of about a hundred and fifty. Do
not merge lead-ins that introduce a table, figure or list, and do not merge run-in bold
definitions such as a list of datasets or classifiers — those are meant to be short.

**Lists.** Use bullets or numbers wherever the text is already enumerating. "First…
Second… Third…" across three paragraphs is a list pretending not to be one.

**Tone.** Write "we propose", not "we introduce". Avoid "to our knowledge" and "the first
application of this" in the abstract; a novelty claim may appear once, hedged, where
prior work is reviewed. Frame the paper around what it achieves, with negative findings
placed in the limitations rather than the headline — but report them fully there, because
a paper claiming only successes reads as cherry-picked and a reported failure makes the
successes more credible, not less.

**Conclusion.** One paragraph.

## 5. References

Verify every reference against a publisher, proceedings or preprint page before adding
it. Read the author list, title, venue, volume, pages and year from the source. Never
recall them, never infer them, never fill a gap with something plausible. If a field
cannot be confirmed, omit that field rather than guess it. This is the single most common
failure mode in AI-assisted writing and the hardest for a human to catch.

Cite at least thirty references for a full research paper. Prefer recent work, within the
last three to four years, for anything that reviews the field, states a competing
approach, or motivates the gap. Older references are correct and necessary where they are
the primary source for a method, dataset, tool or statistical procedure actually used: a
paper that runs SMOTE must cite the paper that defined SMOTE, and no recent survey
substitutes for it.

Every reference must be cited, no reference may be left uncited, and each citation must
sit on the line whose specific claim it supports. A citation attaches to what it supports:
after the method named, after the author, or at sentence end if the whole sentence comes
from that source. Parking all citations at the end of a paragraph makes it impossible to
tell what is attributed to whom.

Always attach citations with a non-breaking space: `word~\cite{key}`. With a plain space
TeX may break the line there, stranding the bracket number at the start of the next line
away from what it cites.

If the journal's `.bst` does not produce the required format, write the reference list as
a `thebibliography` block instead, ordered by first citation. Generate it from the `.bib`
file with a script rather than typing it out — thirty-five entries retyped by hand is
thirty-five fresh chances to introduce an error, and it keeps the `.bib` as the single
source of truth.

## 6. Figures and tables

Every float must be referenced in the text, and the referring sentence must say what the
float shows and why to look at it. "Table 1 audits this" is not acceptable. Where a figure
and a table cover the same result, their captions must not restate the same finding: the
figure caption should describe the figure and say what it adds beyond the table.

Floats must be numbered in citation order. LaTeX numbers a float by where
`\begin{table}` appears, not by where `\ref` does. If Table B is declared before Table A
while the text cites A first, the text will discuss "Table 5" a page before Table 4
exists. Nothing warns you: every reference resolves and the numbers are internally
consistent. Only reading the built PDF in order reveals it.

Tables must fit inside the text block. A table that overruns can be brought in with
`\footnotesize`, tighter `\tabcolsep`, and shorter column headings — a heading three times
the width of any value beneath it is wasting the space.

## 7. Back matter

```latex
\backmatter

\bibliography{references}          % or \input{bibliography}

\bmhead{Supplementary information}
\bmhead{Acknowledgements}
\bmhead{Data availability}

\section*{Declarations}
\begin{itemize}
\item \textbf{Funding} \; ...
\item \textbf{Conflict of interest/Competing interests} \; ...
\item \textbf{Ethics approval} \; ...
\item \textbf{Consent to participate} \; ...
\item \textbf{Consent for publication} \; ...
\item \textbf{Availability of data and materials} \; ...
\item \textbf{Code availability} \; ...
\item \textbf{Authors' contributions} \; ...
\end{itemize}
```

`\backmatter` in this class is only a flag (`\global\backmattertrue`), so the order of
what follows it is yours to choose. Put the reference list first and the declarations
after it.

Do not invent declarations. Funding, competing interests, ethics approval and the authors'
contributions are statements of fact about real people that carry consequences after
publication. Ask the authors what each did and what to declare. If a template being
copied says something false about the work, for instance that no pre-existing datasets
were used when public benchmarks were analysed, keep the accurate wording and take only
the template's structure.

## 8. Integrity requirements, which override everything above

- No plagiarism. All prose original.
- No AI attribution anywhere. Nothing in the manuscript, repository, commit history or
  accompanying material may state or imply the work was produced by an AI. Commits carry
  only the author's own name: no co-author trailer, no tool attribution.
- Nothing asserted from memory. No number, author list, venue, page range, accuracy
  figure, or claim about a cited paper may be written without tracing it to a result file
  or a source page. If it cannot be verified, omit it. Never assert what a cited paper
  reports unless that part of it has been read.
- Every number in the paper should come from a file, not from a person. Where possible,
  generate tables and figures from saved result files by script so that a corrected
  result propagates automatically. Numbers retyped into prose drift from their source,
  and the drift survives review.

## 9. Verify before declaring done

- Does it compile? Warnings are not errors, but each one has a cause worth understanding.
- Is every citation key defined, and every reference cited?
- Are floats numbered in citation order?
- Does any section have exactly one subsection?
- Does every section with subsections open with a paragraph saying what follows?
- Are all abbreviations expanded at first use, including inside figures?
- Any em dashes or compound hyphens left in body text?
- Does the abstract contain dashes, citations or parenthetical statistics?
- Is the single-file version current with the working copy?
- Was every reference checked against its source, or were some written from memory?

Write a checking script rather than re-reading by eye. Structural properties (citation
coverage, float order, dangling references, orphaned floats, lone subsections, dash
violations) are mechanically checkable and will otherwise drift silently across edits.
But note what a script cannot check: whether a reference is real, whether a number
matches its source, and whether a sentence pointing at a table says anything useful.
Those need reading.

**One thing worth knowing before relying on this.** In the work these rules came from,
six invented author lists, an invented document number, an invented paper title and two
invented accuracy figures reached the manuscript. Every one was caught by a human opening
the actual papers, none by any automated check, because a checker validates structure,
not truth. Section 5 and Section 8 are the parts that matter most, and they are the parts
a language model is most likely to skip.

## Authors for this manuscript

```latex
\author[1]{\fnm{Prayag} \sur{Sharma}}\email{prayag.sharma2024@vitstudent.ac.in}

\author*[2]{\fnm{Ravi} \sur{Tiwari}}\email{ravi.tiwari@vit.ac.in}
```

Authors' contributions statement to use verbatim:

> Prayag Sharma conceived the study, designed and implemented the experiments, performed
> the statistical analysis, produced all figures and tables, and drafted the manuscript.
> Ravi Tiwari supervised the work and reviewed the manuscript. All authors read and
> approved the final version of the manuscript.
