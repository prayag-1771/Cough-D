"""Assemble the three parts into the single file Springer requires, then run
the structural checks. A checker validates structure, not truth, so this
covers citation coverage, float order, lone subsections, dashes and abstract
rules only; the reference metadata was checked separately against Crossref."""
import io
import re

PARTS = ['part1.tex', 'part2.tex', 'part3.tex']
OUT = 'Cough_Detection_Wavelet_Denoising_SpringerNature.tex'

src = ''.join(io.open(p, encoding='utf-8').read().rstrip() + '\n\n' for p in PARTS)

# Drop whole line comments so nothing in the delivered file reads as a note.
# Trailing '%' is left alone: in the \AtBeginDocument blocks it suppresses a
# spurious space and removing it would change the output.
kept = [ln for ln in src.split('\n') if not ln.lstrip().startswith('%')]
src = '\n'.join(kept)
src = re.sub(r'\n{3,}', '\n\n', src)

# part2 draws the pipeline with TikZ, so the preamble must load it
src = src.replace(
    '\\usepackage{listings}\n',
    '\\usepackage{listings}\n\\usepackage{tikz}\n'
    '\\usetikzlibrary{arrows.meta,positioning}\n')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write(src)
print('written %s (%d chars)' % (OUT, len(src)))

body = src[src.index('\\maketitle'):]
fails = []

# --- citations -----------------------------------------------------------
defined = set(re.findall(r'\\bibitem\{([^}]+)\}', src))
cited = set()
for grp in re.findall(r'\\cite\{([^}]+)\}', src):
    cited.update(k.strip() for k in grp.split(','))
print('\nreferences defined : %d' % len(defined))
print('reference keys cited: %d' % len(cited))
if cited - defined:
    fails.append('cited but undefined: %s' % sorted(cited - defined))
if defined - cited:
    fails.append('defined but never cited: %s' % sorted(defined - cited))
if len(defined) < 30:
    fails.append('fewer than 30 references (%d)' % len(defined))

# --- float order ---------------------------------------------------------
labels = re.findall(r'\\label\{((?:tab|fig):[^}]+)\}', src)
first_ref = {}
for m in re.finditer(r'\\ref\{((?:tab|fig):[^}]+)\}', body):
    first_ref.setdefault(m.group(1), m.start())
for kind in ('tab', 'fig'):
    declared = [l for l in labels if l.startswith(kind)]
    order = [l for l in sorted(first_ref, key=first_ref.get) if l.startswith(kind)]
    if declared != order:
        fails.append('%s declared %s but first cited %s' % (kind, declared, order))
    missing = [l for l in declared if l not in first_ref]
    if missing:
        fails.append('%s never referenced in text: %s' % (kind, missing))
print('tables: %d, figures: %d, all referenced in declaration order: %s'
      % (len([l for l in labels if l.startswith('tab')]),
         len([l for l in labels if l.startswith('fig')]),
         not any('declared' in f or 'never referenced' in f for f in fails)))

# --- lone subsections ----------------------------------------------------
heads = [(m.start(), m.group(1), m.group(2))
         for m in re.finditer(r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}', body)]
rank = {'section': 0, 'subsection': 1, 'subsubsection': 2}
for i, (_, lvl, name) in enumerate(heads):
    kids = 0
    for _, l2, _ in heads[i + 1:]:
        if rank[l2] <= rank[lvl]:
            break
        if rank[l2] == rank[lvl] + 1:
            kids += 1
    if kids == 1:
        fails.append('"%s" has exactly one child' % name)

# --- dashes in body ------------------------------------------------------
stripped = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', body, flags=re.S)
stripped = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', stripped, flags=re.S)
stripped = re.sub(r'\$[^$]*\$', '', stripped)
# \label and \ref arguments are cross-reference keys, not prose, and their
# internal hyphens are not compound hyphens in body text
stripped = re.sub(r'\\(label|ref|eqref|cite)\{[^}]*\}', '', stripped)
if '---' in stripped or '\u2014' in stripped:
    fails.append('em dash present in body text')
hyph = set(re.findall(r'(?<![\\{A-Za-z0-9])([a-z]{3,}-[a-z]{3,})(?![}])', stripped))
if hyph:
    fails.append('compound hyphens in body: %s' % sorted(hyph))

# --- abstract ------------------------------------------------------------
abst = re.search(r'\\abstract\{(.*?)\n\n', src, re.S)
if abst:
    a = abst.group(1)
    if '\u2014' in a or '---' in a or re.search(r'(?<![\\A-Za-z])[a-z]+-[a-z]+', a):
        fails.append('dash or hyphen in abstract')
    if '\\cite' in a:
        fails.append('citation in abstract')
    if re.search(r'\([^)]*[0-9]\.[0-9]+[^)]*\)', a):
        fails.append('parenthetical statistic in abstract')

# --- size ----------------------------------------------------------------
words = len(re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', stripped).split())
print('body words (excluding bibliography and tabulars): %d' % words)

# --- nothing in the file may read as a note or instruction ----------------
NOTE_PAT = [
    r'\bTODO\b', r'\bFIXME\b', r'\bTBD\b', r'\bXXX+\b', r'lorem', r'ipsum',
    r'placeholder', r'to be filled', r'fill in', r'insert your', r'insert the',
    r'replace (this|with)', r'space reserved', r'values to be', r'if required',
    r'\bnote to\b', r'<[A-Za-z][A-Za-z ./]{2,}>', r'your (name|paper|data)',
    r'\byou should\b', r'\bplease \w+', r'\[insert', r'\bN/A\b',
]
notes = []
for pat in NOTE_PAT:
    for m in re.finditer(pat, src, re.I):
        line = src[:m.start()].count('\n') + 1
        notes.append('line %d: %s' % (line, src[max(0, m.start() - 40):m.end() + 40]
                                      .replace('\n', ' ')))
if notes:
    fails.append('note-like text found: %s' % notes[:5])
print('comment lines in output: %d' % sum(
    1 for ln in src.split('\n') if ln.lstrip().startswith('%')))
print('note-like strings: %d' % len(notes))

print('\n' + ('ALL STRUCTURAL CHECKS PASSED' if not fails else 'ISSUES:'))
for f in fails:
    print('  - %s' % f)
