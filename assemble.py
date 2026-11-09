"""Assemble the manuscript sections into the single file the publisher wants.

    python -I assemble.py --venue ieee --check
    python -I assemble.py --venue springer --check

Both targets ask for one .tex file. Keeping the sections apart while writing
them and joining them here means the join is reproducible and the pieces stay
small enough to reason about. The bibliography is generated from references.bib
in citation order by bib2tex.py, so the .bib remains the only place a reference
is written down, in either house style.

Two targets share one body. The section files carry no class specific markup,
so a venue is a title page, a closing set of statements, a reference style and
the position of the reference list, and nothing else. Where the two houses
genuinely differ inside a sentence, as they do over whether a figure is a
Figure or a Fig., the difference is a macro defined in the preamble rather
than a second copy of the sentence.

Nothing is reformatted on the way through. The output is the concatenation of
the parts in the order the chosen venue gives, with the generated bibliography
placed where that venue puts it.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(ROOT, 'manuscript')

# The order the document is read in. The front matter carries the title block
# and the abstract, so it has to precede everything \maketitle depends on.
SECTIONS = [
    '01_intro.tex',
    '02_related.tex',
    '03_design.tex',
    '04_setup.tex',
    '05_results.tex',
    '06_discussion.tex',
    '07_conclusion.tex',
]

# One body, two title pages. The section files above carry no class specific
# markup, so each target differs only in what precedes and follows them, and
# in where the reference list sits: Springer closes with the back matter and
# then the references, the Access template puts the references between the
# back matter and the author biographies.
VENUES = {
    'springer': {
        'head': ['00_preamble.tex', '01_front.tex'],
        'tail': ['99_back.tex'],
        'after_bib': [],
        'style': 'springer',
        'out': os.path.join('manuscript', 'paper.tex'),
    },
    'ieee': {
        'head': ['00_preamble_ieee.tex', '01_front_ieee.tex'],
        'tail': ['99_back_ieee.tex'],
        'after_bib': ['99_bios_ieee.tex'],
        'style': 'ieee',
        'out': os.path.join('manuscript', 'paper_ieee.tex'),
    },
}


INPUT = re.compile(r'\\input\{([^{}]+)\}')


def read(name, depth=0):
    """One section, with any \\input it names pulled in where it stands.

    The result tables are generated from the saved decisions by tables.py
    rather than written out by hand, so the sections refer to them by
    \\input. The publisher wants one file, and expanding here keeps both
    true at once: the tables stay generated and the output stays single.
    """
    path = os.path.join(MS, name)
    if not os.path.exists(path):
        sys.exit('missing section: %s' % name)
    with open(path, encoding='utf-8') as fh:
        text = fh.read().rstrip() + '\n'
    if depth > 4:
        sys.exit('\\input nested too deeply at %s' % name)

    def swap(m):
        target = m.group(1)
        if not target.endswith('.tex'):
            target += '.tex'
        if not os.path.exists(os.path.join(MS, target)):
            sys.exit('%s refers to a missing file: %s' % (name, target))
        return read(target, depth + 1).rstrip()

    return INPUT.sub(swap, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', choices=sorted(VENUES), default='springer')
    ap.add_argument('--out', default=None)
    ap.add_argument('--check', action='store_true',
                    help='run check_paper.py on the assembled file')
    a = ap.parse_args()
    v = VENUES[a.venue]

    parts = v['head'] + SECTIONS + v['tail']
    bib_name = 'bibliography.tex' if a.venue == 'springer' \
        else 'bibliography_ieee.tex'
    bib_out = os.path.join(MS, bib_name)
    cmd = [sys.executable, '-I', os.path.join(ROOT, 'bib2tex.py'),
           os.path.join(ROOT, 'references.bib')]
    cmd += [os.path.join(MS, n) for n in parts[1:]
            if os.path.exists(os.path.join(MS, n))]
    cmd += ['-o', bib_out, '--style', v['style']]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit('the bibliography and the manuscript disagree; see above')

    chunks = [read(n) for n in parts]
    chunks.append(read(bib_name))
    chunks.extend(read(n) for n in v['after_bib'])
    chunks.append('\\end{document}\n')

    out = os.path.join(ROOT, a.out or v['out'])
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(chunks))

    words = sum(len(c.split()) for c in chunks)
    print('wrote %s: %d parts, %d words of source'
          % (os.path.relpath(out, ROOT), len(chunks), words))

    if a.check:
        # Token level first. If a row is unterminated or a delimiter does not
        # balance, the document will not build at all, and there is no point
        # reporting on its structure until it does.
        tok = subprocess.run([sys.executable, '-I',
                              os.path.join(ROOT, 'code', 'check_tex.py'), out])
        r = subprocess.run([sys.executable, '-I',
                            os.path.join(ROOT, 'check_paper.py'), out,
                            '--venue', a.venue])
        return r.returncode or tok.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
