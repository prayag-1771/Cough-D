"""Assemble the manuscript sections into the single file the publisher wants.

    python -I assemble.py
    python -I assemble.py --out manuscript/paper.tex --check

Springer asks for one .tex file. Keeping the sections apart while writing them
and joining them here means the join is reproducible and the pieces stay small
enough to reason about. The bibliography is generated from references.bib in
citation order by bib2tex.py, so the .bib remains the only place a reference is
written down.

Nothing is reformatted on the way through. The output is the concatenation of
the parts in the order given below, with the generated bibliography placed
after the back matter and before the end of the document.
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
PARTS = [
    '00_preamble.tex',
    '01_front.tex',
    '01_intro.tex',
    '02_related.tex',
    '03_design.tex',
    '04_setup.tex',
    '05_results.tex',
    '06_conclusion.tex',
    '99_back.tex',
]

# Only these carry \cite, and bib2tex reads them to fix the reference order.
BODY = PARTS[1:]


INPUT = re.compile(r'^[ \t]*\\input\{([^}]+)\}[ \t]*$', re.M)


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
    ap.add_argument('--out', default=os.path.join('manuscript', 'paper.tex'))
    ap.add_argument('--check', action='store_true',
                    help='run check_paper.py on the assembled file')
    a = ap.parse_args()

    bib_out = os.path.join(MS, 'bibliography.tex')
    cmd = [sys.executable, '-I', os.path.join(ROOT, 'bib2tex.py'),
           os.path.join(ROOT, 'references.bib')]
    cmd += [os.path.join(MS, n) for n in BODY if os.path.exists(os.path.join(MS, n))]
    cmd += ['-o', bib_out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit('the bibliography and the manuscript disagree; see above')

    chunks = [read(n) for n in PARTS[:-1]]
    chunks.append(read(PARTS[-1]))
    chunks.append(read('bibliography.tex'))
    chunks.append('\\end{document}\n')

    out = os.path.join(ROOT, a.out)
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(chunks))

    words = sum(len(c.split()) for c in chunks)
    print('wrote %s: %d parts, %d words of source'
          % (os.path.relpath(out, ROOT), len(PARTS), words))

    if a.check:
        r = subprocess.run([sys.executable, '-I',
                            os.path.join(ROOT, 'check_paper.py'), out])
        return r.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
