"""Estimate the built length of the manuscript without building it.

    python -I pagecount.py

The submission has to reach a stated number of pages, and the only honest way
to know is to compile the document and look. There is no TeX installation on
this machine, so this estimates instead, and it is an estimate: it counts the
words of running prose, the floats, and the reference entries, and converts
them with rates for a single column Springer page.

The rates are given as a range rather than a single number because the
conversion is the uncertain part. A figure quoted from here should be read as
telling you whether the document is comfortably over the target, close to it,
or short, and not as a page number.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A literal backslash inside a regular expression is written as two
# characters, so this is what the TeX control sequences are matched with.
RBS = chr(92) * 2

# Words of running prose per page, for a single column Springer article at the
# class default size. The spread is the honest uncertainty in the conversion.
RATES = (480, 530, 580)
PER_TABLE = 0.30          # pages, for a table of the sizes used here
PER_FIGURE = 0.42
REFS_PER_PAGE = 17.0
FRONT = 1.3               # title block, abstract, keywords


def strip_commands(s):
    s = re.sub(RBS + r'[a-zA-Z@]+\s*(\[[^\]]*\])?(\{[^{}]*\})?', ' ', s)
    return re.sub(r'[{}$' + RBS + r']', ' ', s)


def main():
    path = os.path.join(ROOT, 'manuscript', 'paper.tex')
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit('no assembled manuscript at %s; run assemble.py first' % path)
    t = io.open(path, encoding='utf-8').read()

    i = t.find('maketitle')
    body = t[i:] if i > 0 else t

    tables = len(re.findall(r'begin\{table', body))
    figures = len(re.findall(r'begin\{figure', body))
    refs = len(re.findall(r'bibitem', body))

    nofloat = re.sub(RBS + r'begin\{(table|figure)\*?\}.*?'
                     + RBS + r'end\{\1\*?\}', ' ', body, flags=re.S)
    j = nofloat.find('begin{thebibliography}')
    prose = nofloat[:j] if j > 0 else nofloat
    words = len(strip_commands(prose).split())

    print('prose words, floats and bibliography excluded: %d' % words)
    print('tables %d, figures %d, reference entries %d'
          % (tables, figures, refs))
    print()
    floats = tables * PER_TABLE + figures * PER_FIGURE
    for rate in RATES:
        pages = words / float(rate) + floats + refs / REFS_PER_PAGE + FRONT
        print('  at %d words per page: about %.1f pages' % (rate, pages))
    low = words / float(RATES[-1]) + floats + refs / REFS_PER_PAGE + FRONT
    print()
    if low >= 25:
        print('comfortably over 25 pages even on the pessimistic rate')
    elif words / float(RATES[0]) + floats + refs / REFS_PER_PAGE + FRONT >= 25:
        print('near 25 pages; the pessimistic rate falls short by %.1f'
              % (25 - low))
    else:
        print('short of 25 pages on every rate, by %.1f at best'
              % (25 - (words / float(RATES[0]) + floats
                       + refs / REFS_PER_PAGE + FRONT)))


if __name__ == '__main__':
    main()
