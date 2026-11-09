"""Estimate the built length of the manuscript without building it.

    python -I pagecount.py
    python -I pagecount.py --venue ieee

The submission has to reach a stated number of pages, and the only honest way
to know is to compile the document and look. There is no TeX installation on
this machine, so this estimates instead, and it is an estimate: it counts the
words of running prose, the floats, and the reference entries, and converts
them with rates for the page the chosen class produces.

The two targets differ by more than a constant. A two column page carries
roughly twice the prose of a single column one and sets the reference list at
a smaller size, so the same body is a much shorter document in the Access
class than in the Springer one. A target length therefore belongs to a venue
and not to the manuscript, which is why the figure to compare against is
given per venue below rather than fixed at one number.

The rates are given as a range rather than a single number because the
conversion is the uncertain part. A figure quoted from here should be read as
telling you whether the document is comfortably over the target, close to it,
or short, and not as a page number.
"""
import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A literal backslash inside a regular expression is written as two
# characters, so this is what the TeX control sequences are matched with.
RBS = chr(92) * 2

VENUES = {
    # rates: words of running prose per page, at the class default size.
    # The spread within a venue is the honest uncertainty in the conversion.
    'springer': {
        'file': 'paper.tex',
        'rates': (480, 530, 580),
        'per_table': 0.30,
        'per_figure': 0.42,
        'refs_per_page': 17.0,
        'front': 1.3,
        'target': 25,
    },
    'ieee': {
        'file': 'paper_ieee.tex',
        'rates': (950, 1050, 1150),
        'per_table': 0.18,
        'per_figure': 0.24,
        'refs_per_page': 42.0,
        'front': 1.0,
        # Access sets no minimum length. It asks instead that the length be
        # commensurate with the work, so there is nothing to clear here and
        # the estimate is reported without a verdict.
        'target': None,
    },
}


def strip_commands(s):
    s = re.sub(RBS + r'[a-zA-Z@]+\s*(\[[^\]]*\])?(\{[^{}]*\})?', ' ', s)
    return re.sub(r'[{}$' + RBS + r']', ' ', s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', choices=sorted(VENUES), default='springer')
    ap.add_argument('path', nargs='?')
    a = ap.parse_args()
    v = VENUES[a.venue]

    path = a.path or os.path.join(ROOT, 'manuscript', v['file'])
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

    print('%s target, from %s' % (a.venue, os.path.basename(path)))
    print('prose words, floats and bibliography excluded: %d' % words)
    print('tables %d, figures %d, reference entries %d'
          % (tables, figures, refs))
    print()
    floats = tables * v['per_table'] + figures * v['per_figure']
    fixed = floats + refs / v['refs_per_page'] + v['front']

    def pages(rate):
        return words / float(rate) + fixed

    for rate in v['rates']:
        print('  at %d words per page: about %.1f pages' % (rate, pages(rate)))
    print()

    target = v['target']
    if target is None:
        print('this class sets no minimum length, so there is nothing to clear')
        return 0
    if pages(v['rates'][-1]) >= target:
        print('comfortably over %d pages even on the pessimistic rate' % target)
    elif pages(v['rates'][0]) >= target:
        print('near %d pages; the pessimistic rate falls short by %.1f'
              % (target, target - pages(v['rates'][-1])))
    else:
        print('short of %d pages on every rate, by %.1f at best'
              % (target, target - pages(v['rates'][0])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
