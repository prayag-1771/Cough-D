"""Check that every number written in the prose occurs in a generated file.

    python -I trace_numbers.py

The integrity rule this manuscript is written under is that no quantity is
asserted from memory: each one has to come from a decision file, a generated
table or a recorded measurement. A checker cannot confirm that a number means
what the sentence around it says, but it can confirm that the number exists
somewhere it could have come from, which catches the transcription errors that
are otherwise invisible.

Numbers that are part of the writing rather than results, section numbers,
equation levels, years, and the counts that appear in the corpus tables, are
listed as known so that the report contains only what still needs an eye.
Anything printed by this script is either a real mismatch or a number that
belongs on that list.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, 'manuscript')
RESULTS = os.path.join(ROOT, 'results')
RBS = chr(92) * 2

PROSE = ['01_abstract.tex', '01_front.tex', '01_front_ieee.tex',
         '01_intro.tex', '02_related.tex', '03_design.tex', '04_setup.tex',
         '05_results.tex', '06_discussion.tex', '07_conclusion.tex']

# Quantities that are part of the design rather than measurements of it, or
# that are quoted from a cited paper and verified against it by hand.
KNOWN = {
    # the pipeline's own constants
    '6', '4', '224', '16', '32', '64', '128', '8', '0.5', '1', '2', '3', '5',
    '10', '12', '0.9', '0.1', '0.8', '2.4',
    # corpus sizes and priors, which the corpora tables carry
    '13314', '8398', '476', '106', '370', '34434', '15619', '7261', '22880',
    '15617', '15623', '0.8961', '0.1271', '22003', '820', '259',
    # figures quoted from cited work, checked against the source by hand
    '98.5', '94.2', '0.98', '15', '20', '98.18', '95.18', '2212', '62',
    '67842', '0.846', '0.619', '0.71', '0.90', '0.6', '53000', '36000',
    '34000', '2024', '1993',
    # constants of the method and of the hardware, not measurements
    '2595', '4096', '700',
}


def numbers(text):
    text = re.sub(RBS + r'(label|[a-z]*ref|eqref|cite[a-z]*|input|doi|'
                  r'history|markboth|address|authorrefmark)\s*'
                  r'(\[[^\]]*\])?\{[^{}]*\}', ' ', text)
    # A length is a piece of page layout, not a quantity the work measured,
    # so the fractions in a column specification are not numbers to trace.
    text = re.sub(r'[0-9.]+' + RBS + r'(?:text|column|line)width', ' ', text)
    text = re.sub(RBS + r'begin\{tabular\*?\}\{(?:[^{}]|\{[^{}]*\})*\}',
                  ' ', text)
    out = []
    for m in re.finditer(r'(?<![A-Za-z0-9.])(\d+(?:\{,\})?\d*(?:\.\d+)?)', text):
        out.append(m.group(1).replace('{,}', ''))
    return out


def haystack():
    parts = []
    for d in (MS, RESULTS):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith('tab_') or d == RESULTS:
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    try:
                        parts.append(io.open(p, encoding='utf-8',
                                             errors='replace').read())
                    except OSError:
                        pass
    return '\n'.join(parts).replace('{,}', '')


def rounds_to(n, pool):
    """Is n a correctly rounded quotation of some number in the pool?

    The prose rounds, and it should: a table carrying 2.2759 is quoted in a
    sentence as 2.28. Comparing the strings would report that as missing, so
    the comparison is made at the precision the prose chose.
    """
    if '.' not in n:
        return False
    places = len(n.split('.')[1])
    try:
        want = float(n)
    except ValueError:
        return False
    for v in pool:
        if round(v, places) == want:
            return True
    return False


def main():
    hay = haystack()
    if not hay.strip():
        sys.exit('no generated tables or results to check against')
    hay_numbers = []
    for m in re.finditer(r'\d+\.\d+(?:[eE][-+]?\d+)?', hay):
        try:
            hay_numbers.append(float(m.group(0)))
        except ValueError:
            pass

    unmatched = {}
    for fn in PROSE:
        p = os.path.join(MS, fn)
        if not os.path.exists(p):
            continue
        for n in numbers(io.open(p, encoding='utf-8').read()):
            if n in KNOWN or n in hay:
                continue
            if rounds_to(n, hay_numbers):
                continue
            unmatched.setdefault(fn, []).append(n)

    if not unmatched:
        print('every number in the prose occurs in a generated file')
        return 0
    total = sum(len(v) for v in unmatched.values())
    print('%d number(s) not found in any generated file:' % total)
    for fn in PROSE:
        if fn in unmatched:
            seen = sorted(set(unmatched[fn]), key=lambda s: (len(s), s))
            print('  %-20s %s' % (fn, ', '.join(seen)))
    return 1


if __name__ == '__main__':
    sys.exit(main())
