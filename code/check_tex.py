"""Catch the LaTeX errors that cannot be seen without a TeX installation.

    python -I check_tex.py                 # every section source
    python -I check_tex.py ../manuscript/paper_ieee.tex

There is no TeX on the machine this manuscript was produced on, so a source
error that only shows up at build time can sit in the repository indefinitely.
One did: every row of the three tables written by hand ended in a single
backslash instead of two, because the shell that wrote them strips one level
of backslash from a here document. The structural checker could not see it,
since it reasons about sections, floats and citations rather than about tokens,
and a single trailing backslash is perfectly good prose as far as it is
concerned.

What is checked here is the mechanical layer underneath that: whether the
delimiters balance, whether a row of a table is terminated, and whether a word
that is only ever a command appears without its backslash. None of these needs
to know what the manuscript says, and all of them are fatal at build time.

The command list is deliberately short. It holds only words that have no
meaning as English in this manuscript, so a bare occurrence is always a lost
backslash and never a false positive.
"""
import glob
import io
import os
import re
import sys

BS = chr(92)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, 'manuscript')

# Words that are commands and nothing else. "end" and "small" are left out on
# purpose: both are ordinary English and would fire on every page.
COMMANDS = (
    'toprule', 'midrule', 'bottomrule', 'botrule', 'addlinespace',
    'multicolumn', 'multirow', 'tabular', 'includegraphics', 'centering',
    'textbf', 'texttt', 'textsf', 'emph', 'eqref', 'bmhead', 'maketitle',
    'documentclass', 'usepackage', 'newcommand', 'bibitem', 'thebibliography',
    'authorrefmark', 'tfootnote', 'markboth', 'corresp', 'titlepgskip',
    'PARstart', 'uppercase', 'orgdiv', 'orgname', 'orgaddress', 'affil',
    'keywords', 'backmatter', 'figref', 'parstart',
)

# Environments whose body is made of rows that must each be terminated.
ROW_ENVS = ('tabular', 'tabular*', 'array')
# Lines inside such an environment that are not rows.
NOT_A_ROW = re.compile(
    r'^\s*(%|' + BS * 2 + r'(begin|end|toprule|midrule|bottomrule|botrule|'
    r'addlinespace|cmidrule|hline|caption|label|small|footnotesize|'
    r'centering|multicolumn2)\b)')


def unescaped(text, ch):
    return len(re.findall(r'(?<!' + BS * 2 + r')' + re.escape(ch), text))


def check_delimiters(path, text, out):
    o, c = unescaped(text, '{'), unescaped(text, '}')
    if o != c:
        out.append('%s: %d unescaped { against %d }' % (path, o, c))
    d = unescaped(text, '$')
    if d % 2:
        out.append('%s: %d math delimiters, which is odd' % (path, d))


def check_bare_commands(path, text, out):
    for name in COMMANDS:
        # The backslash has to be doubled inside the class: a single one
        # would escape the closing bracket and the class would never end.
        # A brace, an underscore or a hyphen before the word means it is the
        # argument of some other command rather than a command itself, as in
        # an environment name, a package name or a file name, so those are
        # excluded too. The cost is that a backslash lost immediately after
        # an opening brace is not seen; the benefit is that the check reports
        # only real damage, which is what makes it worth running.
        pattern = (r'(?<![A-Za-z' + BS * 2 + r'{_-])' + name
                   + r'(?![A-Za-z])')
        for m in re.finditer(pattern, text):
            line = text.count('\n', 0, m.start()) + 1
            out.append('%s:%d: %r without its backslash'
                       % (path, line, name))


def check_rows(path, text, out):
    """Every row of a tabular has to end in a row terminator."""
    depth = 0
    for i, raw in enumerate(text.split('\n'), 1):
        line = raw.rstrip()
        opened = re.search(BS * 2 + r'begin\{(tabular\*?|array)\}', line)
        closed = re.search(BS * 2 + r'end\{(tabular\*?|array)\}', line)
        if opened:
            depth += 1
            continue
        if closed:
            depth = max(0, depth - 1)
            continue
        if not depth or not line.strip():
            continue
        if NOT_A_ROW.match(line):
            continue
        # A row may be wrapped over several lines, so only a line that ends
        # the row is required to carry the terminator; a continuation is one
        # that is followed by more of the same row. The test used here is
        # that the line must end either in a terminator or in something that
        # cannot end a row, which is what a trailing ampersand or an open
        # construct looks like.
        if line.endswith(BS * 2):
            continue
        if line.endswith('&') or line.endswith('{') or line.endswith('%'):
            continue
        # a single trailing backslash is the damage this script exists for
        if line.endswith(BS) and not line.endswith(BS * 2):
            out.append('%s:%d: row ends in one backslash, not two'
                       % (path, i))


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob(os.path.join(MS, '[0-9]*.tex'))) + \
            sorted(glob.glob(os.path.join(MS, 'tab_*.tex')))
    if not paths:
        sys.exit('nothing to check')

    out = []
    for p in paths:
        text = io.open(p, encoding='utf-8', errors='replace').read()
        name = os.path.basename(p)
        check_delimiters(name, text, out)
        check_bare_commands(name, text, out)
        check_rows(name, text, out)

    print('checked %d file(s) for token level damage' % len(paths))
    if not out:
        print('delimiters balance, every row is terminated, and no command '
              'has lost its backslash')
        return 0
    print('%d problem(s):' % len(out))
    for line in out:
        print('  ' + line)
    return 1


if __name__ == '__main__':
    sys.exit(main())
