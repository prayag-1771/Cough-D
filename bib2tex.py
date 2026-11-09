"""Generate a thebibliography block from references.bib, in citation order.

    python -I bib2tex.py references.bib body1.tex body2.tex ... -o bibliography.tex

Springer requires the manuscript as one .tex file, so the reference list is
emitted as a thebibliography block rather than left to BibTeX. Generating it
from the .bib keeps one source of truth: thirty-two entries retyped by hand
would be thirty-two fresh chances to introduce an error, and an error in a
reference is the kind a reader cannot catch without opening the cited paper.

Entries are ordered by first citation in the body files, in the order those
files are given. A key cited but not defined, or defined but never cited, is
reported and causes a non-zero exit, because either one means the manuscript
and the bibliography have drifted apart.

No entry carries a digital object identifier. The .bib holds none, and this
script emits no field it was not given.
"""
import argparse
import os
import re
import sys

# {COVID-19} protects capitalisation for BibTeX and is meaningless in a
# thebibliography block, but {\~n} is an accent and must survive. So a brace
# group is unwrapped only when it contains no backslash and no nested brace.
PLAIN_GROUP = re.compile(r'\{([^{}\\]*)\}')


def unwrap(s):
    prev = None
    while prev != s:
        prev = s
        s = PLAIN_GROUP.sub(r'\1', s)
    return ' '.join(s.split())


def parse(path):
    """Minimal reader for the entries in this project's own .bib."""
    text = open(path, encoding='utf-8').read()
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        kind, key = m.group(1).lower(), m.group(2)
        # Scan from the brace that opens the entry, not from the comma after
        # the key. Starting at the comma made the first field's own closing
        # brace return the depth to zero, so the body ended after the author
        # list and every later field was read as absent. The symptom was a
        # reference list with correct authors and no titles, journals or
        # years, which the entry count in the structural check could not see.
        i, depth = text.index('{', m.start()), 0
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[m.end():i]
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{', body):
            name = fm.group(1).lower()
            j, d = fm.end() - 1, 0
            while j < len(body):
                if body[j] == '{':
                    d += 1
                elif body[j] == '}':
                    d -= 1
                    if d == 0:
                        break
                j += 1
            fields[name] = body[fm.end():j]
        fields['__kind__'] = kind
        entries[key] = fields
    return entries


def initials(given):
    """'David L.' -> 'D.L.'  'S. Grace' -> 'S.G.'  'Chlo{\\\"e}' -> 'C.'"""
    out = []
    for part in given.replace('.', ' ').split():
        letter = re.sub(r'[^A-Za-z]', '', part)
        if letter:
            out.append(letter[0].upper() + '.')
        elif part.startswith('{'):
            out.append(part[1:2].upper() + '.')
    return ''.join(out)


def format_authors(raw):
    names = [a.strip() for a in re.split(r'\s+and\s+', unwrap(raw)) if a.strip()]
    out, trailing = [], ''
    for n in names:
        if n == 'others':
            trailing = ' et al.'
            continue
        if ',' in n:
            sur, given = [p.strip() for p in n.split(',', 1)]
            out.append('%s, %s' % (sur, initials(given)) if given else sur)
        else:
            bits = n.split()
            out.append('%s, %s' % (bits[-1], initials(' '.join(bits[:-1]))))
    return ', '.join(out) + trailing


def ieee_initials(given):
    """'Hassan N.' -> 'H. N. '   IEEE puts a space between initials."""
    out = []
    for part in given.replace('.', ' ').split():
        letter = re.sub(r'[^A-Za-z]', '', part)
        if letter:
            out.append(letter[0].upper() + '.')
        elif part.startswith('{'):
            out.append(part[1:2].upper() + '.')
    return ' '.join(out)


def ieee_authors(raw):
    """Given names abbreviated to initials and placed before the surname.

    The whole list is given rather than truncated, which is what the Access
    template asks for: et al. is used only where the bibliography itself
    records that the names were not published.
    """
    names = [a.strip() for a in re.split(r'\s+and\s+', unwrap(raw)) if a.strip()]
    out, etal = [], False
    for n in names:
        if n == 'others':
            etal = True
            continue
        if ',' in n:
            sur, given = [p.strip() for p in n.split(',', 1)]
        else:
            bits = n.split()
            sur, given = bits[-1], ' '.join(bits[:-1])
        ini = ieee_initials(given)
        out.append(('%s %s' % (ini, sur)).strip())
    if etal:
        return ', '.join(out) + ', \\emph{et al.}'
    if len(out) > 2:
        return ', '.join(out[:-1]) + ', and ' + out[-1]
    return ' and '.join(out)


def ieee_pages(pages):
    """'4811--4815' -> ('pp.', '4811--4815');  '156' -> ('p.', '156')."""
    if not pages:
        return None
    pages = pages.replace('–', '--')
    return ('pp.' if '--' in pages else 'p.'), pages


def format_entry_ieee(f):
    """One reference in the style the Access template sets out.

    Titles and container names are reproduced as the publisher prints them.
    The style manual asks for sentence case and abbreviated periodical
    names, but applying either by rule would mean lowercasing or contracting
    words this script cannot tell apart from proper nouns, and a reference is
    the one place in a manuscript where a reader cannot catch an error
    without going to the cited work. Reproducing what was read from the
    source is therefore preferred over guessing at the house style.
    """
    a = ieee_authors(f.get('author', ''))
    title = unwrap(f.get('title', '')).rstrip('.')
    year = unwrap(f.get('year', ''))
    vol = unwrap(f.get('volume', ''))
    num = unwrap(f.get('number', ''))
    pp = ieee_pages(unwrap(f.get('pages', '')))
    kind = f['__kind__']

    bits = ['%s, ``%s,\'\'' % (a, title)] if a else ['``%s,\'\'' % title]

    if kind == 'article':
        bits.append('\\emph{%s},' % unwrap(f.get('journal', '')))
        if vol:
            bits.append('vol. %s,' % vol)
        if num:
            bits.append('no. %s,' % num)
        if pp:
            bits.append('%s %s,' % pp)
        bits.append('%s.' % year)
    elif kind in ('inproceedings', 'incollection'):
        bits.append('in \\emph{%s},' % unwrap(f.get('booktitle', '')))
        if pp:
            bits.append('%s, %s %s.' % (year, pp[0], pp[1]))
        else:
            bits.append('%s.' % year)
    elif kind == 'book':
        pub = unwrap(f.get('publisher', ''))
        bits = ['%s, \\emph{%s}.' % (a, title)] if a else ['\\emph{%s}.' % title]
        if pub:
            bits.append('%s,' % pub)
        bits.append('%s.' % year)
    else:
        how = unwrap(f.get('howpublished', '')) or unwrap(f.get('note', ''))
        bits.append('%s.' % year)
        if how:
            bits.append('[Online]. Available: %s' % how)

    return ' '.join(b for b in bits if b.strip())


def format_entry(f):
    a = format_authors(f.get('author', ''))
    title = unwrap(f.get('title', '')).rstrip('.')
    year = unwrap(f.get('year', ''))
    pages = unwrap(f.get('pages', '')).replace('--', '\u2013')
    vol = unwrap(f.get('volume', ''))
    num = unwrap(f.get('number', ''))
    kind = f['__kind__']

    if kind == 'article':
        where = unwrap(f.get('journal', ''))
        tail = where
        if vol:
            tail += ' %s' % vol
            if num:
                tail += '(%s)' % num
        if pages:
            tail += ', %s' % pages
        return '%s: %s. %s (%s)' % (a, title, tail, year)

    if kind in ('inproceedings', 'incollection'):
        where = unwrap(f.get('booktitle', ''))
        tail = 'In: %s' % where
        if pages:
            tail += ', pp. %s' % pages
        return '%s: %s. %s (%s)' % (a, title, tail, year)

    how = unwrap(f.get('howpublished', '')) or unwrap(f.get('note', ''))
    if how:
        return '%s: %s. %s (%s)' % (a, title, how, year)
    return '%s: %s (%s)' % (a, title, year)


def cited_order(paths):
    seen, order = set(), []
    for p in paths:
        text = open(p, encoding='utf-8').read()
        for m in re.finditer(r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}', text):
            for k in m.group(1).split(','):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    order.append(k)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bib')
    ap.add_argument('body', nargs='+')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--style', choices=('springer', 'ieee'), default='springer')
    a = ap.parse_args()
    fmt = format_entry_ieee if a.style == 'ieee' else format_entry
    width = '00' if a.style == 'ieee' else None

    entries = parse(a.bib)
    order = cited_order(a.body)

    undefined = [k for k in order if k not in entries]
    uncited = [k for k in entries if k not in order]
    for k in undefined:
        print('  cited but not in the bibliography: %s' % k, file=sys.stderr)
    for k in uncited:
        print('  in the bibliography but never cited: %s' % k, file=sys.stderr)

    lines = ['\\begin{thebibliography}{%s}' % (width or len(order))]
    for k in order:
        if k in entries:
            lines.append('')
            lines.append('\\bibitem{%s}' % k)
            lines.append(fmt(entries[k]))
    lines.append('')
    lines.append('\\end{thebibliography}')
    text = '\n'.join(lines) + '\n'

    if a.out:
        open(a.out, 'w', encoding='utf-8', newline='\n').write(text)
        print('wrote %s: %d entries in citation order'
              % (os.path.basename(a.out), len(order) - len(undefined)))
    else:
        sys.stdout.write(text)

    return 1 if (undefined or uncited) else 0


if __name__ == '__main__':
    sys.exit(main())
