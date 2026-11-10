"""Collect what Overleaf needs, one folder per target.

    python -I overleaf_bundle.py                 # both targets
    python -I overleaf_bundle.py --venue ieee
    python -I overleaf_bundle.py --zip           # archives as well

There is no TeX on the machine this manuscript was produced on, so the
document is built elsewhere, and what gets uploaded has to be assembled again
every time the source changes. Doing it by hand is how a file gets left out,
and a missing font or class file fails the build with an error that says
nothing about which file is absent.

The files are laid out open, in the tree they have to keep, so they can be
dragged in or inspected one at a time; --zip additionally writes an archive
for the upload-project route.

What goes in is derived from the assembled file rather than listed here: the
class comes from its \\documentclass, the figures from its \\includegraphics.
Only the support files that the Access class needs and does not name in a way
a script can read are enumerated, and the folder is refused if any of them is
missing, so an incomplete upload is a failure here rather than a confusing
log on Overleaf.

references.bib is copied in as well, for reference rather than for the build.
The bibliography is already expanded into the manuscript by bib2tex.py, so
the document compiles in one pass with no BibTeX run and nothing in it reads
the .bib.

None of this is committed. The Access class, its fonts and the IEEE logos are
the publisher's to distribute and are ignored by git, so a copy of them must
be ignored too; it is reproduced by running this.
"""
import argparse
import glob
import os
import re
import shutil
import sys
import zipfile

BS = chr(92)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, 'manuscript')
OUT = os.path.join(ROOT, 'overleaf')

# Files the Access class loads without naming them in a form a script can
# follow: the fonts arrive through \pdfmapfile, the logos through the title
# page. The Springer class needs nothing beyond itself.
IEEE_SUPPORT = ['IEEEtran.cls', 'spotcolor.sty',
                'bullet.png', 'logo.png', 'notaglinelogo.png']

# Photographs for the biographies, included when present. The manuscript
# guards them with \IfFileExists, so the build does not need them.
OPTIONAL = ['figs/Prayag_Sharma.jpg', 'figs/Ravi_Tiwari.jpeg']

VENUES = {
    'ieee': {'tex': 'paper_ieee.tex', 'dir': 'ieee_access',
             'support': IEEE_SUPPORT, 'fonts': True},
    'springer': {'tex': 'paper.tex', 'dir': 'springer',
                 'support': [], 'fonts': False},
}


def wanted(venue):
    """Every file the chosen target needs, as paths relative to manuscript."""
    v = VENUES[venue]
    tex = os.path.join(MS, v['tex'])
    if not os.path.exists(tex):
        sys.exit('%s is not assembled; run assemble.py --venue %s first'
                 % (v['tex'], venue))
    src = open(tex, encoding='utf-8').read()

    cls = re.search(BS * 2 + r'documentclass(?:\[[^\]]*\])?\{([^{}]+)\}', src)
    if not cls:
        sys.exit('%s names no document class' % v['tex'])

    files = [v['tex'], cls.group(1) + '.cls'] + list(v['support'])

    # \graphicspath puts figs/ on the search path, so a figure named bare in
    # the source is uploaded under figs/ and found there exactly as here.
    for m in re.finditer(BS * 2 + r'includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}',
                         src):
        name = m.group(1)
        for cand in (name, os.path.join('figs', name)):
            if os.path.exists(os.path.join(MS, cand)):
                files.append(cand.replace(os.sep, '/'))
                break

    if v['fonts']:
        for pat in ('*.map', '*.pfb', '*.tfm'):
            files += [os.path.basename(p)
                      for p in sorted(glob.glob(os.path.join(MS, pat)))]

    for opt in OPTIONAL:
        if os.path.exists(os.path.join(MS, opt)):
            files.append(opt)

    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build(venue, make_zip=False):
    files = wanted(venue)
    missing = [f for f in files if not os.path.exists(os.path.join(MS, f))]
    if missing:
        print('%s: not built, %d file(s) missing:' % (venue, len(missing)))
        for f in missing:
            print('    ' + f)
        return 1

    dest = os.path.join(OUT, VENUES[venue]['dir'])
    # Rewritten from scratch each time, so a file that has left the
    # manuscript does not linger here and get uploaded.
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for f in files:
        target = os.path.join(dest, f.replace('/', os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(MS, f), target)

    # The .bib is for reading, not for building: bib2tex.py has already
    # expanded the bibliography into the manuscript.
    bib = os.path.join(ROOT, 'references.bib')
    if os.path.exists(bib):
        shutil.copy2(bib, os.path.join(dest, 'references.bib'))
        files = files + ['references.bib  (not used by the build)']

    total = sum(os.path.getsize(os.path.join(dest, f.replace('/', os.sep)))
                for f in os.listdir(dest)
                if os.path.isfile(os.path.join(dest, f)))
    fonts = sum(1 for f in files if f.endswith(('.pfb', '.tfm', '.map')))
    named = [f for f in files if not f.endswith(('.pfb', '.tfm', '.map'))]
    print('%s -> %s%s' % (venue, os.path.relpath(dest, ROOT), os.sep))
    for f in named:
        print('      ' + f)
    if fonts:
        print('      %d font and map files' % fonts)

    if make_zip:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, VENUES[venue]['dir'] + '_upload.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for base, _dirs, names in os.walk(dest):
                for n in sorted(names):
                    full = os.path.join(base, n)
                    z.write(full, os.path.relpath(full, dest).replace(
                        os.sep, '/'))
        print('    also %s, %.0f KB' % (os.path.basename(path),
                                        os.path.getsize(path) / 1024.0))

    print('    compile with pdfLaTeX; the bibliography is inlined, so no '
          'BibTeX pass is needed')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', choices=sorted(VENUES), default=None,
                    help='default: both')
    ap.add_argument('--zip', action='store_true',
                    help='also write an archive beside the folder')
    a = ap.parse_args()
    bad = 0
    for venue in ([a.venue] if a.venue else sorted(VENUES)):
        bad |= build(venue, a.zip)
        print()
    return bad


if __name__ == '__main__':
    sys.exit(main())
