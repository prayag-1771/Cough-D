"""Build the archive to upload to Overleaf, one per target.

    python -I overleaf_bundle.py                 # both targets
    python -I overleaf_bundle.py --venue ieee

There is no TeX on the machine this manuscript was produced on, so the
document is built elsewhere, and what gets uploaded has to be assembled by
hand every time the source changes. Doing it by hand is how a file gets left
out, and a missing font or class file fails the build with an error that says
nothing about which file is absent. This writes the archive instead.

What goes in is derived from the assembled file rather than listed here: the
class comes from its \\documentclass, the figures from its \\includegraphics.
Only the support files that the Access class needs and does not name in a way
a script can read are enumerated, and the archive is refused if any of them
is missing, so an incomplete upload is a failure here rather than a confusing
log on Overleaf.

The archives are not committed. The Access class, its fonts and the IEEE
logos are the publisher's to distribute and are ignored by git, so an archive
containing them must be ignored too; it is reproduced by running this.
"""
import argparse
import glob
import os
import re
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
    'ieee': {'tex': 'paper_ieee.tex', 'zip': 'ieee_access_upload.zip',
             'support': IEEE_SUPPORT, 'fonts': True},
    'springer': {'tex': 'paper.tex', 'zip': 'springer_upload.zip',
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


def build(venue):
    files = wanted(venue)
    missing = [f for f in files if not os.path.exists(os.path.join(MS, f))]
    if missing:
        print('%s: not built, %d file(s) missing:' % (venue, len(missing)))
        for f in missing:
            print('    ' + f)
        return 1

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, VENUES[venue]['zip'])
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(MS, f), f)

    fonts = sum(1 for f in files if f.endswith(('.pfb', '.tfm', '.map')))
    named = [f for f in files if not f.endswith(('.pfb', '.tfm', '.map'))]
    print('%s -> %s' % (venue, os.path.relpath(path, ROOT)))
    print('    %d files, %.0f KB' % (len(files),
                                     os.path.getsize(path) / 1024.0))
    for f in named:
        print('      ' + f)
    if fonts:
        print('      and %d font and map files' % fonts)
    print('    compile with pdfLaTeX; the bibliography is inlined, so no '
          'BibTeX pass is needed')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', choices=sorted(VENUES), default=None,
                    help='default: both')
    a = ap.parse_args()
    bad = 0
    for venue in ([a.venue] if a.venue else sorted(VENUES)):
        bad |= build(venue)
        print()
    return bad


if __name__ == '__main__':
    sys.exit(main())
