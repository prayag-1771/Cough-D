"""Drop the folder prefix from every graphics call.

With \\graphicspath set in the preamble, a bare filename resolves whether the
images sit in a figs subfolder or flat beside the document, which removes the
most common upload mistake as a failure mode.
"""
import io
import re

PAT = re.compile(r'(' + re.escape(chr(92)) + r'includegraphics'
                 r'\[[^\]]*\]\{)figs/')

total = 0
for p in ('part2.tex', 'part3.tex'):
    s = io.open(p, encoding='utf-8').read()
    s2, k = PAT.subn(lambda m: m.group(1), s)
    if k:
        io.open(p, 'w', encoding='utf-8', newline='\n').write(s2)
    print('%s: %d prefixes removed' % (p, k))
    total += k
print('total: %d' % total)
