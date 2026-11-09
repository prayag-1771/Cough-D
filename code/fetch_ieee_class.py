"""Install the IEEE Access class and the fonts it needs, from IEEE's own zip.

    python -I fetch_ieee_class.py

The Access class is not on CTAN. It is distributed only inside the template
archive that the journal publishes, together with IEEEtran, a colour package
and the licensed Formata and Giovanni font files that the class loads by name.
A build fails without any one of them, so all of them are installed beside the
manuscript rather than a subset.

Those font files and the journal's logos are IEEE's to distribute and not
ours, so they are kept out of the repository and this script stands in their
place: anyone who has the repository can obtain them again from the publisher
in one step. The template's own sample document, its sample figures and its
BibTeX style are not installed. The reference list is generated as a
thebibliography block by bib2tex.py, so no .bst takes part in the build.
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(os.path.dirname(HERE), 'manuscript')
URL = ('https://ieeeaccess.ieee.org/wp-content/uploads/2025/08/'
       'ACCESS_latex_template_20240429.zip')

# Installed by name or by extension. The extensions are the font machinery:
# .fd maps a family name to a font, .map tells the driver where the outlines
# are, .tfm carries the metrics and .pfb the outlines themselves.
BY_NAME = ('ieeeaccess.cls', 'IEEEtran.cls', 'spotcolor.sty',
           'logo.png', 'notaglinelogo.png', 'bullet.png')
BY_EXT = ('.fd', '.map', '.tfm', '.pfb')


def main():
    print('fetching %s' % URL)
    work = tempfile.mkdtemp(prefix='ieeeaccess_')
    try:
        archive = os.path.join(work, 'template.zip')
        # The server refuses the default Python user agent with a 403, so a
        # plain one is sent. Nothing identifying is included.
        req = urllib.request.Request(URL, headers={'User-Agent': 'curl/8'})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        with open(archive, 'wb') as fh:
            fh.write(data)
        print('  %d bytes, sha256 %s'
              % (len(data), hashlib.sha256(data).hexdigest()[:16]))

        installed = 0
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = os.path.basename(info.filename)
                if not base:
                    continue
                if base not in BY_NAME and not base.endswith(BY_EXT):
                    continue
                # Read the member and write it ourselves, so nothing in the
                # archive can decide where a file lands.
                with z.open(info) as src:
                    payload = src.read()
                with open(os.path.join(MS, base), 'wb') as dst:
                    dst.write(payload)
                installed += 1
        print('installed %d files into manuscript/' % installed)
        missing = [n for n in BY_NAME if not os.path.exists(os.path.join(MS, n))]
        if missing:
            sys.exit('the archive did not contain: %s' % ', '.join(missing))
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
