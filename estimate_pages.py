"""Estimate the compiled length. No TeX distribution is available here, so
this is an estimate from content volume, not a measurement."""
import io
import re

s = io.open('Cough_Detection_Wavelet_Denoising_SpringerNature.tex', encoding='utf-8').read()
body = s[s.index(r'\maketitle'):]

st = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', body, flags=re.S)
st = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', st, flags=re.S)
st = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', st, flags=re.S)
words = len(re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', st).split())

eqs = len(re.findall(r'\\begin\{(?:equation|align)\}', s))
tabs = len(re.findall(r'\\begin\{table\}', s))
figs = len(re.findall(r'\\begin\{figure\}', s))
refs = len(re.findall(r'\\bibitem', s))
secs = len(re.findall(r'\\section\{', s))
subs = len(re.findall(r'\\subsection\{', s))

text_pp = words / 520.0
est = text_pp + tabs * 0.40 + figs * 0.50 + eqs * 0.085 + refs * 0.082 + 1.3

print('body words           %5d   -> %4.1f pp of text' % (words, text_pp))
print('tables               %5d   -> %4.1f pp' % (tabs, tabs * 0.40))
print('figures              %5d   -> %4.1f pp' % (figs, figs * 0.50))
print('display equations    %5d   -> %4.1f pp' % (eqs, eqs * 0.085))
print('references           %5d   -> %4.1f pp' % (refs, refs * 0.082))
print('front matter                   ->  1.3 pp')
print('sections %d, subsections %d' % (secs, subs))
print()
print('ESTIMATED COMPILED LENGTH: ~%.0f pages' % est)
