"""Make the denoising figure caption describe what the figure actually shows.

imagefinal.py builds panels (c) and (f) as the absolute difference between the
two eight bit rendered spectrograms, rescaled to the full range with
NORM_MINMAX, amplified fourfold with convertScaleAbs and false coloured. Two
consequences follow. The panels are not the removed wavelet detail, and
because each is rescaled independently of the other, their densities are on
different scales and cannot be compared across panels. The original caption
asserted both.
"""
import io

P2 = 'part2.tex'
t = io.open(P2, encoding='utf-8').read()

OLD = r"""\caption{Effect of the sym4 decomposition at level six on a representative
cough recording, in the top row, and a representative noncough recording, in
the bottom row. Each panel is a time frequency representation, with time on
the horizontal axis and frequency on the vertical. Panels (a) and (d) show
the conditioned input, (b) and (e) the reconstruction after shrinkage, and
(c) and (f) the components that shrinkage removed. The dominant transient
structure of the cough is preserved from (a) to (b), confirming that the
operation does not suppress the energy the classifier depends on, while the
residual in (c) is visibly denser than that in (f), which is consistent with
the cough carrying more broadband transient content than the noncough
material.}"""

NEW = r"""\caption{Effect of the sym4 decomposition at level six on a representative
cough recording, in the top row, and a representative noncough recording, in
the bottom row. Each panel is a time frequency representation, with time on
the horizontal axis and frequency on the vertical. Panels (a) and (d) show
the conditioned input and (b) and (e) the reconstruction after shrinkage. The
dominant transient structure of the cough is preserved from (a) to (b), which
is the property the operation has to have if it is not to suppress the energy
the classifier depends on. Panels (c) and (f) are diagnostic rather than
quantitative: each is the absolute difference between the two images above it,
rescaled to the full display range and amplified to make faint structure
visible. They indicate where in the plane the reconstruction departs from the
input, and that the departure avoids the region carrying the cough transient.
Because each difference panel is rescaled independently of the other, the
intensities in (c) and (f) are not on a common scale and the two panels should
not be compared against each other.}"""

assert t.count(OLD) == 1, 'figure caption not matched exactly once'
t = t.replace(OLD, NEW)
io.open(P2, 'w', encoding='utf-8', newline='\n').write(t)
print('figure caption corrected')
