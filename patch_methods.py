"""Correct the methods sections against the code the authors actually ran.

Every replacement is asserted to match exactly once, so a drift between this
script and the manuscript stops the run instead of silently doing nothing.
The source of truth is code_extracted.txt, lifted from the submitted PDF.
"""
import io

P2 = 'part2.tex'
t = io.open(P2, encoding='utf-8').read()
orig = t
log = []


def rep(old, new, label, count=1):
    global t
    n = t.count(old)
    assert n == count, '%s: expected %d match, found %d' % (label, count, n)
    t = t.replace(old, new)
    log.append('  %-52s %d' % (label, n))


# ---------------------------------------------------------------- denoising
# wdenoise(x,6) applies the toolbox default, which is empirical Bayesian
# shrinkage with a Cauchy prior and a posterior median rule, not the soft
# thresholding rule the manuscript set out. sym4 and level six are correct.
rep(
    r"""Denoising proceeds by attenuating detail coefficients whose magnitude is
consistent with noise rather than with signal. Writing $d_{j,k}$ for the
$k$th coefficient at level $j$, the soft thresholding rule
\begin{equation}
\tilde{d}_{j,k} = \operatorname{sgn}(d_{j,k}) \cdot
\max\big(|d_{j,k}| - \lambda_j,\; 0\big)
\label{eq:soft}
\end{equation}
shrinks every coefficient toward zero by $\lambda_j$ and sets to zero those
that do not survive the subtraction~\cite{donoho1995}. The rule is preferred
to simply discarding coefficients below the threshold because it is
continuous in $d_{j,k}$, so a coefficient near the threshold does not produce
a discontinuous change in the reconstruction, and because its risk properties
under additive noise are well characterised~\cite{donoho1994}. The signal is
reconstructed from the retained approximation and the shrunken details.""",
    r"""Denoising proceeds by attenuating detail coefficients whose magnitude is
consistent with noise rather than with signal. The classical treatment fixes
a threshold per level and shrinks each coefficient toward zero by that
amount, discarding those that do not survive the
subtraction~\cite{donoho1994, donoho1995}. The procedure used here is the
empirical Bayesian alternative, which places a mixture prior on the detail
coefficients at each level,
\begin{equation}
d_{j,k} \sim (1 - w_j)\,\delta_0 + w_j\,\gamma,
\label{eq:eb}
\end{equation}
where $\delta_0$ is a point mass at zero, $\gamma$ is a heavy tailed Cauchy
density and the mixing weight $w_j$ is estimated from the coefficients at
level $j$ by marginal maximum likelihood. Each coefficient is then replaced
by the median of its posterior distribution~\cite{johnstone2005}. Because
$w_j$ is estimated separately at every level, the amount of shrinkage adapts
to how much of the energy at that scale is attributable to noise instead of
being set by one global rule, and the heavy tail of the prior leaves large
coefficients close to unshrunk. The signal is reconstructed from the retained
approximation and the shrunken details.""",
    'denoising rule: soft threshold -> empirical Bayes')

# ------------------------------------------------------- training table body
rep(
    r"""Learning rate & $10^{-3}$ \\
Batch size & 32 \\
Epochs, ResNet18 and MobileNetV2 & 10 \\
Epochs, DenseNet201 & 3 \\
Validation interval & every 50 iterations (ResNet18, MobileNetV2) \\
 & every 200 iterations (DenseNet201) \\
Sampling rate & 16~kHz, mono \\
Wavelet & sym4, decomposition level 6 \\
Perceptual spectrogram & 512 point window, 256 overlap, 128 filters \\
Network input size & $224 \times 224$ \\""",
    r"""Learning rate, all models except DenseNet201 & $10^{-3}$ \\
Learning rate, DenseNet201 & $10^{-4}$ \\
Batch size & 32 \\
Epochs, compact network, ResNet18, MobileNetV2 & 10 \\
Epochs, DenseNet201 & 3 \\
Monitoring interval & every 50 iterations (ResNet18, MobileNetV2) \\
 & every 200 iterations (DenseNet201) \\
Data used for monitoring & the test partition \\
Network retained & the one at the final iteration \\
Random seed & not fixed \\
Sampling rate, perceptual axis models & 16~kHz, mono \\
Sampling rate, compact network & as recorded, mono \\
Wavelet & sym4, level 6, empirical Bayes \\
Linear axis spectrogram & 256 point window, 200 overlap, \\
 & 256 point transform \\
Perceptual spectrogram & 512 point window, 256 overlap, 128 filters \\
Network input size & $224 \times 224$ \\""",
    'training table rows corrected')

# ---------------------------------------------------------- table caption
rep(
    r"""\caption{Training configuration. The epoch budget is the one respect in which
the transferred models were not treated identically, and
Section~\ref{sec:res-limits} discusses what follows from that for the
interpretation of the DenseNet201 result.}""",
    r"""\caption{Training configuration. The epoch budget and the learning rate are
the two respects in which the transferred models were not treated
identically, and the material held out for measurement was also the material
monitored during training. Section~\ref{sec:res-limits} states what each of
these costs the interpretation of the results.}""",
    'training table caption')

# --------------------------------------------------- learning rate in prose
rep(
    r"""All models were trained in MATLAB using the Deep Learning Toolbox on a
workstation with an NVIDIA RTX 4050 graphics processor. Optimisation used
Adam with a learning rate of $10^{-3}$ and a batch size of 32.
Table~\ref{tab:train} records the settings, including the point on which the
configurations are not uniform.""",
    r"""All models were trained in MATLAB using the Deep Learning Toolbox on a
workstation with an NVIDIA RTX 4050 graphics processor. Optimisation used
Adam with a batch size of 32 throughout, a learning rate of $10^{-3}$ for
every model except DenseNet201, and $10^{-4}$ for DenseNet201.
Table~\ref{tab:train} records the settings, including the three points on
which the configurations are not uniform.""",
    'learning rate in prose')

# ------------------------------- disclosure paragraph after the epoch remark
rep(
    r"""DenseNet201 was trained for three epochs against ten for the other two
transferred models, because its depth makes each epoch considerably more
expensive. This is a real asymmetry in the experimental design and we do not
present it as anything else. Its consequence is discussed in
Section~\ref{sec:res-limits}, where we set out which of our conclusions it
can and cannot affect.""",
    r"""DenseNet201 was trained for three epochs against ten for the other two
transferred models, because its depth makes each epoch considerably more
expensive, and at a learning rate an order of magnitude lower. These are real
asymmetries in the experimental design and we do not present them as anything
else. Their consequence is discussed in Section~\ref{sec:res-limits}, where we
set out which of our conclusions they can and cannot affect.

Two further properties of the procedure must be stated plainly, because they
bear on how every number in Section~\ref{sec:results} should be read. The
first is that no separate validation partition was constructed. The test
partition was supplied to the training routine as the data against which
progress was monitored, and the accuracy reported for each configuration is
measured on that same material. No quantity computed on the test partition
was used to select weights, since the network retained is the one at the final
iteration rather than the one at the best monitored score, but the
arrangement removes the separation between the data used to observe training
and the data used to report performance. The figures below are therefore not
estimates of performance on unseen material in the strict sense, and we treat
them as measurements on a partition that was visible throughout rather than as
generalisation estimates.

The second is that the compact network and the three transferred networks do
not receive comparable inputs, and the difference is larger than the change of
frequency axis alone. For the transferred models the perceptual spectrogram is
computed as a matrix, compressed logarithmically, mapped to the unit interval
and written as a single channel image, which is then replicated across three
channels at load time. For the compact network the linear axis spectrogram is
drawn to a figure and the rendered figure is captured as a raster image, so
the network receives a picture of a plot rather than the transform itself,
carrying the display colour map and whatever resolution the rendering surface
provided. The two paths also differ in sample rate, because the recordings are
resampled to 16~kHz only on the perceptual axis branch. The comparison between
the compact network and the transferred networks therefore varies five things
at once, and Section~\ref{sec:res-limits} explains why we decline to attribute
the difference between them to any one of the five.""",
    'asymmetry + monitoring disclosure')

io.open(P2, 'w', encoding='utf-8', newline='\n').write(t)
print('patched %s  (%d -> %d chars)' % (P2, len(orig), len(t)))
for line in log:
    print(line)
