"""Bring the results and limitations into line with what the design supports.

Adds the interval estimates and the class balanced measures, withdraws the
attribution of the compact to transferred gap, and states the two limitations
the code revealed. Replacements are asserted, so drift stops the run.
"""
import io

P3 = 'part3.tex'
t = io.open(P3, encoding='utf-8').read()
orig = t
log = []


def rep(old, new, label, count=1):
    global t
    n = t.count(old)
    assert n == count, '%s: expected %d match, found %d' % (label, count, n)
    t = t.replace(old, new)
    log.append('  %-54s %d' % (label, n))


# -------------------------------------------- withdraw the gap attribution
rep(
    r"""Two features of the table deserve comment before the individual factors are
separated. The first is that the three transferred architectures lie close
together, spanning 0.77 accuracy points across their six configurations, while
the gap between the best transferred configuration and the compact network is
2.67 points. The decision to transfer weights from a large image corpus
therefore matters considerably more here than the choice among modern
architectures, which is consistent with the general finding that transferred
initialisation is most valuable when the target corpus is small relative to
the capacity of the model~\cite{pan2010}. The second is that every
configuration exceeds the majority class baseline of 68.26 percent by at least
25 points, so all eight models have learned structure in the signal rather
than exploiting the prior.""",
    r"""Two features of the table deserve comment before the individual factors are
separated. The first is that the three transferred architectures lie close
together, spanning 0.77 accuracy points across their six configurations, while
the gap between the best transferred configuration and the compact network is
2.67 points. That gap is the only difference in the table wide enough to
exceed the uncertainty attaching to a single accuracy measurement, for the
reasons set out in Section~\ref{sec:res-precision}, and it is correspondingly
the only one on which a claim could rest. It will not carry the claim that
transferred initialisation is what produced it. The compact network differs
from the transferred networks in five respects simultaneously: the
initialisation, the architecture, the frequency axis, the sample rate at which
the recording was analysed, and whether the network is presented with the
spectrogram as a numerical array or with a raster image of that spectrogram as
plotted to a figure. Any one of the five is capable of moving accuracy by more
than two points, and the design separates none of them. We therefore report
the gap and decline to attribute it. The second feature is that every
configuration exceeds the majority class baseline of 68.26 percent by at least
25 points, so all eight models have learned structure in the signal rather
than exploiting the prior.""",
    'withdraw compact-to-transferred attribution')

# --------------------------------- new subsection ahead of the limitations
NEW = r"""\subsection{Statistical precision}\label{sec:res-precision}

The differences discussed above are small, and their interpretation depends on
how precisely a proportion measured on 4,577 recordings is known. We therefore
attach an interval to every accuracy figure and report measures that do not
inherit the class imbalance. Each accuracy is a binomial proportion estimated
on a finite sample, and we summarise its precision with the Wilson score
interval, which is preferred to the normal approximation because it remains
inside the unit interval and keeps its nominal coverage when the proportion
approaches one, as all of these do~\cite{brown2001}. Because accuracy on a
corpus with 2.15 coughs for every noncough recording is dominated by the
majority class, we also report specificity, the proportion of noncough
recordings correctly rejected, together with balanced accuracy, the unweighted
mean of recall and specificity, and the Matthews correlation coefficient,
which uses all four cells of the confusion matrix and is the measure least
distorted by imbalance~\cite{chicco2020}. Table~\ref{tab:interval} collects
these for the eight configurations.

\begin{table}[t]
\centering
\footnotesize
\caption{Interval estimates and class balanced measures for the eight
configurations. Accuracy is accompanied by its 95 percent Wilson score
interval on the 4,577 test recordings. Specificity is the proportion of the
1,453 noncough recordings correctly rejected, balanced accuracy the unweighted
mean of recall and specificity, and MCC the Matthews correlation coefficient.
Wav abbreviates wavelet denoising applied before spectrogram extraction, and
Perc the perceptual frequency axis. The interval around every individual
accuracy is wider than any difference between a configuration and its pair.}
\label{tab:interval}
\begin{tabular}{@{}llrcrrr@{}}
\toprule
Architecture & Input & Acc. & 95\% interval & Spec. & Bal. & MCC \\
\midrule
Compact & Linear & 93.45 & 92.69--94.13 & 94.98 & 93.85 & 0.855 \\
Compact & Wav + linear & 93.86 & 93.13--94.52 & 92.98 & 93.63 & 0.861 \\
ResNet18 & Perc & 95.76 & 95.14--96.31 & 90.85 & 94.45 & 0.901 \\
ResNet18 & Wav + Perc & \textbf{96.53} & 95.96--97.02 & 94.84 & \textbf{96.07} & \textbf{0.920} \\
MobileNetV2 & Perc & 95.76 & 95.14--96.31 & \textbf{95.73} & 95.75 & 0.904 \\
MobileNetV2 & Wav + Perc & 96.11 & 95.51--96.63 & 93.53 & 95.42 & 0.910 \\
DenseNet201 & Perc & 96.15 & 95.56--96.67 & 93.39 & 95.42 & 0.911 \\
DenseNet201 & Wav + Perc & 96.09 & 95.49--96.61 & 92.43 & 95.11 & 0.909 \\
\bottomrule
\end{tabular}
\end{table}

The intervals settle the status of the denoising comparisons. Their mean width
is 1.20 accuracy points, while the largest difference denoising produces in
any pair is 0.77 points on ResNet18 and the smallest is 0.06 points on
DenseNet201. Every difference the study reports between a configuration and
its denoised counterpart is therefore smaller than the uncertainty on either
of the two measurements being compared, and the intervals of all four pairs
overlap substantially. On this evidence the effect of denoising on accuracy is
not resolved by the experiment in either direction: the data are consistent
with a small benefit, with no effect, and with a small cost, and the pattern
of signs across the four architectures carries no more information than the
magnitudes do. The same arithmetic applies to the 0.77 point spread across the
three transferred architectures, which is likewise contained within a single
interval and does not support ranking them.

It should be said precisely what would settle the question, because the
present design cannot. A paired comparison of two classifiers on identical
test material is properly tested by McNemar's statistic, which depends only on
the two discordant counts, the recordings one model classifies correctly and
the other does not~\cite{dietterich1998}. Those counts are properties of the
paired per recording decisions and are not recoverable from confusion
matrices, which report only the margins. The margins do bound the discordance
from below: between the two ResNet18 configurations at least 81 of the 4,577
recordings must be classified differently, since 58 false positives are
removed and 23 false negatives introduced. The sign and size of the effect,
however, require the decisions themselves, which means retaining per recording
predictions for every configuration. We note this as a requirement on the
design rather than as a result.

The class balanced measures also change which configuration appears best, and
in a way accuracy conceals. Specificity varies across the eight configurations
from 90.85 to 95.73 percent, a spread of 4.88 points that is four times the
width of an accuracy interval and the largest systematic variation anywhere in
the study. ResNet18 on undenoised perceptual spectrograms has the weakest
specificity in the table at 90.85 percent, yet its accuracy of 95.76 percent
is identical to that of MobileNetV2 on the same input, which rejects noncough
material at 95.73 percent. Two configurations indistinguishable on accuracy
thus differ by almost five points in the rate at which they raise false
alarms, which is the quantity that governs whether a continuously running
detector is tolerable to use. Balanced accuracy and the Matthews coefficient
both rank the denoised ResNet18 configuration first, at 96.07 percent and
0.920 respectively, and both separate the compact network from the transferred
ones by a margin larger than they separate the transferred ones from each
other. These two measures are the ones we would ask a reader to compare, and
the ranking they produce is the one the discussion above relies on.

"""

rep(r"""\subsection{Limitations}\label{sec:res-limits}

Four limitations bound what the present results establish, and we state them
with their consequences rather than deferring them.""",
    NEW + r"""\subsection{Limitations}\label{sec:res-limits}

Six limitations bound what the present results establish, and we state them
with their consequences rather than deferring them.""",
    'insert precision subsection + limitation count')

# ------------------------------- correct the single-run limitation paragraph
rep(
    r"""Each configuration was trained once, so no estimate of variation across random
initialisations is available. The differences on which the main conclusions
rest are large relative to this concern: 2.67 points between transferred and
scratch trained models, and a reduction of 58 false positives from 1,453
noncough recordings for the denoising comparison on ResNet18. The smaller
differences in the table, in particular the 0.06 points separating the two
DenseNet201 configurations, should not be read as meaningful in isolation, and
indeed we treat that difference as a null result rather than as a decline.""",
    r"""Each configuration was trained once, and the random seed was not fixed, so
neither an estimate of variation across initialisations nor an exact
reproduction of any individual run is available. This is the most serious of
the six, because Section~\ref{sec:res-precision} shows that every difference
the study reports between a configuration and its denoised counterpart is
smaller than the sampling uncertainty on a single accuracy measurement, before
any variation between runs is added. The consequence is that the denoising
comparisons are unresolved rather than negative, and that the ordering of the
three transferred architectures is not established. Only two findings survive
this limitation: the 2.67 point separation between the compact network and the
transferred networks, which exceeds the interval width but is confounded five
ways and is therefore reported without attribution, and the 4.88 point spread
in specificity, which exceeds it by a wider margin. Repeating every
configuration across several seeds and reporting the distribution rather than
a single value is the minimum that would change this, and retaining per
recording predictions so that paired tests become available is what would
settle the denoising question."""
    ,
    'correct the single-run limitation')

# ----------------------------------------- two limitations the code revealed
rep(
    r"""The evaluation is confined to one corpus and to the binary task.""",
    r"""No separate validation partition was constructed. The material on which the
reported accuracies are measured is the same material that was supplied to the
training routine for monitoring, as recorded in Table~\ref{tab:train}. The
network retained in each case is the one at the final iteration rather than
the one at the best monitored score, so no quantity computed on the test
partition entered the selection of weights, and the figures are not inflated
by that mechanism. What the arrangement does remove is the separation between
the data used to observe training and the data used to report performance,
which means the numbers should be read as measurements on a partition that was
visible throughout rather than as estimates of performance on unseen
recordings. A three way division of the corpus, with the epoch budget and the
learning rate chosen on a validation partition and the test partition touched
once, is the correction, and it is a precondition for the other improvements
listed here rather than an alternative to them.

The inputs given to the compact network and to the transferred networks differ
in more than the frequency axis. The perceptual spectrograms are computed as
arrays, compressed logarithmically and written as single channel images,
whereas the linear axis spectrograms are produced by plotting each transform
to a figure and capturing the rendered figure as a raster image, so the
compact network is presented with a picture of a plot rather than with the
transform. The two paths also analyse the recordings at different sample
rates. The practical consequences are twofold. The gap between the compact
network and the transferred networks cannot be assigned to initialisation,
architecture, frequency axis, sample rate or input encoding, because all five
vary together, and the linear axis images cannot be regenerated exactly by
anyone, including us, because a figure capture depends on the resolution and
the colour handling of the surface it was rendered on. Recomputing the linear
axis spectrograms as arrays on the same 16~kHz material, and pairing them with
a transferred architecture, would remove the confound and restore
reproducibility at the same time; this is the single change we would make
first.

The evaluation is confined to one corpus and to the binary task.""",
    'add monitoring-partition and input-encoding limitations')

# ----------------------------------------------------- conclusion correction
rep(
    r"""would not reveal: perceptual frequency scaling improved every architecture to
which it was applied, transferred initialisation contributed considerably more
than the choice among modern architectures, and wavelet denoising before
spectrogram extraction benefited some networks and not others.""",
    r"""would not reveal: perceptual frequency scaling improved every architecture to
which it was applied, the separation between the compact network and the
transferred networks was wider than any separation among the transferred
networks themselves, and wavelet denoising before spectrogram extraction
produced differences that this design is not precise enough to resolve.""",
    'conclusion: opening claims')

# ------------------------------------------------------------ bibliography
rep(
    r"""\bibitem{nemati2019}""",
    r"""\bibitem{johnstone2005}
Johnstone, I.M., Silverman, B.W.: Empirical Bayes selection of wavelet
thresholds. The Annals of Statistics \textbf{33}(4), 1700--1752 (2005)

\bibitem{brown2001}
Brown, L.D., Cai, T.T., DasGupta, A.: Interval estimation for a binomial
proportion. Statistical Science \textbf{16}(2), 101--133 (2001)

\bibitem{chicco2020}
Chicco, D., Jurman, G.: The advantages of the Matthews correlation
coefficient over F1 score and accuracy in binary classification evaluation.
BMC Genomics \textbf{21}, 6 (2020)

\bibitem{dietterich1998}
Dietterich, T.G.: Approximate statistical tests for comparing supervised
classification learning algorithms. Neural Computation \textbf{10}(7),
1895--1923 (1998)

\bibitem{nemati2019}""",
    'four new bibitems')

io.open(P3, 'w', encoding='utf-8', newline='\n').write(t)
print('patched %s  (%d -> %d chars)' % (P3, len(orig), len(t)))
for line in log:
    print(line)
