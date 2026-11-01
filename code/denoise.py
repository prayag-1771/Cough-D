"""Wavelet denoising for the cough experiments, with a stated shrinkage rule.

A note on fidelity, because it bears on what any rerun can claim.

The original experiments called MATLAB's wdenoise(x,6), whose defaults are the
sym4 wavelet at level six with an empirical Bayesian rule: a mixture prior
with a Cauchy component, the mixing weight estimated by marginal maximum
likelihood, and each coefficient replaced by its posterior median. That is the
Johnstone and Silverman procedure as implemented in EbayesThresh. Reproducing
it exactly outside MATLAB means reimplementing that estimator, and an
approximation to it that is subtly wrong would be worse than a different rule
that is exactly right, because its error would be invisible.

So this module implements two rules that are standard and verifiable instead:

  'visu'   soft thresholding against the universal threshold, sigma * sqrt(2
           log N), with sigma estimated once from the finest detail level by
           the median absolute deviation. This is the Donoho and Johnstone
           procedure, and it is what the manuscript's own equations describe.

  'bayes'  BayesShrink, a level adaptive empirical Bayes rule whose threshold
           at level j is sigma^2 / sigma_x,j, with sigma_x,j the signal
           standard deviation implied at that level. This is empirical Bayes
           in the same spirit as the MATLAB default, with a closed form.

Running both is the point rather than a compromise: if the effect of denoising
survives a change of shrinkage rule it is a property of denoising, and if it
does not then it was a property of one rule. Neither is bit identical to the
original MATLAB run, and results obtained here are a replication under a
specified denoiser, not a reproduction of that run.
"""
import numpy as np
import pywt

WAVELET = 'sym4'
LEVEL = 6
MAD_TO_SIGMA = 0.6745          # MAD of a standard normal


def _sigma_from_finest(detail_finest):
    """Level independent noise estimate from the finest detail coefficients."""
    if detail_finest.size == 0:
        return 0.0
    return float(np.median(np.abs(detail_finest)) / MAD_TO_SIGMA)


def _soft(d, lam):
    return np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)


def denoise(x, rule='visu', wavelet=WAVELET, level=LEVEL):
    """Denoise a 1-D signal. Returns an array the same length as x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2 ** level:
        return x.copy()                      # too short to decompose that far

    coeffs = pywt.wavedec(x, wavelet, level=level, mode='symmetric')
    approx, details = coeffs[0], list(coeffs[1:])

    sigma = _sigma_from_finest(details[-1])  # details[-1] is the finest level
    if sigma <= 0:
        return x.copy()

    if rule == 'visu':
        lam = sigma * np.sqrt(2.0 * np.log(n))
        shrunk = [_soft(d, lam) for d in details]

    elif rule == 'bayes':
        shrunk = []
        for d in details:
            if d.size == 0:
                shrunk.append(d)
                continue
            var_y = float(np.mean(d ** 2))
            var_x = max(var_y - sigma ** 2, 0.0)
            if var_x <= 0:
                shrunk.append(np.zeros_like(d))  # level judged to be all noise
            else:
                shrunk.append(_soft(d, sigma ** 2 / np.sqrt(var_x)))

    else:
        raise ValueError("rule must be 'visu' or 'bayes', got %r" % (rule,))

    out = pywt.waverec([approx] + shrunk, wavelet, mode='symmetric')
    return out[:n]                           # waverec can overshoot by one


# --------------------------------------------------------------------- checks
def _selftest():
    """Both rules must raise the signal to noise ratio on a noisy test signal.

    A denoiser that does not is broken, and the whole experiment downstream
    would be meaningless, so this runs before any audio is touched.
    """
    rng = np.random.default_rng(0)
    n = 16384
    t = np.linspace(0, 1, n, endpoint=False)

    # a transient burst, which is what a cough looks like at this scale
    clean = np.zeros(n)
    for centre in (0.2, 0.5, 0.78):
        env = np.exp(-0.5 * ((t - centre) / 0.012) ** 2)
        clean += env * np.sin(2 * np.pi * 900 * t)

    ok = True
    print('%-8s %10s %10s %10s' % ('rule', 'noisy SNR', 'clean SNR', 'gain dB'))
    for rule in ('visu', 'bayes'):
        for noise_sd in (0.05, 0.2):
            noisy = clean + rng.normal(0, noise_sd, n)
            est = denoise(noisy, rule=rule)
            assert est.shape == noisy.shape, 'length changed: %s' % (est.shape,)

            def snr(y):
                return 10 * np.log10(np.sum(clean ** 2) / np.sum((y - clean) ** 2))

            before, after = snr(noisy), snr(est)
            good = after > before
            ok &= good
            print('%-8s %10.2f %10.2f %10.2f  %s'
                  % (rule + ' @%.2f' % noise_sd, before, after, after - before,
                     'ok' if good else 'FAILED'))

    # a constant signal must survive essentially untouched
    flat = np.ones(4096) * 3.0
    for rule in ('visu', 'bayes'):
        err = float(np.max(np.abs(denoise(flat, rule=rule) - 3.0)))
        good = err < 1e-6
        ok &= good
        print('%-8s constant signal preserved, max error %.2e  %s'
              % (rule, err, 'ok' if good else 'FAILED'))

    # a signal shorter than the decomposition must come back unchanged
    short = np.array([1.0, -2.0, 3.0])
    ok &= np.allclose(denoise(short), short)
    print('short input returned unchanged: %s' % np.allclose(denoise(short), short))

    print()
    print('SELFTEST %s' % ('PASSED' if ok else 'FAILED'))
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
