"""Decoding and spectrogram extraction, with no librosa dependency.

The mel filterbank is built here rather than imported so that the exact
definition is visible and testable: triangular filters on the HTK mel scale,
each normalised to unit area. MATLAB's melSpectrogram uses a Slaney style
bank, so the two are not numerically identical; the choice is stated in the
manuscript rather than left implicit.

Decoding goes through the ffmpeg binary that imageio_ffmpeg ships, because
COUGHVID distributes webm and ogg containers and libsndfile reads neither.
"""
import subprocess
import numpy as np

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 128
OUT_SIZE = 224

_FFMPEG = None


def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def decode(path, sr=SR):
    """Decode any container ffmpeg understands to mono float32 at sr.

    Returns an empty array if the file cannot be decoded, which happens in
    COUGHVID for a handful of truncated uploads.
    """
    cmd = [ffmpeg_exe(), '-nostdin', '-v', 'error', '-i', str(path),
           '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return np.zeros(0, dtype=np.float32)
    if p.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(p.stdout, dtype='<f4').astype(np.float32)


# ------------------------------------------------------------ mel filterbank
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """(n_mels, 1 + n_fft//2) matrix of unit area triangular filters."""
    if fmax is None:
        fmax = sr / 2.0
    n_bins = 1 + n_fft // 2
    fft_hz = np.linspace(0.0, sr / 2.0, n_bins)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        if hi <= lo:
            continue
        rise = (fft_hz - lo) / max(mid - lo, 1e-12)
        fall = (hi - fft_hz) / max(hi - mid, 1e-12)
        fb[i] = np.clip(np.minimum(rise, fall), 0.0, None)
        area = fb[i].sum()
        if area > 0:
            fb[i] /= area
    return fb


_FB = None


def _fb():
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    return _FB


# ----------------------------------------------------------------- transforms
def _frames(x, n_fft=N_FFT, hop=HOP):
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def magnitude(x, n_fft=N_FFT, hop=HOP):
    """One sided magnitude STFT, Hamming window, shape (bins, frames)."""
    w = np.hamming(n_fft)
    F = _frames(np.asarray(x, dtype=np.float64), n_fft, hop) * w
    return np.abs(np.fft.rfft(F, n=n_fft, axis=1)).T


def to_image(S, size=OUT_SIZE):
    """Log compress, map to the unit interval, resize, return uint8.

    The resize uses PIL with bilinear interpolation, matching the behaviour of
    imresize on a greyscale matrix closely enough for this purpose. Time is
    normalised away by the resize, exactly as in the original pipeline, so
    clip duration carries no information into the network.
    """
    from PIL import Image
    S = np.log10(np.asarray(S, dtype=np.float64) + 1e-6)
    lo, hi = float(S.min()), float(S.max())
    S = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)
    img = Image.fromarray((S * 255.0 + 0.5).astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.BILINEAR))


def spectrogram_image(x, kind, size=OUT_SIZE):
    """kind is 'linear' or 'mel'. Returns a (size, size) uint8 array."""
    S = magnitude(x)
    if kind == 'mel':
        S = _fb() @ S
    elif kind != 'linear':
        raise ValueError("kind must be 'linear' or 'mel', got %r" % (kind,))
    return to_image(S, size)


# --------------------------------------------------------------------- checks
def _selftest():
    ok = True

    fb = mel_filterbank()
    print('filterbank shape %s' % (fb.shape,))
    assert fb.shape == (N_MELS, 1 + N_FFT // 2)
    nonzero = int((fb.sum(axis=1) > 0).sum())
    print('filters with nonzero area: %d of %d' % (nonzero, N_MELS))
    areas = fb.sum(axis=1)
    live = areas[areas > 0]
    good = np.allclose(live, 1.0, atol=1e-9)
    ok &= good
    print('live filters are unit area: %s' % good)
    # 128 mel bands over 257 linear bins leaves the lowest filters narrower
    # than one bin, so some are empty by construction. Reporting the count
    # matters because the manuscript claims 128 bands.
    if nonzero < N_MELS:
        print('NOTE %d filters are empty: 128 bands over %d bins is oversampled'
              % (N_MELS - nonzero, 1 + N_FFT // 2))

    # a pure tone must put its mel energy in a stable, low-index-ordered place
    sr = SR
    t = np.arange(sr) / sr
    peaks = []
    for f0 in (200.0, 1000.0, 4000.0):
        S = _fb() @ magnitude(np.sin(2 * np.pi * f0 * t))
        peaks.append(int(np.argmax(S.mean(axis=1))))
    good = peaks == sorted(peaks) and len(set(peaks)) == 3
    ok &= good
    print('mel peak index rises with frequency %s: %s' % (peaks, good))

    img = spectrogram_image(np.sin(2 * np.pi * 1000 * t), 'mel')
    good = img.shape == (OUT_SIZE, OUT_SIZE) and img.dtype == np.uint8
    ok &= good
    print('mel image %s %s: %s' % (img.shape, img.dtype, good))
    img = spectrogram_image(np.sin(2 * np.pi * 1000 * t), 'linear')
    good = img.shape == (OUT_SIZE, OUT_SIZE)
    ok &= good
    print('linear image %s: %s' % (img.shape, good))

    # silence must not produce NaN
    q = spectrogram_image(np.zeros(sr), 'mel')
    good = np.isfinite(q).all()
    ok &= good
    print('silence gives finite output: %s' % good)

    # a very short clip must still produce a full size image
    q = spectrogram_image(np.zeros(100), 'linear')
    ok &= q.shape == (OUT_SIZE, OUT_SIZE)
    print('100 sample clip -> %s' % (q.shape,))

    print()
    print('SELFTEST %s' % ('PASSED' if ok else 'FAILED'))
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
