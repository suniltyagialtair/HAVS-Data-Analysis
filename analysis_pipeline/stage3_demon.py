"""
Stage 3: DEMON Blade-Rate Extraction (Conditional)
====================================================
Hilbert envelope → FFT modulation spectrum → blade-rate + harmonic detection.
Only runs on windows flagged by Stage 1 (elevated RMS energy).

NOTE: Classic DEMON requires cavitation noise (20–50 kHz band).
At 512 Hz sampling (Nyquist 256 Hz), cavitation is unavailable.
Results are derived from low-frequency envelope modulation and
should be treated as indicative. LOFAR tonals and RMS energy
are the primary detection methods.
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfilt
try:
    from .config import (
        DEMON_BLADE_RATE_MIN_HZ, DEMON_BLADE_RATE_MAX_HZ,
        DEMON_HARMONIC_TOL_HZ, DEMON_SHAFT_TOL_HZ,
        DEMON_MIN_PROMINENCE,
        DEMON_SCORE_2X_BONUS, DEMON_SCORE_3X_BONUS,
        DEMON_SCORE_4X_BONUS, DEMON_SCORE_SHAFT_BONUS,
        DEMON_SCORE_NOISE_PENALTY, DEMON_SCORE_THRESHOLD,
        DEMON_HIGH_CONF_MIN_HARMONICS, DEMON_HIGH_CONF_MIN_RMS_FACTOR,
    )
except ImportError:
    from config import (
        DEMON_BLADE_RATE_MIN_HZ, DEMON_BLADE_RATE_MAX_HZ,
        DEMON_HARMONIC_TOL_HZ, DEMON_SHAFT_TOL_HZ,
        DEMON_MIN_PROMINENCE,
        DEMON_SCORE_2X_BONUS, DEMON_SCORE_3X_BONUS,
        DEMON_SCORE_4X_BONUS, DEMON_SCORE_SHAFT_BONUS,
        DEMON_SCORE_NOISE_PENALTY, DEMON_SCORE_THRESHOLD,
        DEMON_HIGH_CONF_MIN_HARMONICS, DEMON_HIGH_CONF_MIN_RMS_FACTOR,
    )


def _bandpass_filter(audio, sr, low_hz, high_hz, order=4):
    """Apply Butterworth bandpass filter using SOS for stability."""
    nyq = sr / 2
    low = max(low_hz / nyq, 0.01)
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        return audio
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio)


def compute_demon_spectrum(audio_chunk, sr, bp_low=30, bp_high=180):
    """Compute DEMON modulation spectrum from audio chunk.

    Steps:
    1. Bandpass filter
    2. Hilbert envelope extraction
    3. AC-coupling (mean removal)
    4. Zero-padded FFT

    Returns freqs (Hz), spectrum magnitude.
    """
    # Clamp bandpass to valid range for sample rate
    nyq = sr / 2
    bp_high = min(bp_high, nyq * 0.95)
    bp_low = max(bp_low, 5)
    if bp_low >= bp_high:
        bp_low = 5
        bp_high = nyq * 0.5

    filtered = _bandpass_filter(audio_chunk, sr, bp_low, bp_high)

    # Hilbert envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # AC-couple
    envelope_ac = envelope - np.mean(envelope)

    # Zero-padded FFT
    n_fft = len(envelope_ac) * 4
    fft_result = np.fft.rfft(envelope_ac, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    spectrum = np.abs(fft_result) / len(envelope_ac)

    return freqs, spectrum


def detect_blade_rate(freqs, spectrum):
    """Detect blade rate, harmonics, and shaft rate from DEMON spectrum.

    Returns dict with blade_rate_hz, shaft_rate_hz, est_n_blades,
    harmonics_found, confidence, demon_score.
    """
    # Restrict to blade-rate search range
    mask = (freqs >= DEMON_BLADE_RATE_MIN_HZ) & (freqs <= DEMON_BLADE_RATE_MAX_HZ)
    f_sub = freqs[mask]
    s_sub = spectrum[mask]

    if len(s_sub) < 5:
        return _null_result()

    # Find peaks — adaptive prominence (median-relative, not absolute)
    from scipy.signal import find_peaks as _fp
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    min_dist = max(1, int(1.0 / freq_res))

    median_level = float(np.median(s_sub))
    adaptive_prom = max(median_level * 1.5, 1e-6)

    peaks, props = _fp(s_sub, prominence=adaptive_prom, distance=min_dist)
    if len(peaks) == 0:
        return _null_result()

    # Sort by prominence
    order = np.argsort(props['prominences'])[::-1]
    peaks = peaks[order]

    # Try each candidate as blade rate
    best = _null_result()
    best_score = 0

    for pk in peaks[:10]:  # check top 10 candidates
        br_hz = float(f_sub[pk])
        prominence = float(props['prominences'][np.where(order == np.where(peaks == pk)[0][0])[0][0]])

        # Check harmonics
        harmonics_found = 0
        for mult in [2, 3, 4]:
            harmonic_hz = br_hz * mult
            idx = np.argmin(np.abs(freqs - harmonic_hz))
            if abs(freqs[idx] - harmonic_hz) <= DEMON_HARMONIC_TOL_HZ:
                if spectrum[idx] > np.mean(spectrum) * 1.5:
                    harmonics_found += 1

        # Check shaft rate sub-harmonics (BR/N for N=3,4,5)
        shaft_found = False
        best_shaft = 0
        best_n_blades = 0
        for n_blades in [3, 4, 5]:
            shaft_hz = br_hz / n_blades
            idx = np.argmin(np.abs(freqs - shaft_hz))
            if abs(freqs[idx] - shaft_hz) <= DEMON_SHAFT_TOL_HZ:
                if spectrum[idx] > np.mean(spectrum) * 1.2:
                    shaft_found = True
                    best_shaft = shaft_hz
                    best_n_blades = n_blades
                    break

        # Score
        score = prominence * 2
        if harmonics_found >= 1:
            score += DEMON_SCORE_2X_BONUS
        if harmonics_found >= 2:
            score += DEMON_SCORE_3X_BONUS
        if harmonics_found >= 3:
            score += DEMON_SCORE_4X_BONUS
        if shaft_found:
            score += DEMON_SCORE_SHAFT_BONUS

        noise_std = float(np.std(spectrum[spectrum < np.percentile(spectrum, 75)]))
        score -= DEMON_SCORE_NOISE_PENALTY * noise_std

        if score > best_score:
            best_score = score
            best = {
                'blade_rate_hz': round(br_hz, 2),
                'shaft_rate_hz': round(best_shaft, 2) if shaft_found else round(br_hz / 3, 2),
                'est_n_blades': best_n_blades if shaft_found else 0,
                'harmonics_found': harmonics_found,
                'demon_score': round(score, 2),
                'peak_prominence': round(prominence, 3),
            }

    # Estimate RPM if we have blade count
    if best['est_n_blades'] > 0:
        best['est_rpm'] = round(best['shaft_rate_hz'] * 60, 0)
    else:
        # Guess 3-blade
        best['est_n_blades'] = 3
        best['shaft_rate_hz'] = round(best['blade_rate_hz'] / 3, 2)
        best['est_rpm'] = round(best['shaft_rate_hz'] * 60, 0)

    return best


def classify_demon_confidence(demon_result, rms_ratio):
    """Classify DEMON result confidence as HIGH or LOW."""
    if (demon_result['harmonics_found'] >= DEMON_HIGH_CONF_MIN_HARMONICS and
            rms_ratio >= DEMON_HIGH_CONF_MIN_RMS_FACTOR):
        return 'HIGH'
    return 'LOW'


def classify_vessel(shaft_rate_hz):
    """Classify vessel type by shaft rate."""
    try:
        from .config import CLASS_LARGE_VESSEL_SR_MAX, CLASS_FISHING_SR_MAX
    except ImportError:
        from config import CLASS_LARGE_VESSEL_SR_MAX, CLASS_FISHING_SR_MAX
    if shaft_rate_hz <= CLASS_LARGE_VESSEL_SR_MAX:
        return 'large_vessel'
    elif shaft_rate_hz <= CLASS_FISHING_SR_MAX:
        return 'fishing'
    else:
        return 'small_craft'


def _null_result():
    return {
        'blade_rate_hz': 0,
        'shaft_rate_hz': 0,
        'est_n_blades': 0,
        'harmonics_found': 0,
        'demon_score': 0,
        'peak_prominence': 0,
        'est_rpm': 0,
    }
