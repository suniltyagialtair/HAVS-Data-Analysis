"""
Stage 2: LOFAR Spectrogram & Tonal Analysis
=============================================
STFT spectrogram with background normalisation and tonal detection.
Provides narrowband spectral characterisation for vessel classification.
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram, find_peaks
try:
    from .config import (
        STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT, FREQ_MAX_HZ,
        BACKGROUND_PERCENTILE, TONAL_THRESHOLD_DB, TONAL_MIN_PERSIST_SEC,
        TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ, TONAL_PEAK_PROMINENCE_DB,
        TONAL_PEAK_MIN_DISTANCE_HZ, TONAL_BAND_WIDTH_HZ,
        ANALYSIS_WINDOW_SEC, ANALYSIS_HOP_SEC,
    )
except ImportError:
    from config import (
        STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT, FREQ_MAX_HZ,
        BACKGROUND_PERCENTILE, TONAL_THRESHOLD_DB, TONAL_MIN_PERSIST_SEC,
        TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ, TONAL_PEAK_PROMINENCE_DB,
        TONAL_PEAK_MIN_DISTANCE_HZ, TONAL_BAND_WIDTH_HZ,
        ANALYSIS_WINDOW_SEC, ANALYSIS_HOP_SEC,
    )


def compute_stft(audio, sr):
    """Compute STFT spectrogram.
    Returns freqs (Hz), times (seconds), Sxx (power spectral density).
    """
    nperseg = int(STFT_WINDOW_SEC * sr)
    noverlap = int(nperseg * STFT_OVERLAP_FRAC)
    nfft = nperseg * STFT_NFFT_MULT

    freqs, times, Sxx = scipy_spectrogram(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, mode='psd', scaling='density'
    )
    return freqs, times, Sxx


def normalise_spectrogram(freqs, times, Sxx):
    """Normalise spectrogram against background (median spectrum).
    Returns normalised spectrogram in dB above background, and background spectrum.
    """
    background = np.percentile(Sxx, BACKGROUND_PERCENTILE, axis=1, keepdims=True)
    background = np.maximum(background, 1e-20)
    spec_norm = Sxx / background
    spec_db = 10 * np.log10(np.maximum(spec_norm, 1e-10))
    return spec_db, background.squeeze()


def detect_tonals_in_window(spec_db_window, freqs):
    """Detect tonal peaks in a single normalised spectrum.
    Returns list of {freq_hz, strength_db, prominence_db}.
    """
    mask = (freqs >= TONAL_FREQ_MIN_HZ) & (freqs <= TONAL_FREQ_MAX_HZ)
    f_sub = freqs[mask]
    s_sub = spec_db_window[mask]

    if len(s_sub) < 10:
        return []

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 0.125
    min_distance = max(1, int(TONAL_PEAK_MIN_DISTANCE_HZ / freq_res))

    peaks, properties = find_peaks(
        s_sub,
        height=TONAL_THRESHOLD_DB,
        prominence=TONAL_PEAK_PROMINENCE_DB,
        distance=min_distance,
    )

    tonals = []
    for i, pk in enumerate(peaks):
        tonals.append({
            'freq_hz': float(f_sub[pk]),
            'strength_db': float(s_sub[pk]),
            'prominence_db': float(properties['prominences'][i]),
        })

    return sorted(tonals, key=lambda t: t['strength_db'], reverse=True)


def detect_persistent_tonals(spec_db, freqs, times):
    """Track tonals across time to find persistent narrowband lines.
    Groups nearby frequency bins into bands, requires sustained presence.
    Returns list of tracked tonals sorted by strength.
    """
    freq_mask = (freqs >= TONAL_FREQ_MIN_HZ) & (freqs <= TONAL_FREQ_MAX_HZ)
    f_sub = freqs[freq_mask]
    s_sub = spec_db[freq_mask, :]

    if len(times) < 2 or len(f_sub) < 2:
        return []

    dt = times[1] - times[0]
    min_frames = max(1, int(TONAL_MIN_PERSIST_SEC / dt))
    freq_res = freqs[1] - freqs[0]

    # Reduce to bands
    band_width_bins = max(1, int(TONAL_BAND_WIDTH_HZ / freq_res))
    n_bands = len(f_sub) // band_width_bins

    band_freqs = []
    band_spec = np.zeros((n_bands, s_sub.shape[1]))
    for bi in range(n_bands):
        start_bin = bi * band_width_bins
        end_bin = min(start_bin + band_width_bins, len(f_sub))
        band_spec[bi, :] = np.max(s_sub[start_bin:end_bin, :], axis=0)
        peak_bin = start_bin + np.argmax(np.mean(s_sub[start_bin:end_bin, :], axis=1))
        band_freqs.append(float(f_sub[peak_bin]))

    # Find sustained exceedances
    above = band_spec > TONAL_THRESHOLD_DB
    persistent = []

    for bi in range(n_bands):
        row = above[bi, :]
        start_idx = None
        for ti in range(len(row)):
            if row[ti] and start_idx is None:
                start_idx = ti
            elif not row[ti] and start_idx is not None:
                if ti - start_idx >= min_frames:
                    persistent.append({
                        'freq_hz': band_freqs[bi],
                        'start_sec': float(times[start_idx]),
                        'end_sec': float(times[ti - 1]),
                        'duration_sec': float(times[ti - 1] - times[start_idx]),
                        'mean_db': float(np.mean(band_spec[bi, start_idx:ti])),
                        'max_db': float(np.max(band_spec[bi, start_idx:ti])),
                    })
                start_idx = None
        if start_idx is not None and len(row) - start_idx >= min_frames:
            persistent.append({
                'freq_hz': band_freqs[bi],
                'start_sec': float(times[start_idx]),
                'end_sec': float(times[-1]),
                'duration_sec': float(times[-1] - times[start_idx]),
                'mean_db': float(np.mean(band_spec[bi, start_idx:])),
                'max_db': float(np.max(band_spec[bi, start_idx:])),
            })

    return sorted(persistent, key=lambda t: t['mean_db'], reverse=True)


def process_lofar_window(audio_chunk, sr, spec_db_chunk, freqs):
    """Process a single 30-second analysis window through LOFAR.
    Returns dict of LOFAR features.
    """
    # Average normalised spectrum across window
    mean_spec = np.mean(spec_db_chunk, axis=1)

    # Detect tonals
    tonals = detect_tonals_in_window(mean_spec, freqs)
    n_tonals = len(tonals)
    tonal_energy = sum(t['strength_db'] for t in tonals) if tonals else 0.0
    strongest_freq = tonals[0]['freq_hz'] if tonals else 0.0
    strongest_db = tonals[0]['strength_db'] if tonals else 0.0
    tonal_freqs = [t['freq_hz'] for t in tonals[:5]]

    # Band energy
    band_masks = {
        'low_5_20': (freqs >= 5) & (freqs < 20),
        'mid_20_80': (freqs >= 20) & (freqs < 80),
        'high_80_256': (freqs >= 80) & (freqs <= 256),
    }
    band_energy = {}
    for name, bmask in band_masks.items():
        band_energy[name] = float(np.mean(mean_spec[bmask])) if bmask.any() else 0.0

    return {
        'n_tonals': n_tonals,
        'tonal_energy': round(tonal_energy, 2),
        'strongest_tonal_hz': round(strongest_freq, 2),
        'strongest_tonal_db': round(strongest_db, 2),
        'tonal_freqs': tonal_freqs,
        'band_low_5_20_db': round(band_energy['low_5_20'], 2),
        'band_mid_20_80_db': round(band_energy['mid_20_80'], 2),
        'band_high_80_256_db': round(band_energy['high_80_256'], 2),
    }


def compute_vessel_score(lofar_features, rms_energy, rms_mean):
    """Compute soft-gated vessel score.
    Gate ramps 0→1 as RMS goes from 0→2× background mean.
    No hard caps — full dynamic range.
    """
    rms_ratio = rms_energy / rms_mean if rms_mean > 0 else 0
    gate_weight = min(rms_ratio / 2.0, 1.0)

    gated_n_tonals = lofar_features['n_tonals'] * gate_weight
    gated_tonal_energy = lofar_features['tonal_energy'] * gate_weight

    score = (
        np.log10(1 + rms_ratio) *
        (1 + gated_n_tonals) *
        (1 + np.log10(1 + gated_tonal_energy))
    )
    return round(float(score), 2)
