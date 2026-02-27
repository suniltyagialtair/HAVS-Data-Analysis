"""
Stage 1: RMS Energy Detection
==============================
Computes broadband RMS energy per analysis window.
This is the primary vessel detection layer — works reliably
regardless of sample rate or acoustic environment.
"""

import numpy as np
try:
    from .config import (
        ANALYSIS_WINDOW_SEC, ANALYSIS_HOP_SEC,
        RMS_SIGMA_THRESHOLD, EVENT_MERGE_RADIUS, EVENT_MIN_WINDOWS,
        SIGNIFICANT_DURATION_MIN, SIGNIFICANT_RMS_FACTOR,
    )
except ImportError:
    from config import (
        ANALYSIS_WINDOW_SEC, ANALYSIS_HOP_SEC,
        RMS_SIGMA_THRESHOLD, EVENT_MERGE_RADIUS, EVENT_MIN_WINDOWS,
        SIGNIFICANT_DURATION_MIN, SIGNIFICANT_RMS_FACTOR,
    )


def compute_rms_windows(audio, sr, file_start_ts):
    """Compute RMS energy for each analysis window.

    Returns list of dicts: [{timestamp, time_offset_sec, window_idx, rms_energy}, ...]
    """
    from datetime import timedelta

    window_samples = int(ANALYSIS_WINDOW_SEC * sr)
    hop_samples = int(ANALYSIS_HOP_SEC * sr)
    n_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)

    results = []
    for wi in range(n_windows):
        start_sample = wi * hop_samples
        end_sample = start_sample + window_samples
        if end_sample > len(audio):
            break

        chunk = audio[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        offset_sec = start_sample / sr
        ts = file_start_ts + timedelta(seconds=offset_sec)

        results.append({
            'window_idx': wi,
            'time_offset_sec': round(offset_sec, 2),
            'timestamp': ts,
            'rms_energy': rms,
            'audio_chunk': chunk,  # kept for Stage 2/3, dropped before output
        })

    return results


def compute_rms_statistics(all_windows):
    """Compute global RMS statistics across all windows.

    Returns dict with mean, std, threshold, dynamic_range, day/night ratio.
    """
    rms_values = np.array([w['rms_energy'] for w in all_windows])
    rms_mean = float(np.mean(rms_values))
    rms_std = float(np.std(rms_values))
    rms_thresh = rms_mean + RMS_SIGMA_THRESHOLD * rms_std

    rms_min = float(np.min(rms_values))
    rms_max = float(np.max(rms_values))
    dynamic_range = rms_max / rms_min if rms_min > 0 else 0

    # Day/night
    day_rms = [w['rms_energy'] for w in all_windows if 6 <= w['timestamp'].hour < 18]
    night_rms = [w['rms_energy'] for w in all_windows if w['timestamp'].hour >= 18 or w['timestamp'].hour < 6]
    day_mean = float(np.mean(day_rms)) if day_rms else 0
    night_mean = float(np.mean(night_rms)) if night_rms else 0
    day_night_ratio = day_mean / night_mean if night_mean > 0 else 0

    n_elevated = int(np.sum(rms_values > rms_thresh))

    return {
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'rms_threshold': rms_thresh,
        'rms_min': rms_min,
        'rms_max': rms_max,
        'dynamic_range': dynamic_range,
        'day_mean': day_mean,
        'night_mean': night_mean,
        'day_night_ratio': day_night_ratio,
        'n_elevated': n_elevated,
        'n_total': len(rms_values),
        'pct_elevated': 100 * n_elevated / len(rms_values),
    }


def detect_events(all_windows, rms_stats):
    """Detect acoustic events using RMS energy threshold with temporal merging.

    Returns list of event dicts.
    """
    from datetime import timedelta

    rms_mean = rms_stats['rms_mean']
    rms_thresh = rms_stats['rms_threshold']

    # Flag elevated windows
    elevated_idx = set()
    for i, w in enumerate(all_windows):
        if w['rms_energy'] > rms_thresh:
            elevated_idx.add(i)

    # Expand each elevated window by ±MERGE_RADIUS
    expanded = set()
    for idx in elevated_idx:
        for offset in range(-EVENT_MERGE_RADIUS, EVENT_MERGE_RADIUS + 1):
            if 0 <= idx + offset < len(all_windows):
                expanded.add(idx + offset)

    # Group contiguous expanded windows into events
    in_event = [i in expanded for i in range(len(all_windows))]
    events = []
    current_start = None

    for i in range(len(all_windows)):
        if in_event[i] and current_start is None:
            current_start = i
        elif not in_event[i] and current_start is not None:
            if i - current_start >= EVENT_MIN_WINDOWS:
                events.append(_build_event(all_windows, current_start, i, rms_mean))
            current_start = None

    # Handle final event
    if current_start is not None and len(all_windows) - current_start >= EVENT_MIN_WINDOWS:
        events.append(_build_event(all_windows, current_start, len(all_windows), rms_mean))

    return events


def _build_event(all_windows, start_idx, end_idx, rms_mean):
    """Build event dict from a range of windows."""
    windows = all_windows[start_idx:end_idx]
    start_ts = windows[0]['timestamp']
    end_ts = windows[-1]['timestamp']
    dur_sec = (end_ts - start_ts).total_seconds()

    rms_values = [w['rms_energy'] for w in windows]
    peak_rms = max(rms_values)
    peak_idx = rms_values.index(peak_rms)
    peak_time = windows[peak_idx]['timestamp']

    return {
        'start': start_ts,
        'end': end_ts,
        'dur_min': dur_sec / 60,
        'n_windows': len(windows),
        'start_idx': start_idx,
        'end_idx': end_idx,
        'peak_rms': peak_rms,
        'peak_time': peak_time,
        'mean_rms': float(np.mean(rms_values)),
        'rms_ratio': peak_rms / rms_mean if rms_mean > 0 else 0,
    }
