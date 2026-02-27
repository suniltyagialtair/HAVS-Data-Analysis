"""
Event Detector
===============
Merges per-window results into discrete acoustic events,
classifies each event, and builds the assessment string.
"""

import numpy as np
import re
try:
    from .config import (
        SIGNIFICANT_DURATION_MIN, SIGNIFICANT_RMS_FACTOR,
        VESSEL_SCORE_THRESHOLD,
    )
except ImportError:
    from config import (
        SIGNIFICANT_DURATION_MIN, SIGNIFICANT_RMS_FACTOR,
        VESSEL_SCORE_THRESHOLD,
    )


def classify_events(events, all_windows):
    """Add assessment and classification to each event.

    Enriches events with: assessment, blade_rate, shaft_rate,
    vessel_class, demon_confidence, tonal_freqs, remarks.
    """
    for ev in events:
        windows = all_windows[ev['start_idx']:ev['end_idx']]

        # Aggregate LOFAR features
        ev['mean_tonals'] = float(np.mean([w.get('n_tonals', 0) for w in windows]))
        ev['max_tonals'] = int(max(w.get('n_tonals', 0) for w in windows))
        ev['mean_tonal_energy'] = float(np.mean([w.get('tonal_energy', 0) for w in windows]))

        # Aggregate vessel score
        scores = [w.get('vessel_score', 0) for w in windows]
        ev['mean_vessel_score'] = float(np.mean(scores))
        ev['max_vessel_score'] = float(max(scores))

        # Best DEMON result (highest score among elevated-RMS windows)
        demon_windows = [w for w in windows if w.get('demon_score', 0) > 0]
        if demon_windows:
            best_demon = max(demon_windows, key=lambda w: w['demon_score'])
            ev['blade_rate'] = best_demon.get('blade_rate_hz', 0)
            ev['shaft_rate'] = best_demon.get('shaft_rate_hz', 0)
            ev['est_n_blades'] = best_demon.get('est_n_blades', 0)
            ev['est_rpm'] = best_demon.get('est_rpm', 0)
            ev['demon_score'] = best_demon.get('demon_score', 0)
            ev['demon_confidence'] = best_demon.get('demon_confidence', 'LOW')
            ev['vessel_class'] = best_demon.get('vessel_class', '')
        else:
            ev['blade_rate'] = 0
            ev['shaft_rate'] = 0
            ev['est_n_blades'] = 0
            ev['est_rpm'] = 0
            ev['demon_score'] = 0
            ev['demon_confidence'] = ''
            ev['vessel_class'] = ''

        # Collect top tonal frequencies across event
        all_freqs = []
        event_sessions = set()
        for w in windows:
            all_freqs.extend(w.get('tonal_freqs', []))
            if w.get('session'):
                event_sessions.add(w['session'])
        ev['sessions'] = sorted(event_sessions)
        # Most common frequencies (rounded to integers)
        if all_freqs:
            from collections import Counter
            freq_counts = Counter(round(f) for f in all_freqs)
            ev['top_tonal_freqs'] = [f for f, _ in freq_counts.most_common(5)]
            # Full list for harmonic checking in HAVS writer
            ev['all_tonal_freqs'] = [f for f, _ in freq_counts.most_common(20)]
        else:
            ev['top_tonal_freqs'] = []
            ev['all_tonal_freqs'] = []

        # Assessment
        is_sig = (ev['dur_min'] >= SIGNIFICANT_DURATION_MIN or
                  ev['rms_ratio'] >= SIGNIFICANT_RMS_FACTOR)
        has_tonals = ev['mean_vessel_score'] >= VESSEL_SCORE_THRESHOLD

        if is_sig:
            if has_tonals and ev['dur_min'] >= 8:
                ev['assessment'] = 'CONFIRMED VESSEL'
            elif has_tonals:
                ev['assessment'] = 'PROBABLE VESSEL'
            elif ev['rms_ratio'] >= 10:
                ev['assessment'] = 'HIGH ENERGY (no tonals)'
            else:
                ev['assessment'] = 'ELEVATED ACTIVITY'
        else:
            ev['assessment'] = 'brief transient'

        # Build remarks string
        ev['remarks'] = _build_remarks(ev)

    return events


def _build_remarks(ev):
    """Build human-readable remarks string for HAVS output."""
    parts = [ev['assessment']]

    if ev.get('demon_confidence'):
        parts.append(f"DEMON: {ev['demon_confidence']} confidence")

    if ev.get('est_n_blades') and ev['est_n_blades'] > 0:
        parts.append(f"{ev['est_n_blades']}-blade {int(ev['est_rpm'])} RPM")

    if ev.get('rms_ratio'):
        parts.append(f"{ev['rms_ratio']:.0f}x background")

    return '; '.join(parts)
