"""
Spectrogram Writer
===================
Image generation module for the Oravont Hydrophone Analysis Pipeline.
Produces 8 types of output figure using the deep-ocean visual aesthetic.

Functions:
    save_file_spectrogram    — Per-session LOFAR spectrogram
    save_event_spectrograms  — 3-panel event figures for high-energy events
    save_timeline            — 4-panel 48-hour overview
    save_distributions       — 4-panel statistical distributions
    save_events_detail       — Top 8 events detail view
    save_score_analysis      — DEMON score histogram + decomposition
    save_pipeline_diagram    — Static 3-stage pipeline block diagram
    save_demon_illustration  — Static 4-panel DEMON signal processing illustration

© Oravont Systems LLP — Confidential
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

try:
    from .config import (
        STYLE, FREQ_MAX_HZ, EVENT_PADDING_MIN, EVENT_SPEC_MIN_RMS_FACTOR,
        SIGNIFICANT_RMS_FACTOR, SIGNIFICANT_DURATION_MIN, RMS_SIGMA_THRESHOLD,
        DEMON_SCORE_2X_BONUS, DEMON_SCORE_3X_BONUS, DEMON_SCORE_4X_BONUS,
        DEMON_SCORE_SHAFT_BONUS, DEMON_SCORE_THRESHOLD,
        STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT,
    )
    from . import stage2_lofar
except ImportError:
    from config import (
        STYLE, FREQ_MAX_HZ, EVENT_PADDING_MIN, EVENT_SPEC_MIN_RMS_FACTOR,
        SIGNIFICANT_RMS_FACTOR, SIGNIFICANT_DURATION_MIN, RMS_SIGMA_THRESHOLD,
        DEMON_SCORE_2X_BONUS, DEMON_SCORE_3X_BONUS, DEMON_SCORE_4X_BONUS,
        DEMON_SCORE_SHAFT_BONUS, DEMON_SCORE_THRESHOLD,
        STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT,
    )
    import stage2_lofar


# =====================================================================
# COMMON STYLE HELPERS
# =====================================================================

def _style_axes(ax):
    """Apply deep-ocean styling to axes."""
    ax.set_facecolor(STYLE['bg'])
    ax.tick_params(colors=STYLE['text'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(STYLE['box_edge'])
    ax.grid(True, alpha=0.2, color=STYLE['box_edge'])


def _style_figure(fig):
    """Set figure background."""
    fig.patch.set_facecolor(STYLE['bg'])


def _save_fig(fig, path, dpi=150):
    """Save figure and prepare it for interactive display."""
    fig.savefig(path, dpi=dpi, facecolor=STYLE['bg'], bbox_inches='tight')
    if fig.canvas.manager:
        fig.canvas.manager.set_window_title(os.path.basename(path))
    print(f'  Saved: {path}')


# =====================================================================
# 1. PER-SESSION LOFAR SPECTROGRAM
# =====================================================================

def save_file_spectrogram(stft_data, output_dir):
    """Generate per-session LOFAR spectrogram with persistent tonals overlay.

    Parameters:
        stft_data: dict with keys freqs, times, spec_db, file_start, session, persistent
        output_dir: directory to save output PNG
    """
    freqs = stft_data['freqs']
    times = stft_data['times']
    spec_db = stft_data['spec_db']
    session = stft_data['session']
    file_start = stft_data['file_start']
    persistent = stft_data.get('persistent', [])

    fig, ax = plt.subplots(figsize=(20, 6), facecolor=STYLE['bg'])
    _style_axes(ax)
    ax.grid(False)

    # Frequency mask
    freq_mask = freqs <= FREQ_MAX_HZ
    f_plot = freqs[freq_mask]
    s_plot = spec_db[freq_mask, :]

    # Time axis as datetime
    time_dates = [file_start + timedelta(seconds=float(t)) for t in times]

    im = ax.pcolormesh(
        time_dates, f_plot, s_plot,
        cmap=STYLE['cmap'], vmin=-3, vmax=20, shading='auto'
    )

    # Overlay persistent tonals
    for tonal in persistent:
        t_start = file_start + timedelta(seconds=tonal['start_sec'])
        t_end = file_start + timedelta(seconds=tonal['end_sec'])
        freq = tonal['freq_hz']
        ax.plot([t_start, t_end], [freq, freq],
                color=STYLE['green'], linewidth=1.5, alpha=0.7)

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('dB above background', color=STYLE['text'], fontsize=10)
    cbar.ax.tick_params(colors=STYLE['text'])

    ax.set_ylabel('Frequency (Hz)', color=STYLE['text'], fontsize=11)
    ax.set_xlabel('Time (IST)', color=STYLE['text'], fontsize=11)
    ax.set_ylim(0, FREQ_MAX_HZ)
    tz_ist = datetime.timezone(timedelta(hours=5, minutes=30))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=tz_ist))

    title_date = file_start.strftime('%d %b %Y %H:%M')
    n_tonals = len(persistent)
    ax.set_title(
        f'LOFAR Spectrogram — Session {session} — {title_date} — {n_tonals} persistent tonals',
        color=STYLE['text'], fontsize=13, fontweight='bold', pad=12
    )

    # Format session ID for filename
    if isinstance(session, int):
        session_str = f"{session:03d}"
    elif session is not None:
        session_str = str(session)
    else:
        session_str = "unknown"

    plt.tight_layout()
    path = os.path.join(output_dir, f'lofar_session_{session_str}.png')
    _save_fig(fig, path, dpi=150)
    return path


# =====================================================================
# 2. EVENT SPECTROGRAMS (3-panel)
# =====================================================================

def save_event_spectrograms(events, all_files, output_dir, rms_stats, all_windows=None, audio_data=None, sr=None):
    """Generate 3-panel event spectrograms for high-energy events.

    Panels:
      1. LOFAR spectrogram with persistent tonals overlaid
      2. RMS energy profile with threshold line
      3. Tonal frequency lines

    Only generates for events with rms_ratio >= EVENT_SPEC_MIN_RMS_FACTOR.

    Parameters:
        events: list of event dicts
        all_files: list of file_info dicts with 'path', 'start_ts', 'session'
        output_dir: directory to save PNGs
        rms_stats: dict with rms_mean, rms_threshold
        all_windows: list of all window dicts (for RMS profile)
        audio_data: optional pre-loaded numpy array of audio for entire session
        sr: optional sample rate for audio_data

    Returns list of generated file paths.
    """
    paths = []
    big_events = [e for e in events if e.get('rms_ratio', 0) >= EVENT_SPEC_MIN_RMS_FACTOR]

    if not big_events:
        print(f'  No events above {EVENT_SPEC_MIN_RMS_FACTOR}× threshold for event spectrograms')
        return paths

    print(f'  Generating event spectrograms for {len(big_events)} events...')

    for ev in sorted(big_events, key=lambda e: e['start']):
        try:
            path = _plot_single_event(ev, all_files, output_dir, rms_stats, all_windows, audio_data, sr)
            if path:
                paths.append(path)
        except Exception as exc:
            print(f'    WARNING: Failed to generate event spectrogram for '
                  f'{ev["start"].strftime("%d %b %H:%M")}: {exc}')

    return paths


def save_demon_plots(events, all_files, output_dir, rms_stats, audio_data=None, sr=None):
    """Generate separate DEMON modulation spectrum plots for high-energy events.

    Each plot shows the DEMON envelope spectrum (0–15 Hz) computed from a
    30-second window centred on the event's peak RMS, with blade rate,
    harmonics, and shaft rate marked.

    Parameters:
        events: list of event dicts
        all_files: list of file_info dicts
        output_dir: directory to save PNGs
        rms_stats: dict with rms_mean, rms_threshold
        audio_data: optional pre-loaded numpy array of audio for entire session
        sr: optional sample rate for audio_data

    Returns list of generated file paths.
    """
    try:
        from stage3_demon import compute_demon_spectrum, detect_blade_rate
    except ImportError:
        try:
            from .stage3_demon import compute_demon_spectrum, detect_blade_rate
        except ImportError:
            print('  WARNING: stage3_demon not available, skipping DEMON plots')
            return []

    paths = []
    big_events = [e for e in events if e.get('rms_ratio', 0) >= EVENT_SPEC_MIN_RMS_FACTOR]

    if not big_events:
        return paths

    print(f'  Generating DEMON plots for {len(big_events)} events...')

    for ev in sorted(big_events, key=lambda e: e['start']):
        try:
            path = _plot_single_demon(ev, all_files, output_dir, rms_stats,
                                       compute_demon_spectrum, detect_blade_rate, audio_data, sr)
            if path:
                paths.append(path)
        except Exception as exc:
            print(f'    WARNING: Failed to generate DEMON plot for '
                  f'{ev["start"].strftime("%d %b %H:%M")}: {exc}')

    return paths


def _plot_single_demon(ev, all_files, output_dir, rms_stats,
                        compute_demon_spectrum, detect_blade_rate, audio_data=None, sr=None):
    """Generate a single DEMON modulation spectrum plot for an event."""
    import soundfile as sf

    # Load ~60s of audio around the peak
    peak_time = ev.get('peak_time', ev['start'] + (ev['end'] - ev['start']) / 2)
    load_start = peak_time - timedelta(seconds=30)
    load_end = peak_time + timedelta(seconds=30)

    if audio_data is not None and sr is not None:
        # Assume audio_data starts at ev['session_start'] or similar. 
        # Since main.py passes audio_data for the WHOLE session, and we have session times.
        # We need the base start time of audio_data. 
        # For simplicity, we can use ev['start'] as a reference if we know it's within audio_data.
        # Let's assume ev['start'] is relative to some session start.
        # Actually, if we have ev['start'] as datetime, and we know audio_data start as datetime.
        # Let's use a simplified approach: use ev['peak_time'] and assume audio_data 
        # is the entire session audio. We need the session start time.
        
        # We'll need ev['session_start_ts'] to be present.
        session_start = ev.get('session_start_ts', ev['start'] - timedelta(seconds=ev.get('time_offset_sec', 0)))
        
        peak_offset = (peak_time - session_start).total_seconds()
        peak_sample = int(peak_offset * sr)
        window_samples = int(30 * sr)
        
        peak_start = max(0, peak_sample - window_samples // 2)
        peak_end = min(len(audio_data), peak_start + window_samples)
        peak_chunk = audio_data[peak_start:peak_end]
        combined_sr = sr
    else:
        overlapping = []
        for fi in all_files:
            if fi['start_ts'] is None:
                continue
            file_end_approx = fi['start_ts'] + timedelta(hours=1)
            if fi['start_ts'] <= load_end and file_end_approx >= load_start:
                overlapping.append(fi)

        if not overlapping:
            return None

        overlapping = sorted(overlapping, key=lambda f: f['start_ts'])

        # Load and extract the 30s peak window
        audio_segments = []
        combined_start = overlapping[0]['start_ts']
        combined_sr = None

        for fi in overlapping:
            try:
                data, sr_file = sf.read(fi['path'], dtype='float64')
                if data.ndim > 1:
                    data = data[:, 0]
                combined_sr = sr_file
                audio_segments.append(data)
            except Exception:
                continue

        if not audio_segments or combined_sr is None:
            return None

        segment = np.concatenate(audio_segments)
        peak_offset = (peak_time - combined_start).total_seconds()
        peak_sample = int(peak_offset * combined_sr)
        window_samples = int(30 * combined_sr)

        peak_start = max(0, peak_sample - window_samples // 2)
        peak_end = min(len(segment), peak_start + window_samples)
        peak_chunk = segment[peak_start:peak_end]

    if len(peak_chunk) < combined_sr * 5:
        return None

    # Compute DEMON spectrum
    demon_freqs, demon_spectrum = compute_demon_spectrum(
        peak_chunk, combined_sr, bp_low=30, bp_high=min(180, combined_sr * 0.45)
    )
    demon_result = detect_blade_rate(demon_freqs, demon_spectrum)

    if demon_result.get('blade_rate_hz', 0) <= 0:
        return None

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor=STYLE['bg'])
    _style_axes(ax)

    demon_mask = demon_freqs <= 15
    ax.plot(demon_freqs[demon_mask], demon_spectrum[demon_mask],
            color=STYLE['accent'], linewidth=1, alpha=0.8)
    ax.fill_between(demon_freqs[demon_mask], 0, demon_spectrum[demon_mask],
                     color=STYLE['accent'], alpha=0.15)

    br = demon_result['blade_rate_hz']
    sr = demon_result['shaft_rate_hz']
    n_blades = demon_result.get('est_n_blades', 0)
    rpm = demon_result.get('est_rpm', 0)
    harmonics_found = demon_result.get('harmonics_found', 0)

    spec_max = float(np.max(demon_spectrum[demon_mask])) if np.any(demon_mask) else 1
    label_offset = spec_max * 0.08

    # Blade rate
    br_idx = np.argmin(np.abs(demon_freqs - br))
    br_amp = float(demon_spectrum[br_idx])
    ax.plot(demon_freqs[br_idx], br_amp, 'v',
            color=STYLE['highlight'], markersize=14, zorder=5)
    ax.text(demon_freqs[br_idx], br_amp + label_offset,
            f'BR\n{br:.1f} Hz', ha='center', va='bottom',
            fontsize=10, color=STYLE['highlight'], fontweight='bold')

    # Harmonics
    for mult, label in [(2, '2×BR'), (3, '3×BR'), (4, '4×BR')]:
        h_hz = br * mult
        if h_hz <= 15:
            h_idx = np.argmin(np.abs(demon_freqs - h_hz))
            h_amp = float(demon_spectrum[h_idx])
            ax.plot(demon_freqs[h_idx], h_amp, 'v',
                    color=STYLE['green'], markersize=10, zorder=4)
            ax.text(demon_freqs[h_idx], h_amp + label_offset * 0.5,
                    f'{label}\n{h_hz:.1f}', ha='center', va='bottom',
                    fontsize=8, color=STYLE['green'])

    # Shaft rate
    if sr > 0 and sr < br:
        sr_idx = np.argmin(np.abs(demon_freqs - sr))
        sr_amp = float(demon_spectrum[sr_idx])
        ax.plot(demon_freqs[sr_idx], sr_amp, 's',
                color=STYLE['red'], markersize=10, zorder=4)
        ax.text(demon_freqs[sr_idx], sr_amp + label_offset,
                f'SR\n{sr:.2f} Hz', ha='center', va='bottom',
                fontsize=9, color=STYLE['red'], fontweight='bold')

    ax.set_xlabel('Modulation Frequency (Hz)', color=STYLE['text'], fontsize=11)
    ax.set_ylabel('DEMON Amplitude', color=STYLE['text'], fontsize=11)
    ax.set_xlim(0, 15)

    # Info box
    rms_ratio = ev.get('rms_ratio', 0)
    assessment = ev.get('assessment', '')
    info_parts = [f'BR={br:.1f} Hz']
    if sr > 0:
        info_parts.append(f'SR={sr:.2f} Hz')
    if n_blades > 0:
        info_parts.append(f'{n_blades}-blade')
    if rpm > 0:
        info_parts.append(f'{rpm:.0f} RPM')
    info_parts.append(f'{harmonics_found} harmonics')
    info_text = '  |  '.join(info_parts)

    ax.text(0.98, 0.92, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color=STYLE['text'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=STYLE['box_fill'],
                      edgecolor=STYLE['box_edge'], alpha=0.9))

    ax.set_title(
        f'DEMON — Event {ev["start"].strftime("%d %b %H:%M")}–{ev["end"].strftime("%H:%M")} '
        f'({rms_ratio:.0f}× bkg) — {assessment}',
        color=STYLE['text'], fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    fname = f'demon_{ev["start"].strftime("%Y%m%d_%H%M")}_{rms_ratio:.0f}x.png'
    path = os.path.join(output_dir, fname)
    _save_fig(fig, path, dpi=150)
    return path


def _plot_single_event(ev, all_files, output_dir, rms_stats, all_windows, audio_data=None, sr=None):
    """Generate a single 3-panel event spectrogram."""
    import soundfile as sf

    pad = timedelta(minutes=EVENT_PADDING_MIN)
    t_start = ev['start'] - pad
    t_end = ev['end'] + pad

    if audio_data is not None and sr is not None:
        session_start = ev.get('session_start_ts', ev['start'] - timedelta(seconds=ev.get('time_offset_sec', 0)))
        
        trim_start = int((t_start - session_start).total_seconds() * sr)
        trim_end = int((t_end - session_start).total_seconds() * sr)
        trim_start = max(0, trim_start)
        trim_end = min(len(audio_data), trim_end)
        segment = audio_data[trim_start:trim_end]
        combined_sr = sr
    else:
        # Find overlapping WAV files
        overlapping = []
        for fi in all_files:
            if fi['start_ts'] is None:
                continue
            file_end_approx = fi['start_ts'] + timedelta(hours=1)
            if fi['start_ts'] <= t_end and file_end_approx >= t_start:
                overlapping.append(fi)

        if not overlapping:
            return None

        overlapping = sorted(overlapping, key=lambda f: f['start_ts'])

        # Load and concatenate audio
        audio_segments = []
        combined_start = overlapping[0]['start_ts']
        combined_sr = None

        for fi in overlapping:
            try:
                audio, sr_file = sf.read(fi['path'])
                if audio.ndim > 1:
                    audio = audio[:, 0]
                audio_segments.append((fi['start_ts'], audio, sr_file))
                combined_sr = sr_file
            except Exception:
                continue

        if not audio_segments or combined_sr is None:
            return None

        # Combine into single array
        total_seconds = (t_end - combined_start).total_seconds()
        total_samples = int(total_seconds * combined_sr)
        combined = np.zeros(total_samples)

        for ts, audio, sr_file in audio_segments:
            offset = int((ts - combined_start).total_seconds() * sr_file)
            end = min(offset + len(audio), len(combined))
            actual_len = end - max(offset, 0)
            if actual_len > 0:
                start_idx = max(0, -offset)
                combined[max(offset, 0):end] = audio[start_idx:start_idx + actual_len]

        # Trim to event window
        trim_start = int((t_start - combined_start).total_seconds() * combined_sr)
        trim_end = int((t_end - combined_start).total_seconds() * combined_sr)
        trim_start = max(0, trim_start)
        trim_end = min(len(combined), trim_end)
        segment = combined[trim_start:trim_end]

    if len(segment) < combined_sr * 10:
        return None

    # Compute STFT
    freqs, stft_times, Sxx = stage2_lofar.compute_stft(segment, combined_sr)
    spec_db, _ = stage2_lofar.normalise_spectrogram(freqs, stft_times, Sxx)

    # Detect persistent tonals
    persistent = stage2_lofar.detect_persistent_tonals(spec_db, freqs, stft_times)

    # Time axis as datetime
    time_dates = [t_start + timedelta(seconds=float(t)) for t in stft_times]

    # --- Build figure (3 panels: spectrogram, RMS, tonals) ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), facecolor=STYLE['bg'],
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    # Consistent time formatting for all panels
    tz_ist = datetime.timezone(timedelta(hours=5, minutes=30))
    time_fmt = DateFormatter('%H:%M', tz=tz_ist)

    # Panel 1: LOFAR Spectrogram
    ax = axes[0]
    _style_axes(ax)
    ax.grid(False)
    freq_mask = freqs <= FREQ_MAX_HZ
    im = ax.pcolormesh(
        time_dates, freqs[freq_mask], spec_db[freq_mask, :],
        cmap=STYLE['cmap'], vmin=-3, vmax=20, shading='auto'
    )
    for tonal in persistent[:15]:
        t_s = t_start + timedelta(seconds=tonal['start_sec'])
        t_e = t_start + timedelta(seconds=tonal['end_sec'])
        ax.plot([t_s, t_e], [tonal['freq_hz'], tonal['freq_hz']],
                color=STYLE['green'], linewidth=1.5, alpha=0.7)

    # Colorbar as inset inside the spectrogram panel (top-right corner)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width='1.5%', height='60%', loc='upper right',
                     borderpad=1.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('dB', color=STYLE['text'], fontsize=8)
    cbar.ax.tick_params(colors=STYLE['text'], labelsize=7)
    cbar.ax.yaxis.set_label_position('left')
    ax.set_ylabel('Frequency (Hz)', color=STYLE['text'], fontsize=10)
    ax.set_ylim(0, FREQ_MAX_HZ)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(time_fmt)
    # Event boundary markers
    for boundary in [ev['start'], ev['end']]:
        ax.axvline(boundary, color=STYLE['green'], linestyle='--', linewidth=1.5, alpha=0.8)

    rms_ratio = ev.get('rms_ratio', 0)
    assessment = ev.get('assessment', '')
    ax.set_title(
        f'Event {ev["start"].strftime("%d %b %H:%M")}–{ev["end"].strftime("%H:%M")} '
        f'({rms_ratio:.0f}× bkg) — {assessment}',
        color=STYLE['text'], fontsize=13, fontweight='bold', pad=12
    )

    # Panel 2: RMS Energy Profile
    ax = axes[1]
    _style_axes(ax)
    if all_windows:
        event_windows = [w for w in all_windows
                         if t_start <= w['timestamp'] <= t_end]
        if event_windows:
            w_times = [w['timestamp'] for w in event_windows]
            w_rms = [w['rms_energy'] for w in event_windows]
            ax.bar(w_times, w_rms, width=timedelta(seconds=12),
                   color=STYLE['accent'], alpha=0.7)
            ax.axhline(rms_stats['rms_threshold'], color=STYLE['red'],
                       linestyle='--', linewidth=1, alpha=0.7,
                       label=f'{RMS_SIGMA_THRESHOLD}σ threshold')
            for boundary in [ev['start'], ev['end']]:
                ax.axvline(boundary, color=STYLE['green'], linestyle='--',
                           linewidth=1.5, alpha=0.8)
            ax.legend(fontsize=8, facecolor=STYLE['bg'],
                      edgecolor=STYLE['box_edge'], labelcolor=STYLE['text'])
    ax.set_ylabel('RMS Energy', color=STYLE['text'], fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(time_fmt)

    # Panel 3: Tonal Lines
    ax = axes[2]
    _style_axes(ax)
    for tonal in persistent[:20]:
        t_s = t_start + timedelta(seconds=tonal['start_sec'])
        t_e = t_start + timedelta(seconds=tonal['end_sec'])
        color = STYLE['highlight'] if tonal['mean_db'] > 12 else STYLE['accent']
        ax.plot([t_s, t_e], [tonal['freq_hz'], tonal['freq_hz']],
                color=color, linewidth=2, alpha=0.8)
        ax.text(t_e, tonal['freq_hz'], f' {tonal["freq_hz"]:.0f} Hz',
                fontsize=7, color=color, va='center')
    for boundary in [ev['start'], ev['end']]:
        ax.axvline(boundary, color=STYLE['green'], linestyle='--',
                   linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Tonal Freq (Hz)', color=STYLE['text'], fontsize=10)
    ax.set_ylim(0, FREQ_MAX_HZ)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(time_fmt)
    ax.set_xlabel('Time (IST)', color=STYLE['text'], fontsize=10)

    plt.tight_layout()
    fname = f'event_{ev["start"].strftime("%Y%m%d_%H%M")}_{rms_ratio:.0f}x.png'
    path = os.path.join(output_dir, fname)
    _save_fig(fig, path, dpi=150)
    return path


# =====================================================================
# 3. 48-HOUR TIMELINE (4 panels)
# =====================================================================

def save_timeline(all_windows, rms_stats, output_dir):
    """Generate 4-panel 48-hour timeline.

    Panels: RMS energy (log), vessel score, tonal count, strongest tonal frequency.
    """
    timestamps = [w['timestamp'] for w in all_windows]
    rms_vals = [w['rms_energy'] for w in all_windows]
    vessel_scores = [w.get('vessel_score', 0) for w in all_windows]
    tonal_counts = [w.get('n_tonals', 0) for w in all_windows]
    strongest_freqs = [w.get('strongest_tonal_hz', 0) for w in all_windows]

    thresh = rms_stats['rms_threshold']

    fig, axes = plt.subplots(4, 1, figsize=(22, 14), sharex=True, facecolor=STYLE['bg'])

    # Date range for title
    date_start = timestamps[0].strftime('%d %b')
    date_end = timestamps[-1].strftime('%d %b %Y')

    # Panel 1: RMS Energy (log)
    ax = axes[0]
    _style_axes(ax)
    colors = [STYLE['red'] if r > thresh else '#2c3e50' for r in rms_vals]
    ax.scatter(timestamps, rms_vals, c=colors, s=3, alpha=0.5)
    ax.axhline(thresh, color=STYLE['accent'], linestyle='--', linewidth=1,
               label=f'{RMS_SIGMA_THRESHOLD}σ threshold ({thresh:.5f})')
    ax.set_ylabel('RMS Energy', color=STYLE['text'], fontsize=11)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=9, facecolor=STYLE['bg'],
              edgecolor=STYLE['box_edge'], labelcolor=STYLE['text'])
    ax.set_title(
        f'Oravont Hydrophone Analysis — {date_start} – {date_end}',
        color=STYLE['text'], fontsize=14, fontweight='bold', pad=12
    )

    # Panel 2: Vessel Score
    ax = axes[1]
    _style_axes(ax)
    ax.scatter(timestamps, vessel_scores, c=STYLE['accent'], s=3, alpha=0.4)
    ax.set_ylabel('Vessel Score', color=STYLE['text'], fontsize=11)
    ax.set_yscale('symlog', linthresh=1)

    # Panel 3: Tonal Count
    ax = axes[2]
    _style_axes(ax)
    ax.scatter(timestamps, tonal_counts, c=STYLE['green'], s=3, alpha=0.4)
    ax.set_ylabel('Tonal Count', color=STYLE['text'], fontsize=11)

    # Panel 4: Strongest Tonal Frequency
    ax = axes[3]
    _style_axes(ax)
    # Only plot non-zero
    nz_mask = [f > 0 for f in strongest_freqs]
    nz_times = [t for t, m in zip(timestamps, nz_mask) if m]
    nz_freqs = [f for f, m in zip(strongest_freqs, nz_mask) if m]
    ax.scatter(nz_times, nz_freqs, c=STYLE['highlight'], s=3, alpha=0.4)
    ax.set_ylabel('Strongest Tonal (Hz)', color=STYLE['text'], fontsize=11)
    ax.set_ylim(0, FREQ_MAX_HZ)
    tz_ist = datetime.timezone(timedelta(hours=5, minutes=30))
    ax.xaxis.set_major_formatter(DateFormatter('%d %b\n%H:%M', tz=tz_ist))
    ax.set_xlabel('Time (IST)', color=STYLE['text'], fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, 'timeline.png')
    _save_fig(fig, path, dpi=150)
    return path


# =====================================================================
# 4. STATISTICAL DISTRIBUTIONS (4 panels)
# =====================================================================

def save_distributions(all_windows, rms_stats, output_dir):
    """Generate 4-panel statistical distribution plots.

    Panels: RMS histogram, blade rate histogram (elevated-RMS only),
    shaft rate histogram, hourly mean RMS.
    """
    rms_vals = np.array([w['rms_energy'] for w in all_windows])
    thresh = rms_stats['rms_threshold']
    rms_mean = rms_stats['rms_mean']

    # Elevated-RMS windows
    elevated = [w for w in all_windows if w['rms_energy'] > thresh]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=STYLE['bg'])

    # RMS histogram
    ax = axes[0, 0]
    _style_axes(ax)
    ax.hist(rms_vals, bins=100, color=STYLE['accent'], alpha=0.7, edgecolor='none')
    ax.axvline(thresh, color=STYLE['red'], linestyle='--', linewidth=2,
               label=f'{RMS_SIGMA_THRESHOLD}σ = {thresh:.5f}')
    ax.set_xlabel('RMS Energy', color=STYLE['text'])
    ax.set_ylabel('Count', color=STYLE['text'])
    ax.set_title('RMS Energy Distribution', color=STYLE['text'], fontsize=11)
    ax.legend(fontsize=9, facecolor=STYLE['bg'], edgecolor=STYLE['box_edge'],
              labelcolor=STYLE['text'])

    # Blade rate histogram (elevated-RMS only)
    ax = axes[0, 1]
    _style_axes(ax)
    br_vals = [w.get('blade_rate_hz', 0) for w in elevated if w.get('blade_rate_hz', 0) > 0]
    if br_vals:
        ax.hist(br_vals, bins=60, color=STYLE['green'], alpha=0.7, edgecolor='none')
    ax.set_xlabel('Blade Rate (Hz)', color=STYLE['text'])
    ax.set_ylabel('Count', color=STYLE['text'])
    ax.set_title('Blade Rate Distribution (elevated RMS)', color=STYLE['text'], fontsize=11)

    # Shaft rate histogram
    ax = axes[1, 0]
    _style_axes(ax)
    sr_vals = [w.get('shaft_rate_hz', 0) for w in elevated if w.get('shaft_rate_hz', 0) > 0]
    if sr_vals:
        ax.hist(sr_vals, bins=40, color=STYLE['highlight'], alpha=0.7, edgecolor='none')
    ax.set_xlabel('Shaft Rate (Hz)', color=STYLE['text'])
    ax.set_ylabel('Count', color=STYLE['text'])
    ax.set_title('Shaft Rate Distribution (elevated RMS)', color=STYLE['text'], fontsize=11)

    # Hourly mean RMS
    ax = axes[1, 1]
    _style_axes(ax)
    hourly = {}
    for w in all_windows:
        hour_key = w['timestamp'].replace(minute=0, second=0, microsecond=0)
        hourly.setdefault(hour_key, []).append(w['rms_energy'])
    h_times = sorted(hourly.keys())
    h_means = [np.mean(hourly[t]) for t in h_times]
    ax.bar(h_times, h_means, width=timedelta(hours=0.8),
           color=STYLE['accent'], alpha=0.7)
    ax.axhline(rms_mean, color=STYLE['text'], linestyle=':', alpha=0.5,
               label=f'Mean={rms_mean:.5f}')
    ax.axhline(thresh, color=STYLE['red'], linestyle='--', alpha=0.7,
               label=f'{RMS_SIGMA_THRESHOLD}σ={thresh:.5f}')
    tz_ist = datetime.timezone(timedelta(hours=5, minutes=30))
    ax.xaxis.set_major_formatter(DateFormatter('%d\n%H', tz=tz_ist))
    ax.set_xlabel('Time (IST)', color=STYLE['text'])
    ax.set_ylabel('Mean RMS Energy', color=STYLE['text'])
    ax.set_title('Hourly Mean RMS Energy', color=STYLE['text'], fontsize=11)
    ax.legend(fontsize=8, facecolor=STYLE['bg'], edgecolor=STYLE['box_edge'],
              labelcolor=STYLE['text'])

    plt.suptitle('Statistical Overview', color=STYLE['text'],
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'distributions.png')
    _save_fig(fig, path, dpi=150)
    return path


# =====================================================================
# 5. EVENTS DETAIL (top 8)
# =====================================================================

def save_events_detail(all_windows, events, output_dir):
    """Generate detail views for top 8 significant events.

    Each subplot shows RMS energy bars + blade rate markers.
    """
    # Filter significant events
    sig = [e for e in events
           if e.get('dur_min', 0) >= SIGNIFICANT_DURATION_MIN
           or e.get('rms_ratio', 0) >= SIGNIFICANT_RMS_FACTOR]

    if not sig:
        print('  No significant events for detail view')
        return None

    # Top 8 by peak RMS
    top = sorted(sig, key=lambda e: e.get('peak_rms', 0), reverse=True)[:8]
    top = sorted(top, key=lambda e: e['start'])  # chronological

    n = len(top)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), facecolor=STYLE['bg'])
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(nrows * ncols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        _style_axes(ax)

        if idx >= n:
            ax.axis('off')
            continue

        ev = top[idx]
        pad = timedelta(minutes=5)
        t_start = ev['start'] - pad
        t_end = ev['end'] + pad

        # Windows in this event
        window = [w for w in all_windows
                  if t_start <= w['timestamp'] <= t_end]

        if not window:
            ax.axis('off')
            continue

        w_times = [w['timestamp'] for w in window]
        w_rms = [w['rms_energy'] for w in window]
        w_br = [w.get('blade_rate_hz', 0) for w in window]

        ax2 = ax.twinx()
        ax.bar(w_times, w_rms, width=timedelta(seconds=12),
               color=STYLE['accent'], alpha=0.6, label='RMS')
        ax2.scatter(w_times, w_br, c=STYLE['highlight'], s=15,
                    alpha=0.8, label='Blade Rate', zorder=5)

        ax.set_ylabel('RMS', color=STYLE['accent'], fontsize=9)
        ax2.set_ylabel('BR (Hz)', color=STYLE['highlight'], fontsize=9)
        ax2.tick_params(colors=STYLE['highlight'])

        rms_ratio = ev.get('rms_ratio', 0)
        assessment = ev.get('assessment', '')
        title = (f"{ev['start'].strftime('%d %b %H:%M')}–{ev['end'].strftime('%H:%M')} "
                 f"({rms_ratio:.0f}× bkg) — {assessment}")
        ax.set_title(title, color=STYLE['text'], fontsize=10)
        tz_ist = datetime.timezone(timedelta(hours=5, minutes=30))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=tz_ist))

    plt.suptitle('Major Acoustic Events — Detail View', color=STYLE['text'],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'events_detail.png')
    _save_fig(fig, path, dpi=150)
    return path


# =====================================================================
# 6. SCORE ANALYSIS (DEMON score histogram + decomposition)
# =====================================================================

def save_score_analysis(all_windows, output_dir):
    """Generate DEMON score histogram and stacked decomposition chart."""
    demon_scores = [w.get('demon_score', 0) for w in all_windows]
    # Only consider windows where DEMON ran
    nonzero_scores = [s for s in demon_scores if s > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=STYLE['bg'])

    # Score distribution
    ax = axes[0]
    _style_axes(ax)
    if nonzero_scores:
        # Bin scores
        bins = np.arange(0, max(nonzero_scores) + 1, 0.5)
        ax.hist(nonzero_scores, bins=bins, color=STYLE['accent'], alpha=0.7,
                edgecolor='none', orientation='horizontal')
    n_with_demon = len(nonzero_scores)
    n_total = len(all_windows)
    ax.set_ylabel('DEMON Score', color=STYLE['text'])
    ax.set_xlabel('Number of Windows', color=STYLE['text'])
    ax.set_title(
        f'DEMON Score Distribution ({n_with_demon}/{n_total} windows)',
        color=STYLE['text'], fontsize=11, fontweight='bold'
    )
    ax.axhline(DEMON_SCORE_THRESHOLD, color=STYLE['red'], linestyle=':',
               linewidth=1.5, label=f'Threshold = {DEMON_SCORE_THRESHOLD}')
    ax.legend(fontsize=9, facecolor=STYLE['bg'], edgecolor=STYLE['box_edge'],
              labelcolor=STYLE['text'])

    # Stacked decomposition
    ax = axes[1]
    _style_axes(ax)
    components = ['Prominence\n(variable)', 'Shaft-rate\nbonus',
                  '4×BR\nbonus', '3×BR\nbonus', '2×BR\nbonus']
    values = [0.0, DEMON_SCORE_SHAFT_BONUS, DEMON_SCORE_4X_BONUS,
              DEMON_SCORE_3X_BONUS, DEMON_SCORE_2X_BONUS]
    colors_stack = [STYLE['text'], STYLE['highlight'], STYLE['accent'],
                    STYLE['accent'], STYLE['accent']]

    bottom = 0
    total_bonus = sum(values)
    for comp, val, col in zip(components, values, colors_stack):
        if val > 0:
            ax.bar(0, val, bottom=bottom, color=col, alpha=0.7,
                   width=0.5, edgecolor=STYLE['box_edge'])
            ax.text(0, bottom + val / 2, f'{comp}\n+{val:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
        bottom += val

    ax.axhline(total_bonus, color=STYLE['green'], linestyle='--', linewidth=2,
               label=f'Max bonuses = {total_bonus:.2f}')
    ax.axhline(DEMON_SCORE_THRESHOLD, color=STYLE['red'], linestyle=':',
               linewidth=1.5, label=f'Threshold = {DEMON_SCORE_THRESHOLD}')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_title('Score Decomposition', color=STYLE['text'], fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=9, facecolor=STYLE['bg'], edgecolor=STYLE['box_edge'],
              labelcolor=STYLE['text'])
    ax.grid(True, alpha=0.15, color=STYLE['box_edge'], axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'score_analysis.png')
    _save_fig(fig, path, dpi=180)
    return path


# =====================================================================
# 7. PIPELINE DIAGRAM (static)
# =====================================================================

def save_pipeline_diagram(output_dir):
    """Generate static 3-stage pipeline block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), facecolor=STYLE['bg'])
    ax.set_facecolor(STYLE['bg'])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(8, 6.6,
            'Oravont Hydrophone Analysis Pipeline — 3-Stage Processing',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=STYLE['accent'], fontfamily='monospace')

    # Stage boxes (top row)
    boxes = [
        (0.3, 4.5, 2.2, 1.6, 'STAGE 1\nRMS ENERGY',
         '30s windows\n15s overlap\nBroadband power', STYLE['accent']),
        (3.2, 4.5, 2.4, 1.6, 'STAGE 2\nLOFAR/STFT',
         '4s STFT, 75% overlap\n0.125 Hz bins\nTonal detection', STYLE['green']),
        (6.3, 4.5, 2.4, 1.6, 'STAGE 3\nDEMON',
         'Hilbert envelope\nBlade-rate FFT\nConditional on RMS', STYLE['highlight']),
        (9.4, 4.5, 2.4, 1.6, 'EVENT\nDETECTOR',
         'σ-threshold merge\nClassification\nRemarks builder', STYLE['accent']),
        (12.5, 4.5, 2.8, 1.6, 'OUTPUTS',
         'HAVS XLSX\nTransit CSV\nSpectrograms\nTimelines', STYLE['green']),
    ]

    for x, y, w, h, title, detail, color in boxes:
        rect = FancyBboxPatch((x, y - h / 2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=STYLE['box_fill'],
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.25, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color, fontfamily='monospace')
        ax.text(x + w / 2, y - 0.35, detail, ha='center', va='center',
                fontsize=7, color=STYLE['text'], fontfamily='monospace', alpha=0.8)

    # Arrows between stages
    arrow_pairs = [(2.5, 3.2), (5.6, 6.3), (8.7, 9.4), (11.8, 12.5)]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2, 4.5), xytext=(x1, 4.5),
                    arrowprops=dict(arrowstyle='->', color=STYLE['accent'], lw=2))

    # Conditional gate annotation (Stage 3)
    ax.annotate('', xy=(6.3, 3.5), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=STYLE['red'],
                                lw=1.5, linestyle='dashed'))
    ax.text(4.4, 3.1,
            'Stage 3 fires only on elevated-RMS windows',
            ha='center', fontsize=8, color=STYLE['red'],
            fontfamily='monospace', style='italic')

    # Input box
    input_rect = FancyBboxPatch((0.3, 0.3), 3.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=STYLE['box_fill'],
                                edgecolor=STYLE['accent'],
                                linewidth=1.5, linestyle='dashed')
    ax.add_patch(input_rect)
    ax.text(2.05, 1.3, 'WAV Input', ha='center',
            fontsize=9, fontweight='bold', color=STYLE['accent'],
            fontfamily='monospace')
    ax.text(2.05, 0.7,
            '512 Hz (or auto-detect)  |  1-hour sessions\n'
            'Corrupt file handling  |  Checkpoint/resume',
            ha='center', fontsize=7, color=STYLE['text'],
            fontfamily='monospace', alpha=0.8)

    # Scoring box
    score_rect = FancyBboxPatch((5.0, 0.3), 5.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=STYLE['box_fill'],
                                edgecolor=STYLE['highlight'],
                                linewidth=1.5, linestyle='dashed')
    ax.add_patch(score_rect)
    ax.text(7.75, 1.3, 'Vessel Scoring (Soft-Gated)', ha='center',
            fontsize=9, fontweight='bold', color=STYLE['highlight'],
            fontfamily='monospace')
    ax.text(7.75, 0.7,
            'RMS gate ramps 0→1 as energy rises  |  log-scaled  |  No hard caps\n'
            'Full dynamic range: 0.3 (quiet night) → 700+ (tanker CPA)',
            ha='center', fontsize=7, color=STYLE['text'],
            fontfamily='monospace', alpha=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pipeline_diagram.png')
    _save_fig(fig, path, dpi=180)
    return path


# =====================================================================
# 8. DEMON ILLUSTRATION (static, 4-panel)
# =====================================================================

def save_demon_illustration(output_dir):
    """Generate static 4-panel DEMON signal processing illustration.

    Uses synthetic data to show:
    (a) Bandpass-filtered signal with amplitude modulation
    (b) Hilbert envelope extraction
    (c) DEMON modulation spectrum with labelled peaks
    (d) Harmonic scoring decomposition
    """
    from scipy.signal import hilbert as sp_hilbert

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=STYLE['bg'])
    for ax in axes.flat:
        _style_axes(ax)

    # Synthetic signal: carrier at 80 Hz, modulated by blade rate at 8 Hz
    np.random.seed(42)
    t = np.linspace(0, 1, 4000)
    carrier = np.sin(2 * np.pi * 80 * t)
    blade_mod = 1 + 0.3 * np.sin(2 * np.pi * 8 * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = blade_mod * carrier + noise

    # Panel (a): Bandpass-filtered signal
    ax = axes[0, 0]
    ax.plot(t[:2000], signal[:2000], color=STYLE['accent'], alpha=0.6, linewidth=0.5)
    ax.set_title('(a) Bandpass-Filtered Signal',
                 color=STYLE['text'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)', color=STYLE['text'], fontsize=9)
    ax.set_ylabel('Amplitude', color=STYLE['text'], fontsize=9)

    # Panel (b): Hilbert envelope
    analytic = sp_hilbert(signal)
    envelope = np.abs(analytic)

    ax = axes[0, 1]
    ax.plot(t[:2000], signal[:2000], color=STYLE['accent'], alpha=0.3,
            linewidth=0.5, label='Signal')
    ax.plot(t[:2000], envelope[:2000], color=STYLE['highlight'],
            linewidth=1.5, label='Hilbert Envelope')
    ax.set_title('(b) Hilbert Envelope Extraction',
                 color=STYLE['text'], fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor=STYLE['bg'],
              edgecolor=STYLE['box_edge'], labelcolor=STYLE['text'])

    # Panel (c): DEMON modulation spectrum
    envelope_ac = envelope - np.mean(envelope)
    n_fft = len(envelope_ac) * 4
    fft_result = np.fft.rfft(envelope_ac, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1 / 4000)
    demon_spectrum = np.abs(fft_result) / len(envelope_ac)
    mask = freqs <= 50

    ax = axes[1, 0]
    ax.plot(freqs[mask], demon_spectrum[mask], color=STYLE['green'], linewidth=1)
    # Label peaks
    for h, label in [(8, 'BR\n8 Hz'), (16, '2×BR'), (24, '3×BR'),
                     (32, '4×BR'), (4, 'SR\n4 Hz')]:
        idx = np.argmin(np.abs(freqs - h))
        color = STYLE['highlight'] if h == 8 else (
            STYLE['accent'] if h == 4 else STYLE['text'])
        ax.plot(freqs[idx], demon_spectrum[idx], 'v', color=color, markersize=8)
        ax.text(freqs[idx], demon_spectrum[idx] + 0.0005, label,
                ha='center', va='bottom', fontsize=7, color=color,
                fontweight='bold')
    ax.set_title('(c) DEMON Modulation Spectrum',
                 color=STYLE['text'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Modulation Frequency (Hz)', color=STYLE['text'], fontsize=9)
    ax.set_ylabel('Amplitude', color=STYLE['text'], fontsize=9)

    # Panel (d): Harmonic scoring
    ax = axes[1, 1]
    _style_axes(ax)
    components = ['2×BR\n+2.50', '3×BR\n+1.67', '4×BR\n+1.25', 'Shaft\n+3.00']
    values = [DEMON_SCORE_2X_BONUS, DEMON_SCORE_3X_BONUS,
              DEMON_SCORE_4X_BONUS, DEMON_SCORE_SHAFT_BONUS]
    colors_bar = [STYLE['accent'], STYLE['accent'], STYLE['accent'], STYLE['highlight']]

    bars = ax.bar(components, values, color=colors_bar, alpha=0.7,
                  edgecolor=STYLE['box_edge'], width=0.5)
    ax.axhline(DEMON_SCORE_THRESHOLD, color=STYLE['red'], linestyle=':',
               linewidth=1.5, label=f'Detection threshold = {DEMON_SCORE_THRESHOLD}')
    total = sum(values)
    ax.axhline(total, color=STYLE['green'], linestyle='--', linewidth=1.5,
               label=f'Max bonus total = {total:.2f}')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', color=STYLE['text'],
                fontsize=9, fontweight='bold')

    ax.set_title('(d) Harmonic Scoring Components',
                 color=STYLE['text'], fontsize=10, fontweight='bold')
    ax.set_ylabel('Score Contribution', color=STYLE['text'], fontsize=9)
    ax.legend(fontsize=8, facecolor=STYLE['bg'], edgecolor=STYLE['box_edge'],
              labelcolor=STYLE['text'])
    ax.grid(True, alpha=0.15, color=STYLE['box_edge'], axis='y')

    plt.suptitle('DEMON Signal Processing — Illustrative Example',
                 color=STYLE['text'], fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'demon_illustration.png')
    _save_fig(fig, path, dpi=180)
    return path
