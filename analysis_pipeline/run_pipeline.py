#!/usr/bin/env python3
"""
Oravont Hydrophone Analysis Pipeline
======================================
Three-stage passive acoustic vessel detection and classification.

  Stage 1: RMS energy detection (broadband)
  Stage 2: LOFAR spectrogram / tonal analysis (narrowband)
  Stage 3: DEMON blade-rate extraction (conditional, on elevated-RMS windows)

Usage:
    python run_pipeline.py /path/to/wav/files/ --output ./results/
    python run_pipeline.py /path/to/wav/files/ --output ./results/ --no-spectrograms
    python run_pipeline.py /path/to/wav/files/ --output ./results/ --sessions 113-160

Oravont Systems LLP — Confidential
"""

import argparse
import os
import sys
import re
import json
import time as _time
import gc
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf

warnings.filterwarnings('ignore')

# Allow running as module or standalone
try:
    from .config import *
    from . import stage1_rms, stage2_lofar, stage3_demon, event_detector, havs_writer
except ImportError:
    from config import *
    import stage1_rms, stage2_lofar, stage3_demon, event_detector, havs_writer


def parse_filename(fname):
    """Extract session number and start timestamp from filename."""
    m = re.match(r'session_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_to_', fname)
    if not m:
        return None, None
    session = int(m.group(1))
    ts = datetime.strptime(f'{m.group(2)}_{m.group(3)}', '%Y-%m-%d_%H-%M-%S')
    return session, ts


def discover_wav_files(input_dirs, keep_sessions=None):
    """Find all WAV files across input directories."""
    all_files = []
    for d in input_dirs:
        if not os.path.exists(d):
            print(f'  WARNING: {d} not found, skipping')
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith('.wav'):
                continue
            session, ts = parse_filename(f)
            if session is None:
                session = len(all_files) + 1
                ts = None
            if keep_sessions and session not in keep_sessions:
                continue
            all_files.append({
                'filename': f,
                'path': os.path.join(d, f),
                'session': session,
                'start_ts': ts,
            })

    if all([f['start_ts'] for f in all_files]):
        all_files = sorted(all_files, key=lambda x: x['start_ts'])
    else:
        all_files = sorted(all_files, key=lambda x: x['filename'])

    return all_files


def process_one_file(file_info, file_idx, total_files, rms_mean_estimate=None):
    """Process a single WAV file through all three stages."""
    fname = file_info['filename']
    fpath = file_info['path']
    file_start = file_info['start_ts']
    session = file_info['session']

    print(f'  [{file_idx+1}/{total_files}] session_{session} — '
          f'{file_start.strftime("%d %b %H:%M") if file_start else "unknown"}')

    try:
        audio, sr = sf.read(fpath)
    except Exception as e:
        print(f'    ERROR reading file: {e}')
        return [], None

    if audio.ndim > 1:
        audio = audio[:, 0]

    duration_sec = len(audio) / sr
    print(f'    {sr} Hz, {duration_sec:.0f}s, {len(audio)/1e6:.1f}M samples')

    if file_start is None:
        file_start = datetime(2026, 1, 1) + timedelta(seconds=file_idx * 3600)

    # Stage 1
    windows = stage1_rms.compute_rms_windows(audio, sr, file_start)

    # Stage 2
    freqs, stft_times, Sxx = stage2_lofar.compute_stft(audio, sr)
    spec_db, background = stage2_lofar.normalise_spectrogram(freqs, stft_times, Sxx)
    persistent = stage2_lofar.detect_persistent_tonals(spec_db, freqs, stft_times)

    for w in windows:
        chunk_start_sec = w['time_offset_sec']
        chunk_end_sec = chunk_start_sec + ANALYSIS_WINDOW_SEC
        stft_mask = (stft_times >= chunk_start_sec) & (stft_times < chunk_end_sec)

        if not stft_mask.any():
            w.update({'n_tonals': 0, 'tonal_energy': 0, 'strongest_tonal_hz': 0,
                      'strongest_tonal_db': 0, 'tonal_freqs': [],
                      'band_low_5_20_db': 0, 'band_mid_20_80_db': 0, 'band_high_80_256_db': 0,
                      'vessel_score': 0})
            continue

        spec_chunk = spec_db[:, stft_mask]
        lofar = stage2_lofar.process_lofar_window(w['audio_chunk'], sr, spec_chunk, freqs)
        w.update(lofar)

        rms_mean = rms_mean_estimate if rms_mean_estimate else 0.001
        w['vessel_score'] = stage2_lofar.compute_vessel_score(lofar, w['rms_energy'], rms_mean)

    # Stage 3 (conditional)
    rms_values = [w['rms_energy'] for w in windows]
    local_mean = np.mean(rms_values)
    local_std = np.std(rms_values)
    local_thresh = local_mean + RMS_SIGMA_THRESHOLD * local_std

    for w in windows:
        if w['rms_energy'] > local_thresh:
            demon_freqs, demon_spec = stage3_demon.compute_demon_spectrum(w['audio_chunk'], sr)
            demon_result = stage3_demon.detect_blade_rate(demon_freqs, demon_spec)
            rms_ratio = w['rms_energy'] / local_mean if local_mean > 0 else 0
            demon_result['demon_confidence'] = stage3_demon.classify_demon_confidence(demon_result, rms_ratio)
            demon_result['vessel_class'] = stage3_demon.classify_vessel(demon_result['shaft_rate_hz'])
            w.update({
                'blade_rate_hz': demon_result['blade_rate_hz'],
                'shaft_rate_hz': demon_result['shaft_rate_hz'],
                'est_n_blades': demon_result['est_n_blades'],
                'est_rpm': demon_result['est_rpm'],
                'demon_score': demon_result['demon_score'],
                'demon_confidence': demon_result['demon_confidence'],
                'vessel_class': demon_result['vessel_class'],
            })
        else:
            w.update({
                'blade_rate_hz': 0, 'shaft_rate_hz': 0, 'est_n_blades': 0,
                'est_rpm': 0, 'demon_score': 0, 'demon_confidence': '', 'vessel_class': '',
            })

    for w in windows:
        w['file'] = fname
        w['session'] = session
        w['sample_rate'] = sr

    for w in windows:
        w.pop('audio_chunk', None)

    n_persistent = len(persistent)
    if persistent:
        top_freqs = ', '.join(f'{t["freq_hz"]:.1f}Hz ({t["duration_sec"]:.0f}s)' for t in persistent[:5])
    else:
        top_freqs = 'none'
    print(f'    {len(windows)} windows, {n_persistent} persistent tonals: {top_freqs}')

    stft_data = {
        'freqs': freqs, 'times': stft_times, 'spec_db': spec_db,
        'file_start': file_start, 'session': session, 'sr': sr,
        'persistent': persistent,
    }

    del audio, Sxx
    gc.collect()

    return windows, stft_data


def main():
    parser = argparse.ArgumentParser(
        description='Oravont Hydrophone Analysis Pipeline — 3-stage vessel detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', nargs='+',
                        help='Input directory(ies) containing WAV files')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--sessions', '-s', default=None,
                        help='Session range to process, e.g. "113-160" or "all"')
    parser.add_argument('--no-spectrograms', action='store_true',
                        help='Skip spectrogram image generation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    keep_sessions = None
    if args.sessions and args.sessions != 'all':
        m = re.match(r'(\d+)-(\d+)', args.sessions)
        if m:
            keep_sessions = set(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            keep_sessions = set(int(s) for s in args.sessions.split(','))

    print('=' * 60)
    print('ORAVONT HYDROPHONE ANALYSIS PIPELINE')
    print('=' * 60)
    print(f'Input:    {", ".join(args.input)}')
    print(f'Output:   {args.output}')
    print(f'Sessions: {f"{min(keep_sessions)}-{max(keep_sessions)}" if keep_sessions else "ALL"}')
    print()

    all_files = discover_wav_files(args.input, keep_sessions)
    if not all_files:
        print('ERROR: No WAV files found')
        sys.exit(1)

    print(f'Found {len(all_files)} WAV files')
    if all_files[0]['start_ts']:
        print(f'  First: session {all_files[0]["session"]} — {all_files[0]["start_ts"]}')
        print(f'  Last:  session {all_files[-1]["session"]} — {all_files[-1]["start_ts"]}')
    print()

    checkpoint_file = os.path.join(args.output, '_checkpoint.json')
    processed_sessions = set()
    all_windows = []

    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        processed_sessions = set(ckpt.get('processed_sessions', []))
        print(f'Resuming: {len(processed_sessions)} files already done')

    start_time = _time.time()
    files_to_process = [f for f in all_files if f['session'] not in processed_sessions]
    print(f'Files to process: {len(files_to_process)}')
    print()

    stft_data_list = []

    for fi, file_info in enumerate(files_to_process):
        windows, stft_data = process_one_file(file_info, fi, len(files_to_process))
        all_windows.extend(windows)
        if stft_data:
            stft_data_list.append(stft_data)
        processed_sessions.add(file_info['session'])

        if (fi + 1) % CHECKPOINT_INTERVAL == 0 or fi == len(files_to_process) - 1:
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed_sessions': sorted(processed_sessions)}, f)
            elapsed = _time.time() - start_time
            rate = (fi + 1) / (elapsed / 60) if elapsed > 0 else 0
            remaining = (len(files_to_process) - fi - 1) / rate if rate > 0 else 0
            print(f'    Checkpoint: {len(processed_sessions)} files, '
                  f'{len(all_windows)} windows, '
                  f'{elapsed/60:.1f}min elapsed, ~{remaining:.0f}min remaining')

    total_time = _time.time() - start_time
    print(f'\n{"="*60}')
    print(f'PROCESSING COMPLETE')
    print(f'  Files: {len(processed_sessions)}')
    print(f'  Windows: {len(all_windows)}')
    print(f'  Time: {total_time/60:.1f} minutes')
    print()

    if not all_windows:
        print('No windows produced — check input files')
        sys.exit(1)

    print('Computing global statistics...')
    rms_stats = stage1_rms.compute_rms_statistics(all_windows)
    print(f'  RMS dynamic range: {rms_stats["dynamic_range"]:.0f}×')
    print(f'  Day/night ratio: {rms_stats["day_night_ratio"]:.2f}×')
    print(f'  Elevated windows: {rms_stats["n_elevated"]} ({rms_stats["pct_elevated"]:.1f}%)')

    print('Recomputing vessel scores with global baseline...')
    for w in all_windows:
        lofar = {k: w.get(k, 0) for k in ['n_tonals', 'tonal_energy', 'strongest_tonal_hz',
                                             'strongest_tonal_db', 'tonal_freqs']}
        w['vessel_score'] = stage2_lofar.compute_vessel_score(lofar, w['rms_energy'], rms_stats['rms_mean'])

    print('Detecting events...')
    events = stage1_rms.detect_events(all_windows, rms_stats)
    print(f'  Raw events: {len(events)}')

    events = event_detector.classify_events(events, all_windows)
    sig = [e for e in events if e['assessment'] not in ('brief transient',)]
    print(f'  Significant events: {len(sig)}')

    confirmed = [e for e in sig if 'CONFIRMED' in e['assessment']]
    probable = [e for e in sig if 'PROBABLE' in e['assessment']]
    print(f'  Confirmed vessels: {len(confirmed)}')
    print(f'  Probable vessels: {len(probable)}')

    print('\nWriting HAVS output...')
    session_range = (f'Sessions {min(processed_sessions)}-{max(processed_sessions)}'
                     if processed_sessions else '')
    notes = [session_range] if session_range else []
    havs_path = os.path.join(args.output, 'havs_results.xlsx')
    havs_writer.write_havs_xlsx(events, havs_path, notes=notes)
    print(f'  Saved: {havs_path}')

    print('Writing events summary...')
    import csv
    events_path = os.path.join(args.output, 'transit_events.csv')
    with open(events_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'event_id', 'start', 'end', 'dur_min', 'n_windows',
            'peak_rms', 'rms_ratio', 'blade_rate', 'shaft_rate',
            'est_n_blades', 'est_rpm', 'mean_tonals', 'assessment', 'remarks',
        ])
        writer.writeheader()
        for i, ev in enumerate(events):
            writer.writerow({
                'event_id': i + 1,
                'start': ev['start'].isoformat(),
                'end': ev['end'].isoformat(),
                'dur_min': round(ev['dur_min'], 1),
                'n_windows': ev['n_windows'],
                'peak_rms': round(ev['peak_rms'], 6),
                'rms_ratio': round(ev['rms_ratio'], 1),
                'blade_rate': round(ev.get('blade_rate', 0), 1),
                'shaft_rate': round(ev.get('shaft_rate', 0), 2),
                'est_n_blades': ev.get('est_n_blades', 0),
                'est_rpm': int(ev.get('est_rpm', 0)),
                'mean_tonals': round(ev.get('mean_tonals', 0), 1),
                'assessment': ev['assessment'],
                'remarks': ev.get('remarks', ''),
            })
    print(f'  Saved: {events_path}')

    # Generate spectrograms and figures
    if not args.no_spectrograms:
        print('\nGenerating figures...')
        try:
            from . import spectrogram_writer
        except ImportError:
            import spectrogram_writer

        # 1. Per-file LOFAR spectrograms
        if GENERATE_FILE_SPECTROGRAMS and stft_data_list:
            print('  Per-file spectrograms...')
            for stft_data in stft_data_list:
                spectrogram_writer.save_file_spectrogram(stft_data, args.output)
            print(f'  Saved {len(stft_data_list)} file spectrograms')

        # 2. Event spectrograms (3-panel)
        if GENERATE_EVENT_SPECTROGRAMS and events:
            print('  Event spectrograms...')
            event_paths = spectrogram_writer.save_event_spectrograms(
                events, all_files, args.output, rms_stats, all_windows)
            print(f'  Saved {len(event_paths)} event spectrograms')

        # 3. DEMON plots (separate, per event)
        if GENERATE_EVENT_SPECTROGRAMS and events:
            demon_paths = spectrogram_writer.save_demon_plots(
                events, all_files, args.output, rms_stats)
            if demon_paths:
                print(f'  Saved {len(demon_paths)} DEMON plots')

    # Summary
    print(f'\n{"="*60}')
    print(f'PIPELINE COMPLETE')
    print(f'{"="*60}')
    print(f'  Files processed:     {len(processed_sessions)}')
    print(f'  Analysis windows:    {len(all_windows)}')
    print(f'  Events detected:     {len(events)}')
    print(f'  Confirmed vessels:   {len(confirmed)}')
    print(f'  Probable vessels:    {len(probable)}')
    print(f'  RMS dynamic range:   {rms_stats["dynamic_range"]:.0f}×')
    print(f'  Day/night ratio:     {rms_stats["day_night_ratio"]:.2f}×')
    print(f'  Processing time:     {total_time/60:.1f} min')
    print(f'\n  Output: {args.output}/')
    for f in sorted(os.listdir(args.output)):
        if not f.startswith('.') and not f.startswith('_'):
            fpath = os.path.join(args.output, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f'    {f:45s} {size/1024:>8.0f} KB')


if __name__ == '__main__':
    main()
