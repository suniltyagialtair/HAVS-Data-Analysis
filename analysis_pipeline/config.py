"""
Oravont Hydrophone Analysis Pipeline — Configuration
=====================================================
All tunable parameters for the three-stage processing pipeline.
Site-specific calibration is done by adjusting these values.
"""

# =====================================================================
# STAGE 1: RMS ENERGY DETECTION
# =====================================================================
ANALYSIS_WINDOW_SEC = 30        # seconds per output row
ANALYSIS_HOP_SEC = 15           # hop → 50% overlap
RMS_SIGMA_THRESHOLD = 1.0       # σ above mean for elevated energy
EVENT_MERGE_RADIUS = 8          # ±N windows for temporal merging (each window = 15s)
EVENT_MIN_WINDOWS = 5           # minimum windows per event
SIGNIFICANT_DURATION_MIN = 5    # minutes for "significant" transit
SIGNIFICANT_RMS_FACTOR = 5.0    # ×background for "significant" transit

# =====================================================================
# STAGE 2: LOFAR SPECTROGRAM / TONAL ANALYSIS
# =====================================================================
STFT_WINDOW_SEC = 4.0           # seconds → 0.25 Hz native resolution
STFT_OVERLAP_FRAC = 0.75        # 75% overlap → ~1 spectrum/second
STFT_NFFT_MULT = 2              # zero-pad factor → 0.125 Hz bins
FREQ_MAX_HZ = 256               # analysis up to Nyquist (for 512 Hz)

# Background normalisation
BACKGROUND_PERCENTILE = 50      # median spectrum as background

# Tonal detection
TONAL_THRESHOLD_DB = 8.0        # dB above normalised background
TONAL_MIN_PERSIST_SEC = 30      # must persist this long
TONAL_FREQ_MIN_HZ = 3.5         # ignore below (swell band)
TONAL_FREQ_MAX_HZ = 256         # upper limit
TONAL_PEAK_PROMINENCE_DB = 3.0  # minimum prominence
TONAL_PEAK_MIN_DISTANCE_HZ = 1.0  # minimum spacing between peaks
TONAL_BAND_WIDTH_HZ = 2.0       # group spectrum into N Hz bands

# Vessel scoring (soft-gated)
VESSEL_SCORE_THRESHOLD = 2.0    # score above this = vessel candidate

# =====================================================================
# STAGE 3: DEMON BLADE-RATE EXTRACTION (conditional)
# =====================================================================
# Only runs on windows where Stage 1 flags elevated RMS
DEMON_BLADE_RATE_MIN_HZ = 2.0   # search range for blade rate
DEMON_BLADE_RATE_MAX_HZ = 15.0
DEMON_HARMONIC_TOL_HZ = 1.0     # ±tolerance for harmonic matching
DEMON_SHAFT_TOL_HZ = 0.5        # ±tolerance for shaft-rate matching
DEMON_MIN_PROMINENCE = 0.5      # minimum peak prominence in DEMON spectrum

# Scoring
DEMON_SCORE_2X_BONUS = 2.50
DEMON_SCORE_3X_BONUS = 1.67
DEMON_SCORE_4X_BONUS = 1.25
DEMON_SCORE_SHAFT_BONUS = 3.00
DEMON_SCORE_NOISE_PENALTY = 0.50
DEMON_SCORE_THRESHOLD = 5.0

# =====================================================================
# EVENT CLASSIFICATION
# =====================================================================
# Vessel class by shaft rate
CLASS_LARGE_VESSEL_SR_MAX = 3.0
CLASS_FISHING_SR_MAX = 10.0
# Above 10 Hz → small_craft

# DEMON confidence
DEMON_HIGH_CONF_MIN_HARMONICS = 2  # need ≥2 harmonics for HIGH
DEMON_HIGH_CONF_MIN_RMS_FACTOR = 5.0

# =====================================================================
# SPECTROGRAM IMAGE GENERATION
# =====================================================================
GENERATE_FILE_SPECTROGRAMS = True
GENERATE_EVENT_SPECTROGRAMS = True
EVENT_PADDING_MIN = 5
EVENT_SPEC_MIN_RMS_FACTOR = 5.0

# =====================================================================
# CHECKPOINT / RESUME
# =====================================================================
CHECKPOINT_INTERVAL = 5  # save checkpoint every N files

# =====================================================================
# VISUAL STYLE (deep-ocean aesthetic)
# =====================================================================
STYLE = {
    'bg': '#0a1628',
    'text': '#caf0f8',
    'accent': '#00b4d8',
    'highlight': '#ff6b35',
    'green': '#2ecc71',
    'red': '#e74c3c',
    'box_fill': '#0d2137',
    'box_edge': '#234567',
    'cmap': 'inferno',
}
