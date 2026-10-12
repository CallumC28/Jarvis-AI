"""How loud the mic and Jarvis's voice are right now (0–1), for the orb to pulse with.

Written by the audio threads, read by the UI each frame; a float assignment is atomic.
"""

from __future__ import annotations

import numpy as np

mic = 0.0     # updated every 80 ms by the always-on wake word thread
voice = 0.0   # updated every 0.1 s while Jarvis speaks


def loudness(samples: np.ndarray, full_scale: float) -> float:
    """RMS on a -60…-15 dBFS scale: room noise ≈ 0, normal speech ≈ 0.5–0.9."""
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    db = 20 * np.log10(rms / full_scale + 1e-9)
    return float(np.clip((db + 60) / 45, 0.0, 1.0))
