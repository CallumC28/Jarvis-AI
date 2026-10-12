from __future__ import annotations

import numpy as np

from home_agent.voice.level import loudness


def test_loudness_is_zero_for_silence_and_rises_with_volume():
    t = np.arange(1280) / 16_000
    quiet = (300 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
    loud = (8_000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)

    assert loudness(np.zeros(1280, np.int16), 32_768) == 0.0
    assert 0.0 < loudness(quiet, 32_768) < loudness(loud, 32_768) <= 1.0
    assert loudness(np.array([], np.float32), 1.0) == 0.0
