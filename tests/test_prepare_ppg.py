"""Regression tests for BUT PPG PPG-channel extraction.

BUT PPG v2.0.0 stores PPG in two shapes: most records are 3-channel RGB
smartphone signals (300, 3) with PPG_R/PPG_G/PPG_B; a few early records are a
non-standard single-channel header (1, 300). An earlier version only accepted
the single-channel form and silently skipped ~98% of the dataset — these tests
lock in that both shapes yield the green 300-sample window.
"""

from __future__ import annotations

import numpy as np

from butppg.data.prepare import PPG_LEN, _extract_ppg


class _Rec:
    def __init__(self, p_signal, sig_name):
        self.p_signal = p_signal
        self.sig_name = sig_name


def test_rgb_record_picks_green_channel():
    rgb = np.zeros((PPG_LEN, 3), dtype=np.float32)
    rgb[:, 1] = np.arange(PPG_LEN)  # green ramp, red/blue zero
    rec = _Rec(rgb, [".u. .. PPG_R", ".u. .. PPG_G", ".u. .. PPG_B"])
    ppg = _extract_ppg(rec)
    assert ppg.shape == (PPG_LEN,)
    assert np.allclose(ppg, np.arange(PPG_LEN))  # green, not red/blue


def test_legacy_single_channel_flattens():
    leg = np.arange(PPG_LEN, dtype=np.float32).reshape(1, PPG_LEN)
    rec = _Rec(leg, [".u. 0 0 0 0 0"] * PPG_LEN)
    ppg = _extract_ppg(rec)
    assert ppg.shape == (PPG_LEN,)
    assert np.allclose(ppg, np.arange(PPG_LEN))


def test_rgb_without_names_falls_back_to_index_1():
    rgb = np.zeros((PPG_LEN, 3), dtype=np.float32)
    rgb[:, 1] = np.arange(PPG_LEN)
    rec = _Rec(rgb, ["a", "b", "c"])
    assert np.allclose(_extract_ppg(rec), np.arange(PPG_LEN))
