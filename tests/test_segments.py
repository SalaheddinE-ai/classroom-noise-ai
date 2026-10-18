"""Pure helpers: segments, masks, envelopes."""
import numpy as np
import pytest

from core.segments import (
    concat_segments, envelope, frame_rms, intersect_segments, mask_to_segments, merge_segments,
    overlap_from_masks, segments_to_mask, subtract_segments, total_duration,
)


def test_mask_to_segments_and_merge():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    segs = mask_to_segments(mask, 0.5)
    assert segs == [(0.5, 1.5), (2.5, 3.0), (3.5, 5.0)]
    assert merge_segments(segs, gap=0.5) == [(0.5, 1.5), (2.5, 5.0)]
    assert mask_to_segments(np.zeros(0), 0.1) == []


def test_subtract_intersect_total():
    a, b = [(0.0, 2.0)], [(0.5, 1.0), (1.5, 3.0)]
    assert subtract_segments(a, b) == [(0.0, 0.5), (1.0, 1.5)]
    assert intersect_segments(a, b) == [(0.5, 1.0), (1.5, 2.0)]
    assert total_duration([(0, 1), (0.5, 2)]) == pytest.approx(2.0)
    assert subtract_segments([(0, 1)], [(0, 1)]) == []


def test_concat_and_mask_and_envelope():
    sr = 100
    x = np.arange(300, dtype=np.float32)
    assert concat_segments(x, [(0.0, 0.5), (2.0, 2.5)], sr).tolist() == list(range(0, 50)) + list(range(200, 250))
    assert segments_to_mask([(0.5, 1.0)], 200, sr).sum() == 50
    env = envelope(np.concatenate([np.zeros(100), np.ones(100)]), 4)
    assert env == [0.0, 0.0, 1.0, 1.0]
    assert envelope(np.zeros(50), 5) == [0.0] * 5 and envelope(np.zeros(0), 5) == []


def test_overlap_from_masks_and_frame_rms():
    m1 = np.array([1, 1, 1, 0, 0], dtype=bool)
    m2 = np.array([0, 0, 1, 1, 0], dtype=bool)
    assert overlap_from_masks([m1, m2], 0.1) == [pytest.approx((0.2, 0.3))]
    assert overlap_from_masks([m1], 0.1) == []
    r = frame_rms(np.concatenate([np.zeros(160), np.ones(160)]).astype(np.float32), 16000, 10)
    assert r.tolist() == [0.0, 1.0]
