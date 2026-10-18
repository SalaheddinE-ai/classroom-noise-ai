"""Real Silero VAD (weights are bundled in the pip package)."""
import numpy as np
import pytest

import os
if os.environ.get("CNAI_TEST_REAL_TORCH") != "1":
    pytest.skip("needs real torch", allow_module_level=True)
import torch
pytest.importorskip("silero_vad")

from core.vad import SileroVAD


@pytest.fixture(scope="module")
def vad():
    return SileroVAD()


def test_silence_and_noise_are_not_speech(vad):
    assert not vad.analyze(torch.zeros(32000)).is_speech
    torch.manual_seed(0)
    assert not vad.analyze(torch.randn(32000) * 0.3).is_speech


def test_short_input_handled(vad):
    assert vad.analyze(torch.zeros(100)).speech_ratio == 0.0


def test_real_speech_detected_if_available(vad):
    """Uses silero's own test wav when present in the package."""
    import glob, os, silero_vad, soundfile as sf
    files = glob.glob(os.path.join(os.path.dirname(silero_vad.__file__), "**", "*.wav"), recursive=True)
    if not files:
        pytest.skip("no sample wav bundled")
    data, sr = sf.read(files[0], dtype="float32")
    assert sr == 16000
    assert vad.analyze(torch.from_numpy(data[:32000])).max_prob > 0.5
