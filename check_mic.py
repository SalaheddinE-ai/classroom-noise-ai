"""Microphone diagnostic.

    python check_mic.py            # tests EVERY input device (3 s each) - keep talking
    python check_mic.py 15         # tests only device 15

Prints the level (pseudo-dB) and Silero VAD speech ratio per device.
"""
import sys

import numpy as np
import sounddevice as sd
import torch

from core.audio_utils import compute_db
from core.vad import SileroVAD

SECONDS = 3
SR = 16000


def test_device(index: int, name: str, vad: SileroVAD) -> None:
    try:
        from core.audio_stream import AudioStream
        stream = AudioStream(sample_rate=SR, device=index)
        stream.start()
        import time as _t
        _t.sleep(SECONDS)
        stream.stop()
        rec = stream._buffer.reshape(-1, 1)
        if rec.shape[0] < SR:
            raise RuntimeError("no audio frames received")
    except Exception as exc:
        print(f"[{index:>2}] FAILED  {name}  ->  {str(exc)[:160]}")
        return
    x = rec[:, 0].astype(np.float32) / 32768.0
    db = compute_db(x)
    n = max(1, len(x) // SR)
    ratios = [vad.analyze(torch.from_numpy(x[i * SR:(i + 1) * SR])).speech_ratio for i in range(n)]
    status = "SILENCE" if np.abs(x).max() < 0.001 else ("OK" if max(ratios) >= 0.3 else "audio, no speech")
    print(f"[{index:>2}] {status:<16} {db:5.1f} dB  speech {max(ratios):4.0%}  peak {np.abs(x).max():.3f}  {name}")


def main() -> None:
    vad = SileroVAD()
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        targets = [(idx, sd.query_devices(idx)["name"])]
    else:
        targets = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    print(f"Speak continuously during each {SECONDS}-second test.\n")
    for index, name in targets:
        test_device(index, name, vad)
    print("\nPut the index of the best device in .env:  CNAI_INPUT_DEVICE=<index>")


if __name__ == "__main__":
    main()
