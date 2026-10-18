"""Continuous microphone reader producing sliding 16 kHz / mono / int16 windows."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.audio_utils import PCM16_SCALE, compute_db

logger = logging.getLogger(__name__)


class AudioStream:
    """Sliding-window microphone capture.

    The audio callback fills a ring buffer; :meth:`read_chunk` blocks until
    ``hop_seconds`` of fresh audio has arrived and then returns the latest
    ``chunk_seconds`` window as 16-bit PCM bytes. If the consumer is slower than
    real time, intermediate windows are dropped (always the freshest audio).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        device: Optional[int] = None,
        backend: str = "sounddevice",
        calibration_offset_db: float = 0.0,
        blocksize: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.hop_samples = int(sample_rate * hop_seconds)
        self.device = device
        self.backend = backend
        self.calibration_offset_db = calibration_offset_db
        self.blocksize = blocksize

        self._max_samples = self.chunk_samples * 2
        self._buffer = np.zeros(0, dtype=np.int16)
        self._new_samples = 0
        self._cond = threading.Condition()
        self._closed = True
        self.last_audio_time = 0.0
        self.last_chunk_time = 0.0  # when the latest window was handed to the consumer
        self.dropped_windows = 0  # windows skipped because the consumer was slower than real time
        self.total_samples = 0
        self._sd_stream: Any = None
        self._pa: Any = None
        self._pa_stream: Any = None

    # ------------------------------------------------------------- lifecycle
    @property
    def is_running(self) -> bool:
        return not self._closed

    def start(self) -> None:
        """Open the input device and start capturing."""
        if self.is_running:
            return
        with self._cond:
            self._buffer = np.zeros(0, dtype=np.int16)
            self._new_samples = 0
            self._closed = False
        if self.backend == "pyaudio":
            self._start_pyaudio()
        else:
            self._start_sounddevice()
        logger.info("Audio stream started (%s, device=%s)", self.backend, self.device)

    def stop(self) -> None:
        """Stop capturing and release the device."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        try:
            if self._sd_stream is not None:
                self._sd_stream.stop()
                self._sd_stream.close()
            if self._pa_stream is not None:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            if self._pa is not None:
                self._pa.terminate()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.exception("Error while closing audio stream")
        finally:
            self._sd_stream = self._pa_stream = self._pa = None

    # -------------------------------------------------------------- backends
    @staticmethod
    def resample_int16(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Cheap mono resampler (block averaging for integer ratios, interpolation otherwise)."""
        if src_sr == dst_sr or samples.size == 0:
            return samples
        x = samples.astype(np.float32)
        if src_sr % dst_sr == 0:
            k = src_sr // dst_sr
            n = (x.size // k) * k
            return x[:n].reshape(-1, k).mean(axis=1).astype(np.int16)
        n_out = int(round(x.size * dst_sr / src_sr))
        idx = np.linspace(0, x.size - 1, n_out)
        return np.interp(idx, np.arange(x.size), x).astype(np.int16)

    def _start_sounddevice(self) -> None:
        import sounddevice as sd

        dev_index = self.device if self.device is not None else sd.default.device[0]
        info = sd.query_devices(dev_index, "input")
        hostapi = sd.query_hostapis(info["hostapi"])["name"]
        native_sr = int(info["default_samplerate"])
        is_wasapi = "WASAPI" in hostapi

        # (stream sample rate, extra settings). Settings only for WASAPI devices.
        attempts = [(self.sample_rate, None)]
        if is_wasapi:
            attempts.append((self.sample_rate, sd.WasapiSettings(auto_convert=True)))
        if native_sr != self.sample_rate:
            attempts.append((native_sr, sd.WasapiSettings(auto_convert=True) if is_wasapi else None))

        errors = []
        for stream_sr, extra in attempts:
            def callback(indata, frames, time_info, status, _sr=stream_sr) -> None:  # noqa: ANN001
                if status:
                    logger.debug("sounddevice status: %s", status)
                mono = indata[:, 0]
                self._ingest(self.resample_int16(mono, _sr, self.sample_rate) if _sr != self.sample_rate else mono)

            try:
                stream = sd.InputStream(
                    samplerate=stream_sr,
                    channels=1,
                    dtype="int16",
                    blocksize=self.blocksize if stream_sr == self.sample_rate else 0,
                    device=self.device,
                    callback=callback,
                    extra_settings=extra,
                )
                stream.start()
            except Exception as exc:
                errors.append(f"{stream_sr} Hz ({hostapi}{', auto_convert' if extra else ''}): {exc}")
                continue
            self._sd_stream = stream
            logger.info("Input '%s' [%s] opened at %d Hz", info["name"], hostapi, stream_sr)
            return
        raise RuntimeError(
            f"Cannot open input device {dev_index} '{info['name']}' [{hostapi}]. Tried: " + " | ".join(errors)
        )

    def _start_pyaudio(self) -> None:
        import pyaudio

        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            self._ingest(np.frombuffer(in_data, dtype=np.int16))
            return (None, pyaudio.paContinue)

        self._pa = pyaudio.PyAudio()
        self._pa_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=self.blocksize,
            stream_callback=callback,
        )
        self._pa_stream.start_stream()

    # ----------------------------------------------------------------- data
    def _ingest(self, samples: np.ndarray) -> None:
        with self._cond:
            if self._closed:
                return
            self._buffer = np.concatenate((self._buffer, samples.astype(np.int16)))[-self._max_samples :]
            self._new_samples += len(samples)
            self.total_samples += len(samples)
            self.last_audio_time = time.time()
            self._cond.notify_all()

    def read_chunk(self, timeout: Optional[float] = 1.0) -> Optional[bytes]:
        """Return the latest window as PCM16 bytes, or ``None`` on timeout / stop."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._closed
                or (self._new_samples >= self.hop_samples and len(self._buffer) >= self.chunk_samples),
                timeout=timeout,
            )
            if not ready or self._closed:
                return None
            missed = self._new_samples // self.hop_samples - 1
            if missed > 0:
                self.dropped_windows += missed
            self._new_samples = 0
            self.last_chunk_time = time.time()
            return self._buffer[-self.chunk_samples :].tobytes()

    def recent_db(self, seconds: float = 0.5) -> float:
        """Pseudo-dB level of the most recent ``seconds`` (for the live gauge)."""
        n = int(self.sample_rate * seconds)
        with self._cond:
            tail = self._buffer[-n:].copy()
        if tail.size == 0:
            return 0.0
        return compute_db(tail.astype(np.float32) / PCM16_SCALE, self.calibration_offset_db)

    # --------------------------------------------------------------- devices
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """List audio devices visible to PortAudio."""
        import sounddevice as sd

        devices = []
        for index, dev in enumerate(sd.query_devices()):
            devices.append(
                {
                    "index": index,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "max_output_channels": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
        return devices
