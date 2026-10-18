"""Reproducible evaluation of speaker identification on YOUR labelled, consented recordings.

    python -m evaluation.run_eval --manifest evaluation/manifest.json --out results.json

Manifest (paths relative to the manifest file; 16-bit WAV/FLAC, any sample rate):

    {
      "enrolled": {"alice": ["enroll/alice_1.wav", "enroll/alice_2.wav"], "bob": ["enroll/bob_1.wav"]},
      "trials":   [{"file": "test/alice_a.wav", "speaker": "alice"},
                   {"file": "test/stranger.wav", "speaker": null}],
      "overlap":  [{"file": "test/mix_ab.wav", "speakers": ["alice", "bob"], "overlap": [[1.0, 3.0]]}]
    }

``trials`` are single-speaker clips (``speaker: null`` = not enrolled, an impostor). ``overlap`` clips
are processed with the same window/hop as the live pipeline. Nothing is simulated: every number in
the output comes from running the real models on the files you list. Requires the full install
(torch, speechbrain, silero-vad); optional overlap modes use whatever is enabled in your .env.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from evaluation import metrics as M


def _load(path: Path, sample_rate: int) -> np.ndarray:
    from core.audio_utils import load_audio_bytes

    return load_audio_bytes(path.read_bytes(), sample_rate).numpy().astype(np.float32)


def run(manifest_path: Path) -> Dict[str, Any]:
    import torch

    from config import Settings
    from core.multi_speaker import MultiSpeakerAnalyzer, SpeakerMatcher
    from core.OSD import OverlapSpeechDetector
    from core.speaker_id import StudentRecognizer
    from core.vad import SileroVAD
    from database.vector_store import VectorStore

    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = Settings()
    sr = settings.sample_rate
    store = VectorStore(settings.embedding_dim)
    recognizer = StudentRecognizer(settings, store)

    def embed(wav: np.ndarray) -> np.ndarray:
        return recognizer.extract_embedding(torch.from_numpy(np.ascontiguousarray(wav)))

    for name, files in manifest["enrolled"].items():
        store.set_name(name, name)
        for f in files:
            store.add(name, embed(_load(root / f, sr)))
    matcher = SpeakerMatcher(store, settings)

    # ---- single-speaker trials: raw scores + threshold decisions
    outcomes, genuine, impostor, times, durations = [], [], [], [], []
    for trial in manifest.get("trials", []):
        wav = _load(root / trial["file"], sr)
        t0 = time.perf_counter()
        emb = embed(wav)
        match = matcher.match(emb, settings.speaker_match_threshold)
        times.append((time.perf_counter() - t0) * 1000)
        durations.append(len(wav) / sr)
        outcomes.append((trial["speaker"], match.student_id if match.is_identified else None))
        scores = matcher.scores(emb)
        if trial["speaker"] is None:
            impostor.append(max(scores.values()) if scores else 0.0)
        elif trial["speaker"] in scores:
            genuine.append(scores[trial["speaker"]])

    report: Dict[str, Any] = {
        "settings": {"speaker_match_threshold": settings.speaker_match_threshold,
                     "min_match_margin": settings.min_match_margin, "samples_per_student": settings.samples_per_student},
        "identification": M.identification_metrics(outcomes),
        "threshold_sweep": M.sweep_thresholds(genuine, impostor),
        "embedding_latency": M.latency_stats(times, times, durations),
    }

    # ---- overlap clips through the real analyzer, window by window
    overlap_items = manifest.get("overlap", [])
    if overlap_items:
        vad = SileroVAD(sr, settings.vad_threshold, settings.vad_min_speech_ratio)
        osd = OverlapSpeechDetector(settings, embed_fn=recognizer.extract_embedding)
        osd.load()
        analyzer = MultiSpeakerAnalyzer(settings, store, embed)  # sepformer only if you wire a separator
        win, hop = settings.chunk_samples, settings.hop_samples
        f1s, ref_sets, hyp_sets, lat, proc, aud = [], [], [], [], [], []
        for item in overlap_items:
            wav = _load(root / item["file"], sr)
            total = len(wav) / sr
            hyp_overlap: List[M.Interval] = []
            named: set = set()
            for start in range(0, max(len(wav) - win, 0) + 1, hop):
                chunk = wav[start: start + win]
                t = torch.from_numpy(chunk)
                t0 = time.perf_counter()
                vad_r = vad.analyze(t)
                osd_r = osd.detect(t)
                analysis = analyzer.analyze(chunk, start / sr, vad_r, osd_r, settings.speaker_match_threshold,
                                            vad.frame_seconds)
                ms = (time.perf_counter() - t0) * 1000
                lat.append(ms)
                proc.append(ms)
                aud.append(len(chunk) / sr)
                hyp_overlap += analysis.overlap_segments
                named |= {o.match.student_name for o in analysis.speakers if o.is_identified}
            f1s.append(M.overlap_prf(item.get("overlap", []), hyp_overlap, total))
            ref_sets.append(item["speakers"])
            hyp_sets.append(sorted(named))
        report["overlap_detection"] = f1s
        report["overlap_attribution"] = M.attribution_accuracy(ref_sets, hyp_sets)
        report["overlap_window_latency"] = M.latency_stats(lat, proc, aud)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.manifest)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
