<div align="center">

# Classroom-Noise-AI

**Real-time classroom noise monitoring with speaker identification**

![Version](https://img.shields.io/badge/version-2.1.1-blue)
![Python](https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Backend](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Runs](https://img.shields.io/badge/inference-100%25%20local-success)

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Optional Components](#4-optional-components)
5. [GPU Setup (Windows 11, NVIDIA RTX 4060)](#5-gpu-setup-windows-11-nvidia-rtx-4060)
6. [Configuration](#6-configuration)
7. [Database Migration](#7-database-migration)
8. [Running the Project](#8-running-the-project)
9. [How Identification Works](#9-how-identification-works)
10. [Overlapping Speech](#10-overlapping-speech)
11. [Evaluation](#11-evaluation)
12. [API Reference](#12-api-reference)
13. [Privacy and Data Protection](#13-privacy-and-data-protection)
14. [Troubleshooting](#14-troubleshooting)
15. [Tests](#15-tests)
16. [Research Roadmap and Papers](#16-research-roadmap-and-papers)

---

## 1. Overview

**Pipeline:** Silero VAD + loudness gate → overlap detection → ECAPA-TDNN speaker embeddings (cosine similarity against registered students) → temporal tracking → grouped events → cooldowns → local TTS warning.

**Stack:** FastAPI backend · Streamlit dashboard · SQLite · SpeechBrain.

> [!IMPORTANT]
> Speaker recognition provides **estimates, not proof of identity**. A similarity score is a cosine similarity between voice embeddings. It is **not** a probability or a confidence percentage.

### Key features

| Feature | Description |
|---|---|
| Voice Activity Detection | Silero VAD with a supplementary dB gate |
| Speaker identification | ECAPA-TDNN embeddings, cosine matching with threshold and runner-up margin |
| Overlap handling | Three modes: `sepformer`, `pyannote_slots`, `mixture_embedding` (plus `auto`) |
| Temporal tracking | Tentative → confirmed identities from accumulated evidence |
| Event aggregation | Grouped events, global and per-speaker cooldowns, local TTS |
| Dashboard | Registration, live monitor, grouped alerts, analytics |
| Privacy by design | Local processing, no raw audio retention by default |

---

## 2. Architecture

The diagram shows the full V2.1 flow: audio capture, overlap detection, speaker identification, temporal tracking and alerts.

![Classroom-Noise-AI V2.1 architecture](Architecture%20Classroom-Noise-AI%20V2.1%20en%20fran%C3%A7ais.png)

> [!NOTE]
> Place `Architecture Classroom-Noise-AI V2.1 en français.png` in the same folder as this README so the image renders on GitHub. Renaming it without spaces or accents (for example `docs/architecture.png`) is recommended.

---

## 3. Quick Start

Python **3.10–3.12** recommended.

**Windows (cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> On Linux, PyAudio needs PortAudio first: `sudo apt install portaudio19-dev`.
> If PyAudio fails to build, remove it from `requirements.txt` (the default backend is `sounddevice`).

---

## 4. Optional Components

Nothing heavy is installed or loaded by default.

```bash
pip install -r requirements-optional.txt   # pyannote, Piper, Kokoro
```

| Component | Enables | Requires | Switch |
|---|---|---|---|
| pyannote segmentation | `pyannote_slots` overlap mode (who is active when) | Hugging Face token and accepted model terms | `HF_TOKEN`, `CNAI_OSD_BACKEND` |
| SepFormer (SpeechBrain, already a dependency) | `sepformer` overlap mode (isolated audio per voice) | Model download; CPU is slow, GPU recommended | `CNAI_SEPARATION_ENABLED=true` |
| Piper / Kokoro | Offline TTS | Voice `.onnx` file / `espeak-ng` | `CNAI_TTS_BACKEND` |

Without these components, the lightweight `mixture_embedding` fallback is used.

> [!WARNING]
> Model licences (pyannote, SpeechBrain checkpoints, Kokoro, Piper voices) are the user's responsibility. Check them before institutional use.

---

## 5. GPU Setup (Windows 11, NVIDIA RTX 4060)

If `torch.cuda.is_available()` returns `False`, you almost certainly have a **CPU-only torch** build (the default PyPI wheel on Windows, which `pip install -r requirements.txt` or a speechbrain/silero-vad install can reinstall).

Fix, in PowerShell inside the `.venv`:

```powershell
nvidia-smi                                   # driver must be installed and show the RTX 4060
pip uninstall -y torch torchaudio torchvision
pip cache purge
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu126
python scripts/check_gpu.py                  # must print cuda_available = True
```

- The CUDA toolkit is **not** needed: the wheel bundles the runtime. Only a recent NVIDIA driver is required.
- Install `torch` and `torchaudio` together from the same index.
- Install the rest **without touching torch**. If `torch.__version__` ends in `+cpu` after `pip install -r requirements.txt`, repeat the `cu126` command (add `--force-reinstall`).
- `GET /status` → `compute` shows the device actually in use.
- A ready-made profile is provided in `.env.rtx4060.example`.

---

## 6. Configuration

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env

python main.py devices        # list microphone / speaker indices
```

`.env.example` lists **every** setting with its default and a one-line explanation (prefix `CNAI_`). Values are validated at start-up and invalid ones are refused.

| Group | Main settings |
|---|---|
| Thresholds | `CNAI_SPEAKER_MATCH_THRESHOLD`, `CNAI_MIN_MATCH_MARGIN` |
| Temporal tracking | `CNAI_TRACKER_*` |
| Overlap | `CNAI_OVERLAP_MODE`, `CNAI_OVERLAP_SENSITIVITY`, `CNAI_MAX_SIMULTANEOUS_SPEAKERS` |
| Warnings | `CNAI_COOLDOWN_SECONDS`, `CNAI_SPEAKER_COOLDOWN_SECONDS`, `CNAI_WARNING_*`, `CNAI_TTS_ANNOUNCE_NAMES` |
| Enrollment | `CNAI_ENROLLMENT_*`, `CNAI_MIN_VOICE_SAMPLES` |
| Privacy | `CNAI_STORE_ENROLLMENT_AUDIO`, `CNAI_RAW_AUDIO_RETENTION_SECONDS` |

> [!IMPORTANT]
> **Thresholds are not universal.** `0.75` is only a starting point. Measure FAR/FRR on your own consented recordings ([section 11](#11-evaluation)) before relying on it.

---

## 7. Database Migration

```bash
python main.py init-db        # equivalent: python -m database.db_session
```

Creates missing tables and **adds missing columns** (nullable, forward-only) to databases created by older versions. Existing students, voice profiles and logs stay valid. The API also runs this at start-up.

---

## 8. Running the Project

```bash
# Terminal 1 - backend (http://127.0.0.1:8000/docs)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 - dashboard (http://localhost:8501)
streamlit run ui/app.py

# Or both at once
python main.py all
```

### Dashboard tabs

| Tab | Content |
|---|---|
| **Student Registration** | Student profiles, several voice samples each |
| **Live Monitor** | Noise level, microphone waveform, speaker timeline, latency |
| **Recent Alerts** | Grouped, expandable events |
| **Analytics and Logs** | Historical analysis and logs |

---

## 9. How Identification Works

```
capture → validation → dB gate (supplementary) → Silero VAD segments → overlap detection
        → embeddings → matching → temporal tracking → event aggregation → API/WebSocket → dashboard
```

### Enrollment

Each sample is checked (duration, level, clipping, dynamic range) and rejected with a reason if unusable. Each accepted sample gets a quality weight. The profile is the quality-weighted, normalised centroid of the latest `CNAI_SAMPLES_PER_STUDENT` samples, so one poor sample cannot dominate.

A new sample that does not resemble the student, or that matches *another* student, is refused unless `force=true`.

### Matching

- Cosine similarity is computed against each profile.
- A match is accepted only above `CNAI_SPEAKER_MATCH_THRESHOLD` **and** with a margin over the runner-up (`CNAI_MIN_MATCH_MARGIN`).
- A student is never assigned to two streams of the same window.
- Otherwise the result is `Unknown Speaker` with a `rejection_reason`.
- Segments shorter than `CNAI_MIN_IDENTIFY_SECONDS` are not identified.

### Temporal tracking

An identity is **tentative** until enough distinct speech and agreeing observations accumulate (`CNAI_TRACKER_CONFIRM_SECONDS` / `CNAI_TRACKER_CONFIRM_HITS`), then **confirmed**. Mixed-audio evidence can never confirm. Only past evidence is used; no future information.

### Events

Consecutive detections within `CNAI_EVENT_MERGE_GAP_SECONDS` form one event with stable speaker ids. A global cooldown, a per-speaker cooldown, a warning grace period and a minimum gap between warnings prevent repeated TTS.

---

## 10. Overlapping Speech

Three distinct capabilities must not be confused:

| Term | Answers | Provided by |
|---|---|---|
| Speaker **identification** | *Who* is this voice? | ECAPA-TDNN + cosine matching |
| **Diarization / overlap detection** | *When* does each speaker talk? Are several active? | pyannote segmentation, or embedding dispersion (fallback) |
| Source **separation** | The *audio* of each speaker | SepFormer (optional) |

> [!NOTE]
> ECAPA-TDNN is an embedding model: **it does not separate overlapping voices.**

### Overlap modes (`CNAI_OVERLAP_MODE`)

| Mode | What you get | Limits |
|---|---|---|
| `sepformer` | One estimated waveform per source, each embedded and matched (`signal_type=separated`) | Separation is an estimate; fixed number of sources (default model: 2); artefacts can create or merge speakers; CPU cost |
| `pyannote_slots` | Per-speaker activity and overlap timestamps; identity from regions where the speaker is alone (`masked_mixture`) | A mask on mixed audio is **not** isolated speech; speakers who never talk alone stay Unknown |
| `mixture_embedding` | Fallback: names a student only if they dominate an analysis window | Approximate; cannot confirm identities; overlap duration may be `null` |
| `auto` (default) | Best available of the above | None |

The dashboard labels waveforms accordingly: *separated audio*, *mixed microphone signal during this speaker's activity (not isolated audio)*, or timeline-only.

The system does **not** claim to identify every speaker in arbitrary overlapping speech. When evidence is insufficient, the speaker stays `Unknown Speaker`.

---

## 11. Evaluation

Evaluation uses **real data only**.

```bash
python -m evaluation.run_eval --manifest evaluation/manifest.json --out results.json
```

This runs the real models on **your** labelled, consented recordings (format described in the module docstring) and reports:

- identification accuracy, FAR/FRR and threshold sweep
- unknown-speaker rejection
- overlap precision / recall / F1 and DER
- overlap attribution and event duplication
- median / p95 latency and real-time factor

> [!NOTE]
> No benchmark numbers are shipped or claimed in this repository. `tests/` exercise the metric *code* with synthetic values only.

### Evaluation principle

Evaluate modules separately, using consented, labelled recordings representative of the intended environment. Do not lower identification thresholds solely to reduce `Unknown Speaker` results: this can increase false identifications.

---

## 12. API Reference

Interactive docs: `http://127.0.0.1:8000/docs`

### Endpoints

| Area | Endpoints |
|---|---|
| System | `GET /health` · `GET /status` · `GET /devices` · `WS /ws/stream` |
| Monitoring | `POST /monitor/start` · `POST /monitor/stop` |
| Students | `GET /students` · `GET/PATCH/DELETE /students/{id}` |
| Registration | `POST /students/register` (multipart: `name`, `age`, `student_code`, `audio`; optional: `notes`, `group_name`, `allow_duplicate_name`, `force`) |
| Voice samples | `POST /students/{id}/voice` · `DELETE /students/{id}/voice/{sample_id}` · `GET /students/{id}/voice/{sample_id}/audio` (only if enrollment audio is stored) |
| Events | `GET /events` · `GET /events/{event_id}` · `GET /live/visualization` |
| Logs and settings | `GET /logs?start=&end=&event_type=GROUP` · `GET/PUT /settings` |

### Event schema (version 2.1, additive)

Legacy keys (`event_type`, `epoch`, `student`, `match_score`, `db`, `spoken`, ...) are kept. New keys:

| Key | Description |
|---|---|
| `schema_version`, `event_id` | Schema version and unique event identifier |
| `started_at`, `ended_at`, `duration_ms` | ISO-8601 timestamps with UTC offset, duration in milliseconds |
| `speakers[]` | Per speaker: `similarity_score`, `runner_up_similarity`, `decision_status` (`confirmed` / `tentative` / `unknown`), `rejection_reason`, `signal_type` |
| `estimated_active_speaker_count`, `unknown_speaker_count` | Speaker counts |
| `overlap_duration_ms` | Overlap duration |
| `detection_method`, `identification_method` | Methods used |
| `warning_triggered`, `processing_latency_ms`, `metadata` | Warning state, latency and extras |

> [!NOTE]
> A measurement that is not available is `null`, never a fabricated `0`.

---

## 13. Privacy and Data Protection

Classroom voices, especially of minors, are sensitive. **Embeddings are biometric data too.**

- Everything runs locally; no audio leaves the machine. Edge-TTS (online fallback) sends only the warning text. Use Piper or Kokoro for fully offline operation.
- Raw classroom audio is not stored (`CNAI_RAW_AUDIO_RETENTION_SECONDS=0`).
- Enrollment WAVs are **not** kept by default (`CNAI_STORE_ENROLLMENT_AUDIO=false`); only the embedding is.
- Deleting a student removes the profile and all samples, embeddings and recordings, and anonymises the student inside stored events. Single samples can be deleted too.
- Uploads are size- and format-checked; file access is confined to the data directory.
- Spoken warnings can omit names (`CNAI_TTS_ANNOUNCE_NAMES=false`).

> [!WARNING]
> The API has **no authentication**. Bind it to `127.0.0.1` (default) or put it behind your own access control.

> [!IMPORTANT]
> Using local storage does not by itself make the system compliant. The responsible institution must review it. See [`docs/privacy-and-consent.md`](docs/privacy-and-consent.md) for the checklist (consent, institutional authorisation, retention, access control, Moroccan Law 09-08).

---

## 14. Troubleshooting

| Problem | Solution |
|---|---|
| Windows + SpeechBrain symlink error | Enable Developer Mode, or run the first start as administrator |
| `torchaudio` / `speechbrain` import errors | Install matching `torch` / `torchaudio` versions together |
| Many `Unknown Speaker` results | Enroll 2–3 samples per student in the real room, check `rejection_reason`, then tune the threshold with the evaluation workflow instead of guessing |
| Source separation unavailable | The dashboard shows the reason; monitoring continues with the fallback |
| Microphone diagnostic | `python check_mic.py [device_index]` |
| `cuda_available = False` | See [GPU Setup](#5-gpu-setup-windows-11-nvidia-rtx-4060) |

---

## 15. Tests

```bash
pip install pytest
python -m pytest tests -v
```

Heavy models are mocked. Tests that need real torch, soundfile, plotly or silero are skipped automatically.

---

## 16. Research Roadmap and Papers

This section maps the main implementation components to the research papers and resources that explain them. A linked paper provides scientific background; it does **not** mean the paper's complete method is implemented in this repository.

### Reference map

| Module | Resource | Role in this project | Type |
|---|---|---|---|
| 1. Audio capture and VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | Speech detection, segmentation, timestamps, streaming inference | Implementation and docs |
| 2. Diarization and overlap | [pyannote.audio (2019)](https://arxiv.org/abs/1911.01255) · [code](https://github.com/pyannote/pyannote-audio) | Speaker segmentation, speaker changes, overlap detection (optional mode) | Paper |
| 2. Diarization and overlap | [EEND-EDA (2021)](https://arxiv.org/abs/2106.10654) | Background on end-to-end diarization; **not integrated** in the codebase | Paper |
| 3. Speaker identification | [ECAPA-TDNN (2020)](https://arxiv.org/abs/2005.07143) · [SpeechBrain model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Speaker embeddings compared to student profiles via cosine similarity | Paper and model |
| 3. Speaker identification | [VoxCeleb (2017)](https://arxiv.org/abs/1706.08612) · [dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Understanding recording variability and noise; not the algorithm itself | Dataset paper |
| 4. Speech separation | [SepFormer (2020)](https://arxiv.org/abs/2010.13154) · [SpeechBrain model](https://huggingface.co/speechbrain/sepformer-wsj02mix) | Optional `sepformer` mode: one waveform per source before embedding and matching | Paper and model |
| 5. Evaluation | [pyannote.metrics (2017)](https://github.com/pyannote/pyannote-metrics) · [docs](https://pyannote.github.io/pyannote-metrics/) | DER, false alarms, missed speech, speaker confusion, error analysis | Toolkit |

### Module notes

**Silero VAD.** Study chunk processing, sample-rate requirements, speech thresholds and latency.

**pyannote.audio.** Pyannote-based overlap handling is optional and requires the relevant model setup and accepted model terms.

**ECAPA-TDNN.** Covers channel attention, temporal statistics and speaker verification. The similarity score is not a probability: acceptance also depends on the threshold, runner-up margin, segment quality and temporal tracking.

**VoxCeleb.** Study variation caused by recording conditions, background noise and voice capture.

**SepFormer.** Transformer-based separation with short- and long-term temporal dependencies. Separated waveforms are estimates, and artefacts can reduce identification accuracy. The default model may be designed for a fixed number of sources.

**pyannote.metrics.** Study DER, false alarms, missed speech, speaker confusion and reproducible evaluation. The project's evaluation workflow additionally reports identification accuracy, FAR/FRR, unknown rejection, overlap P/R/F1, event duplication, latency and real-time factor.

### Recommended reading order

1. **Silero VAD**: verify audio capture and speech segmentation.
2. **ECAPA-TDNN**: understand embeddings and cosine matching; start here when investigating `Unknown Speaker` results.
3. **pyannote.audio**: understand diarization and overlap detection.
4. **VoxCeleb**: learn about speaker-recognition data and recording variability.
5. **SepFormer**: study optional source separation for simultaneous speech.
6. **pyannote.metrics**: design evaluation and analyze errors before tuning thresholds.

### Responsible use

Voice recordings and speaker embeddings can be sensitive biometric data, particularly in classrooms involving minors. Obtain appropriate consent and institutional authorization, minimize retention, restrict access, and follow the [privacy and consent checklist](docs/privacy-and-consent.md) and applicable local requirements.
