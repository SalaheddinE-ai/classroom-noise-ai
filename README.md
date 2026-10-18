**\# classroom-noise-ai (V2.1.1)**

## Architecture globale du projet

L’illustration ci-dessous présente l’architecture globale de Classroom-Noise-AI V2.1 : de la capture audio à la détection des chevauchements de voix, à l’identification des locuteurs, au suivi temporel et aux alertes.

![Architecture globale de Classroom-Noise-AI V2.1](Architecture%20Classroom-Noise-AI%20V2.1%20en%20fran%C3%A7ais.png)

> **Important :** placez le fichier image `Architecture Classroom-Noise-AI V2.1 en français.png` dans le même dossier que ce README pour que l’image s’affiche correctement sur GitHub.

Real-time classroom noise monitoring with **\*\*speaker
identification\*\***: Silero VAD + loudness gate -\>

overlap detection -\> ECAPA-TDNN speaker embeddings (cosine similarity
against registered students) -\>

temporal tracking -\> grouped events -\> cooldowns -\> local TTS
warning. FastAPI backend, Streamlit dashboard.

\> Speaker recognition gives **\*\*estimates, not proof of
identity\*\***. A similarity score is a cosine

\> similarity between voice embeddings - it is **\*\*not\*\*** a
probability or a confidence percentage.

**\## 1. Install (Python 3.10-3.12 recommended)**

Windows (cmd):

    python -m venv .venv

    .venv`\Scripts`{=tex}`\activate`{=tex}

    python -m pip install --upgrade pip

    pip install -r requirements.txt

Linux / macOS:

    python3 -m venv .venv && source .venv/bin/activate

    python -m pip install --upgrade pip

    pip install -r requirements.txt

PyAudio on Linux needs PortAudio first: \`sudo apt install
portaudio19-dev\`.

If PyAudio fails to build, remove it from requirements.txt (default
backend is sounddevice).

**\## 2. Optional components (nothing heavy is installed or loaded by
default)**

    pip install -r requirements-optional.txt          # pyannote, Piper,
Kokoro

\| Component \| Enables \| Needs \| Switch \|

\|---\|---\|---\|---\|

\| pyannote segmentation \| \`pyannote_slots\` overlap mode (who is
active when) \| Hugging Face token + accepted model terms \|
\`HF_TOKEN\`, \`CNAI_OSD_BACKEND\` \|

\| SepFormer (SpeechBrain, already a dependency) \| \`sepformer\`
overlap mode (isolated audio per voice) \| model download, CPU is slow,
GPU recommended \| \`CNAI_SEPARATION_ENABLED=true\` \|

\| Piper / Kokoro \| offline TTS \| voice \`.onnx\` file / \`espeak-ng\`
\| \`CNAI_TTS_BACKEND\` \|

Without them the lightweight \`mixture_embedding\` fallback is used.
Model licences (pyannote, SpeechBrain

checkpoints, Kokoro, Piper voices) are the user's responsibility - check
them before institutional use.

**\## 2b. GPU on Windows 11 (NVIDIA RTX 4060)**

A \`False\` from \`torch.cuda.is_available()\` almost always means a
**\*\*CPU-only torch\*\*** (the default wheel on

PyPI for Windows, which \`pip install -r requirements.txt\` or
speechbrain/silero-vad reinstall). Fix, in PowerShell

inside the \`.venv\`:

    nvidia-smi                                   \# driver must be
installed and show the RTX 4060

    pip uninstall -y torch torchaudio torchvision

    pip cache purge

    pip install --no-cache-dir torch torchaudio --index-url
https://download.pytorch.org/whl/cu126

    python scripts/check_gpu.py                  # must print
cuda_available = True

The CUDA toolkit is NOT needed (the wheel bundles the runtime), only a
recent NVIDIA driver. Install torch and

torchaudio together from the same index. Afterwards install the rest
**\*\*without touching torch\*\***:

\`pip install -r requirements.txt\` may replace it again - if
\`torch.\_\_version\_\_\` ends in \`+cpu\`, repeat the

\`pip install ... cu126\` line (add \`--force-reinstall\`). \`/status\`
-\> \`compute\` shows the device really used.

A ready profile is in \`.env.rtx4060.example\`.

**\## 3. Configure**

    copy .env.example .env        (Windows)      \|      cp .env.example
.env   (Linux/macOS)

    python main.py devices        # find microphone / speaker indices

\`.env.example\` lists **\*\*every\*\*** setting with its default and a
one-line explanation (\`CNAI\_\` prefix).

Important groups: thresholds (\`CNAI_SPEAKER_MATCH_THRESHOLD\`,
\`CNAI_MIN_MATCH_MARGIN\`), temporal tracking

(\`CNAI_TRACKER\_\*\`), overlap (\`CNAI_OVERLAP_MODE\`,
\`CNAI_OVERLAP_SENSITIVITY\`, \`CNAI_MAX_SIMULTANEOUS_SPEAKERS\`),

warnings (\`CNAI_COOLDOWN_SECONDS\`, \`CNAI_SPEAKER_COOLDOWN_SECONDS\`,
\`CNAI_WARNING\_\*\`, \`CNAI_TTS_ANNOUNCE_NAMES\`),

enrollment (\`CNAI_ENROLLMENT\_\*\`, \`CNAI_MIN_VOICE_SAMPLES\`),
privacy (\`CNAI_STORE_ENROLLMENT_AUDIO\`,

\`CNAI_RAW_AUDIO_RETENTION_SECONDS\`). Values are validated at start-up;
invalid ones are refused.

**\*\*Thresholds are not universal.\*\*** 0.75 is only a starting point:
measure FAR/FRR on your own consented

recordings (section 8) before relying on it.

**\## 4. Database migration**

    python main.py init-db        # equivalent: python -m
database.db_session

Creates missing tables and **\*\*adds missing columns\*\*** (nullable,
forward-only) to databases made by older

versions; existing students, voice profiles and logs stay valid. The API
also does this at start-up.

**\## 5. Run**

    \# Terminal 1 - backend (http://127.0.0.1:8000/docs)

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

    \# Terminal 2 - dashboard (http://localhost:8501)

    streamlit run ui/app.py

    \# or both: python main.py all

Tabs: **\*\*Student Registration\*\*** (profiles, several samples each),
**\*\*Live Monitor\*\*** (noise level, microphone

waveform, speaker timeline, latency), **\*\*Recent Alerts\*\***
(grouped, expandable events), **\*\*Analytics and Logs\*\***.

**\## 6. How identification works**

Pipeline: \`capture -\> validation -\> dB gate (supplementary) -\>
Silero VAD segments -\> overlap detection -\>

embeddings -\> matching -\> temporal tracking -\> event aggregation -\>
API/WebSocket -\> dashboard\`.

\* **\*\*Enrollment.\*\*** Each sample is checked (duration, level,
clipping, dynamic range) and rejected with a

  reason if unusable. A sample gets a quality weight; the profile is the
quality-weighted, normalised

  centroid of the latest \`CNAI_SAMPLES_PER_STUDENT\` samples, so one
poor sample cannot dominate. A new sample

  that does not resemble the student, or that matches *\*another\**
student, is refused unless \`force=true\`.

\* **\*\*Matching.\*\*** Cosine similarity to each profile; accepted
only above \`CNAI_SPEAKER_MATCH_THRESHOLD\` **\*\*and\*\***

  with a margin over the runner-up (\`CNAI_MIN_MATCH_MARGIN\`); a
student is never assigned to two streams of the

  same window. Otherwise the result is \`Unknown Speaker\` with a
\`rejection_reason\`. Segments shorter than

  \`CNAI_MIN_IDENTIFY_SECONDS\` are not identified.

\* **\*\*Temporal tracking.\*\*** An identity is **\*\*tentative\*\***
until enough distinct speech and agreeing observations

  accumulate (\`CNAI_TRACKER_CONFIRM_SECONDS/HITS\`), then
**\*\*confirmed\*\***. Mixed-audio evidence can never confirm.

  Past evidence only - no future information is used.

\* **\*\*Events.\*\*** Consecutive detections within
\`CNAI_EVENT_MERGE_GAP_SECONDS\` form one event with stable speaker ids.

  Global cooldown, per-speaker cooldown, a warning grace period and a
minimum gap between warnings prevent

  repeated TTS.

**\## 7. Overlapping speech - what each mode can and cannot do**

Three different things must not be confused:

\| Term \| Answers \| Provided by \|

\|---\|---\|---\|

\| Speaker **\*\*identification\*\*** \| *\*who\** is this voice? \|
ECAPA-TDNN + cosine matching \|

\| **\*\*Diarization / overlap detection\*\*** \| *\*when\** does each
speaker talk, are several active? \| pyannote segmentation, or embedding
dispersion (fallback) \|

\| Source **\*\*separation\*\*** \| the *\*audio\** of each speaker \|
SepFormer (optional) \|

ECAPA-TDNN is an embedding model: **\*\*it does not separate overlapping
voices.\*\***

\| \`CNAI_OVERLAP_MODE\` \| What you get \| Limits \|

\|---\|---\|---\|

\| \`sepformer\` \| one estimated waveform per source, each embedded and
matched (\`signal_type=separated\`) \| separation is an estimate; fixed
number of sources (default model: 2); artefacts can create or merge
speakers; CPU cost \|

\| \`pyannote_slots\` \| per-speaker activity, overlap timestamps;
identity from regions where the speaker is alone (\`masked_mixture\`) \|
a mask on mixed audio is **\*\*not\*\*** isolated speech; speakers who
never talk alone stay Unknown \|

\| \`mixture_embedding\` \| fallback: names a student only if they
dominate an analysis window \| approximate; cannot confirm identities;
overlap duration may be \`null\` \|

\| \`auto\` (default) \| best available of the above \| - \|

The dashboard labels waveforms accordingly: "separated audio" vs "mixed
microphone signal during this speaker's

activity (not isolated audio)" vs timeline-only. The system does
**\*\*not\*\*** claim to identify every speaker in

arbitrary overlapping speech; when evidence is insufficient the speaker
stays \`Unknown Speaker\`.

**\## 8. Evaluation (real data only)**

    python -m evaluation.run_eval --manifest evaluation/manifest.json
--out results.json

Runs the real models on **\*\*your\*\*** labelled, consented recordings
(format in the module docstring) and reports

identification accuracy, FAR/FRR and threshold sweep, unknown rejection,
overlap precision/recall/F1, DER,

overlap attribution, event duplication, median/p95 latency and real-time
factor. No benchmark numbers are

shipped or claimed in this repository; \`tests/\` exercise the metric
*\*code\** with synthetic values only.

**\## 9. API quick reference**

    GET  /health   GET /status   WS /ws/stream   POST /monitor/start  
POST /monitor/stop   GET /devices

    GET  /students                  POST /students/register (multipart:
name, age, student_code, audio;

                                          optional: notes, group_name,
allow_duplicate_name, force)

    GET/PATCH/DELETE /students/{id}   POST /students/{id}/voice   DELETE
/students/{id}/voice/{sample_id}

    GET  /students/{id}/voice/{sample_id}/audio     (only if enrollment
audio is stored)

    GET  /events   GET /events/{event_id}   GET /live/visualization

    GET  /logs?start=&end=&event_type=GROUP         GET/PUT /settings

**\*\*Event schema (version 2.1, additive).\*\*** Legacy keys
(\`event_type\`, \`epoch\`, \`student\`, \`match_score\`, \`db\`,

\`spoken\`, ...) are kept; new keys include \`schema_version\`,
\`event_id\`, \`started_at\`, \`ended_at\`, \`duration_ms\`,

\`speakers\[\]\` (\`similarity_score\`, \`runner_up_similarity\`,
\`decision_status\` = confirmed/tentative/unknown,

\`rejection_reason\`, \`signal_type\`),
\`estimated_active_speaker_count\`, \`unknown_speaker_count\`,

\`overlap_duration_ms\`, \`detection_method\`,
\`identification_method\`, \`warning_triggered\`,

\`processing_latency_ms\`, \`metadata\`. Timestamps are ISO-8601 with
UTC offset, durations are milliseconds,

and a measurement that is not available is \`null\` (never a fabricated
0).

**\## 10. Privacy and data protection**

Classroom voices - especially of minors - are sensitive.
**\*\*Embeddings are biometric data too.\*\***

\* Everything runs locally; no audio leaves the machine. Edge-TTS
(online fallback) sends only the warning

  text; use Piper/Kokoro for fully offline operation.

\* Raw classroom audio is not stored
(\`CNAI_RAW_AUDIO_RETENTION_SECONDS=0\`). Enrollment WAVs are
**\*\*not\*\*** kept by

  default (\`CNAI_STORE_ENROLLMENT_AUDIO=false\`); only the embedding
is.

\* Deleting a student removes the profile, all
samples/embeddings/recordings, and anonymises the student inside

  stored events. Single samples can be deleted too.

\* Uploads are size- and format-checked; file access is confined to the
data directory.

\* The API has **\*\*no authentication\*\***: bind it to \`127.0.0.1\`
(default) or put it behind your own access control.

\* Spoken warnings can omit names (\`CNAI_TTS_ANNOUNCE_NAMES=false\`).

See \`docs/privacy-and-consent.md\` for the checklist (consent,
institutional authorisation, retention, access

control, Moroccan Law 09-08). **\*\*Using local storage does not by
itself make the system compliant\*\*** - the

responsible institution must review it.

**\## 11. Troubleshooting**

\* Windows + SpeechBrain symlink error: enable Developer Mode or run the
first start as administrator.

\* torchaudio/speechbrain import errors: install matching \`torch\` /
\`torchaudio\` versions together.

\* Many "Unknown Speaker": enroll 2-3 samples per student in the real
room, check \`rejection_reason\`, then tune

  the threshold with the evaluation workflow instead of guessing.

\* Source separation unavailable: the dashboard shows the reason;
monitoring continues with the fallback.

\* Microphone diagnostic: \`python check_mic.py \[device_index\]\`.

**\## 12. Tests**

    pip install pytest

    python -m pytest tests -v

Heavy models are mocked; tests that need real torch / soundfile / plotly
/ silero are skipped automatically.

------------------------------------------------------------------------

## 13. Research roadmap and papers

This section maps the project's main implementation components to the
research papers and technical resources that explain them. A linked
paper provides scientific background; it does not by itself mean that
the paper's complete method is implemented in this repository.

### Module 1 --- Audio capture and Voice Activity Detection (VAD)

**Silero VAD --- official repository and documentation**\
Resource: https://github.com/snakers4/silero-vad

-   **Purpose in this project:** speech/non-speech detection,
    segmentation, timestamps, and streaming inference.
-   **Study:** chunk processing, sample-rate requirements, speech
    thresholds, and latency.
-   **Reference type:** implementation and documentation, not a
    conventional research paper.

### Module 2 --- Speaker diarization and overlap detection

**Paper 1 --- pyannote.audio: Neural Building Blocks for Speaker
Diarization (2019)**\
Paper: https://arxiv.org/abs/1911.01255\
Implementation: https://github.com/pyannote/pyannote-audio

-   **Purpose in this project:** understand speaker segmentation,
    speaker changes, and overlapping-speech detection.
-   **Implementation note:** pyannote-based overlap handling is optional
    and requires the relevant model setup and accepted model terms.

**Paper 2 --- Encoder-Decoder Based Attractors for End-to-End Neural
Diarization (EEND-EDA, 2021)**\
Paper: https://arxiv.org/abs/2106.10654

-   **Purpose in this project:** background on end-to-end diarization
    and representing multiple active speakers over time.
-   **Implementation note:** this paper is a research reference; it does
    not imply that EEND-EDA is currently integrated into the codebase.

### Module 3 --- Speaker identification with ECAPA-TDNN

**Paper 3 --- ECAPA-TDNN: Emphasized Channel Attention, Propagation and
Aggregation in TDNN Based Speaker Verification (2020)**\
Paper: https://arxiv.org/abs/2005.07143\
SpeechBrain pretrained model:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

-   **Purpose in this project:** speaker embeddings, channel attention,
    temporal statistics, and speaker verification.
-   **Project connection:** the system compares an incoming segment's
    embedding with registered student profiles using cosine similarity.
-   **Important limitation:** the similarity score is not a probability.
    Acceptance also depends on the configured threshold, runner-up
    margin, segment quality, and temporal tracking.

**Paper 4 --- VoxCeleb: A Large-Scale Speaker Identification Dataset
(2017)**\
Paper: https://arxiv.org/abs/1706.08612\
Dataset website: https://www.robots.ox.ac.uk/\~vgg/data/voxceleb/

-   **Purpose in this project:** understand speaker-recognition datasets
    and variability in real-world speech.
-   **Study:** variation caused by recording conditions, background
    noise, and how a person's voice is captured.
-   **Reference type:** dataset paper; VoxCeleb is not the
    identification algorithm itself.

### Module 4 --- Separation of simultaneous speech

**Paper 5 --- Attention Is All You Need in Speech Separation (SepFormer,
2020)**\
Paper: https://arxiv.org/abs/2010.13154\
SpeechBrain pretrained model:
https://huggingface.co/speechbrain/sepformer-wsj02mix

-   **Purpose in this project:** understand Transformer-based speech
    separation and short-/long-term temporal dependencies.
-   **Project connection:** the optional `sepformer` mode estimates a
    separate waveform for each source before speaker embedding and
    matching.
-   **Important limitation:** separated waveforms are estimates;
    artifacts can reduce identification accuracy. The default model
    configuration may be designed for a fixed number of sources.

### Module 5 --- Evaluation and error analysis

**Paper 6 --- pyannote.metrics: A Toolkit for Reproducible Evaluation,
Diagnostic, and Error Analysis of Speaker Diarization Systems (2017)**\
Project and documentation: https://github.com/pyannote/pyannote-metrics\
Documentation: https://pyannote.github.io/pyannote-metrics/

-   **Purpose in this project:** evaluate diarization and diagnose
    errors across pipeline stages.
-   **Study:** Diarization Error Rate (DER), false alarms, missed
    speech, speaker confusion, and reproducible evaluation.
-   **Project connection:** the evaluation workflow also reports
    identification accuracy, FAR/FRR, unknown-speaker rejection, overlap
    precision/recall/F1, event duplication, latency, and real-time
    factor.

### Recommended reading order

1.  **Silero VAD documentation** --- verify audio capture and speech
    segmentation.
2.  **ECAPA-TDNN** --- understand embeddings and cosine matching; start
    here when investigating `Unknown Speaker` results.
3.  **pyannote.audio** --- understand diarization and overlap detection.
4.  **VoxCeleb** --- learn about speaker-recognition data and recording
    variability.
5.  **SepFormer** --- study optional source separation for simultaneous
    speech.
6.  **pyannote.metrics** --- design evaluation and analyze errors before
    tuning thresholds.

### Evaluation principle

Evaluate the modules separately and use consented, labelled recordings
representative of the intended environment. Measure false acceptance and
false rejection rates (FAR/FRR), unknown-speaker rejection, diarization
error, overlap detection, end-to-end latency, and real-time factor. Do
not lower identification thresholds solely to reduce `Unknown Speaker`
results: this can increase false identifications.

### Privacy and responsible use

Voice recordings and speaker embeddings can be sensitive biometric data,
particularly in classrooms involving minors. Obtain appropriate consent
and institutional authorization, minimize retention, restrict access,
and follow the project's `docs/privacy-and-consent.md` checklist and
applicable local requirements.
