"""Asynchronous local TTS (Piper / Kokoro-82M, with Edge-TTS fallback)."""
from __future__ import annotations

import asyncio
import logging
import random
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Settings
from database.event_types import EventType

logger = logging.getLogger(__name__)

TEMPLATES: Dict[str, Dict[EventType, List[str]]] = {
    "en": {
        EventType.SINGLE_STUDENT: [
            "{name}, please lower your voice.",
            "{name}, you are a bit loud. Please quiet down.",
            "Excuse me {name}, let's keep our voices down.",
            "{name}, please respect the classroom and speak softly.",
            "A gentle reminder, {name}: quiet voices, please.",
        ],
        EventType.GROUP: [
            "Everyone, please lower your voices. The whole class is too loud.",
            "Class, too many people are talking at once. Please quiet down.",
            "Let's all take a moment and keep our voices down, please.",
            "Several conversations are going on at the same time. Please settle down.",
            "Attention everyone, the noise level is too high. Quiet voices, please.",
        ],
        EventType.GENERAL: [
            "Please keep the noise down, class.",
            "The classroom is getting loud. Let's lower our voices.",
            "A gentle reminder to use quiet voices, everyone.",
            "The noise level is high. Please focus and speak softly.",
            "Let's keep the room calm and quiet, please.",
        ],
    },
    "fr": {
        EventType.SINGLE_STUDENT: [
            "{name}, merci de baisser la voix.",
            "{name}, tu es un peu bruyant. Merci de te calmer.",
            "Excuse-moi {name}, parlons plus doucement.",
            "{name}, merci de respecter la classe et de parler doucement.",
            "Petit rappel, {name} : on parle doucement, s'il te plaît.",
        ],
        EventType.GROUP: [
            "Tout le monde, merci de baisser la voix. La classe est trop bruyante.",
            "La classe, trop de personnes parlent en même temps. Merci de vous calmer.",
            "Prenons tous un instant et gardons une voix basse, s'il vous plaît.",
            "Plusieurs conversations ont lieu en même temps. Merci de vous calmer.",
            "Attention tout le monde, le niveau sonore est trop élevé. Voix basses, s'il vous plaît.",
        ],
        EventType.GENERAL: [
            "Merci de faire moins de bruit, la classe.",
            "La classe devient bruyante. Baissons la voix.",
            "Petit rappel à tous : on utilise une voix douce.",
            "Le niveau sonore est élevé. Merci de vous concentrer et de parler doucement.",
            "Gardons une classe calme et silencieuse, s'il vous plaît.",
        ],
    },
}


# Used when several students are identified in one overlapping-speech event.
NAMED_GROUP_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        "{names}, please lower your voices.",
        "{names}, you are talking at the same time. Please quiet down.",
        "A gentle reminder to {names}: quiet voices, please.",
    ],
    "fr": [
        "{names}, merci de baisser la voix.",
        "{names}, vous parlez en même temps. Merci de vous calmer.",
        "Petit rappel pour {names} : on parle doucement, s'il vous plaît.",
    ],
}
_AND = {"en": "and", "fr": "et"}


class TTSEngine:
    """Speaks intervention messages without blocking the audio-processing thread.

    Backends are tried in order (``auto``: Piper -> Kokoro -> Edge-TTS). Synthesis and
    playback run on a dedicated asyncio loop living in a daemon thread.
    """

    GRACE_SECONDS = 1.5  # ignore the room just after playback (speaker echo)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.language = settings.language
        self.volume = settings.tts_volume
        self._speaking = threading.Event()
        self._grace_until = 0.0
        self._kokoro = None
        self._kokoro_lang: Optional[str] = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="tts-loop", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- state
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def is_busy(self) -> bool:
        """True while speaking and shortly afterwards."""
        return self._speaking.is_set() or time.monotonic() < self._grace_until

    def close(self) -> None:
        """Stop the background loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------ templates
    def render_message(
        self,
        event_type: EventType,
        student_name: Optional[str] = None,
        names: Optional[List[str]] = None,
    ) -> str:
        """Pick a random template variation for the event type.

        ``names`` (identified students of an overlapping-speech event) selects the named group
        templates; without it the behaviour is unchanged.
        """
        if event_type == EventType.GROUP and names:
            lang = self.language if self.language in NAMED_GROUP_TEMPLATES else "en"
            joiner = f" {_AND[lang]} "
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + joiner + names[-1]
            return random.choice(NAMED_GROUP_TEMPLATES[lang]).format(names=joined)
        if event_type == EventType.SINGLE_STUDENT and not student_name:
            event_type = EventType.GENERAL
        pool = TEMPLATES.get(self.language, TEMPLATES["en"])[event_type]
        return random.choice(pool).format(name=student_name or "")

    # ------------------------------------------------------------- speaking
    def speak_nowait(self, text: str) -> None:
        """Schedule ``text`` to be spoken and return immediately (thread-safe)."""
        self._speaking.set()
        asyncio.run_coroutine_threadsafe(self.speak(text), self._loop)

    async def speak(self, text: str) -> bool:
        """Synthesise and play ``text``. Returns ``True`` if audio was played."""
        self._speaking.set()
        try:
            loop = asyncio.get_running_loop()
            audio, sr = await self._synthesize(text)
            if audio is None:
                return False
            audio = np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)
            await loop.run_in_executor(None, self._play, audio, sr)
            return True
        except Exception:
            logger.exception("TTS playback failed")
            return False
        finally:
            self._grace_until = time.monotonic() + self.GRACE_SECONDS
            self._speaking.clear()

    def _play(self, audio: np.ndarray, sr: int) -> None:
        import sounddevice as sd

        sd.play(audio, sr, device=self.settings.output_device)
        sd.wait()

    # ----------------------------------------------------------- synthesis
    def _backend_order(self) -> List[str]:
        wanted = self.settings.tts_backend
        if wanted != "auto":
            return [wanted] + [b for b in ("piper", "kokoro", "edge") if b != wanted]
        return ["piper", "kokoro", "edge"]

    async def _synthesize(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        loop = asyncio.get_running_loop()
        for backend in self._backend_order():
            try:
                if backend == "piper":
                    return await loop.run_in_executor(None, self._synth_piper, text)
                if backend == "kokoro":
                    return await loop.run_in_executor(None, self._synth_kokoro, text)
                if backend == "edge":
                    return await self._synth_edge(text)
            except Exception as exc:
                logger.info("TTS backend '%s' unavailable: %s", backend, exc)
        logger.error("No TTS backend could synthesise the message")
        return None, 0

    def _resolve_piper_model(self) -> Path:
        if self.settings.piper_model_path and Path(self.settings.piper_model_path).exists():
            return Path(self.settings.piper_model_path)
        folder = self.settings.models_dir / "piper"
        candidates = sorted(folder.glob(f"{self.language}_*.onnx")) if folder.exists() else []
        if not candidates:
            raise FileNotFoundError(f"No Piper voice ({self.language}_*.onnx) in {folder}")
        return candidates[0]

    def _synth_piper(self, text: str) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        model = self._resolve_piper_model()
        out = self.settings.tts_cache_dir / f"piper_{uuid.uuid4().hex}.wav"
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(model), "-f", str(out)],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=60,
        )
        try:
            audio, sr = sf.read(out, dtype="float32")
        finally:
            out.unlink(missing_ok=True)
        return audio, sr

    def _synth_kokoro(self, text: str) -> Tuple[np.ndarray, int]:
        from kokoro import KPipeline

        lang = "a" if self.language == "en" else "f"
        if self._kokoro is None or self._kokoro_lang != lang:
            self._kokoro = KPipeline(lang_code=lang)
            self._kokoro_lang = lang
        voice = self.settings.kokoro_voice_en if self.language == "en" else self.settings.kokoro_voice_fr
        parts = []
        for _, _, chunk in self._kokoro(text, voice=voice):
            parts.append(chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk))
        if not parts:
            raise RuntimeError("Kokoro returned no audio")
        return np.concatenate(parts).astype(np.float32), 24000

    async def _synth_edge(self, text: str) -> Tuple[np.ndarray, int]:
        import edge_tts
        import soundfile as sf

        voice = self.settings.edge_voice_en if self.language == "en" else self.settings.edge_voice_fr
        out = self.settings.tts_cache_dir / f"edge_{uuid.uuid4().hex}.mp3"
        await edge_tts.Communicate(text, voice).save(str(out))
        try:
            audio, sr = sf.read(out, dtype="float32")  # MP3 supported by libsndfile >= 1.1
        finally:
            out.unlink(missing_ok=True)
        return audio, sr
