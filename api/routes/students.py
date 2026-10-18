"""Student registration, voice profiles and management endpoints."""
from __future__ import annotations

import json
import logging
import uuid
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.deps import get_service
from api.schemas import StudentOut, StudentProfileOut, StudentUpdate, VoiceSampleOut
from core.audio_utils import load_audio_bytes
from core.quality import QualityReport, assess_sample
from database.db_session import get_db
from database.models import Student, VoiceProfile
from engine.persistence import anonymize_student_in_logs
from engine.runner import MonitorService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/students", tags=["students"])

MIN_SAMPLE_SECONDS = 1.5  # kept for backward compatibility; the live value is CNAI_MIN_SAMPLE_SECONDS


# ---------------------------------------------------------------------------- helpers
def _status(n_samples: int, min_samples: int) -> str:
    if n_samples == 0:
        return "no_samples"
    return "ready" if n_samples >= min_samples else "needs_samples"


def _to_out(student: Student, n_samples: int, min_samples: int = 1) -> StudentOut:
    return StudentOut(
        id=student.id,
        name=student.name,
        age=student.age,
        student_code=student.student_code,
        created_at=student.created_at,
        voice_samples=n_samples,
        notes=student.notes,
        group_name=student.group_name,
        recognition_status=_status(n_samples, min_samples),
    )


def _count_samples(db: Session, student_id: str) -> int:
    return db.execute(
        select(func.count(VoiceProfile.id)).where(VoiceProfile.student_id == student_id)
    ).scalar_one()


def _decode_upload(data: bytes, sample_rate: int) -> np.ndarray:
    """Decode an uploaded recording to a mono float32 array (separate function: easy to stub in tests)."""
    return load_audio_bytes(data, sample_rate).numpy().astype(np.float32)


MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # a few seconds of speech never needs more


def _embed_upload(service: MonitorService, upload: UploadFile):
    """Decode an uploaded recording and return ``(waveform, embedding, quality report)``.

    The recording is checked (duration, level, clipping, dynamics) before any embedding is made;
    an unusable sample is rejected with the reasons instead of polluting the profile.
    """
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large")
    sr = service.settings.sample_rate
    try:
        wav = _decode_upload(data, sr)
    except Exception as exc:
        raise HTTPException(422, f"Unreadable audio file: {exc}") from exc
    minimum = service.settings.min_sample_seconds
    if wav.size / sr < minimum:
        raise HTTPException(422, f"Recording too short (minimum {minimum}s)")
    quality = assess_sample(wav, sr, service.settings)
    if not quality.ok:
        raise HTTPException(422, "Voice sample rejected: " + "; ".join(quality.reasons))
    service.load_models()
    return wav, service.embed_audio(wav), quality


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _store_profile(service: MonitorService, db: Session, student_id: str, wav: np.ndarray,
                   embedding: np.ndarray, quality: Optional[QualityReport] = None) -> VoiceProfile:
    """Store the embedding (always) and the recording (only if CNAI_STORE_ENROLLMENT_AUDIO=true)."""
    path_text = None
    if service.settings.store_enrollment_audio:
        path = service.settings.samples_dir / f"{student_id}_{uuid.uuid4().hex[:8]}.wav"
        _write_wav(path, wav, service.settings.sample_rate)
        path_text = str(path)
    profile = VoiceProfile(
        student_id=student_id,
        embedding_blob=VoiceProfile.pack_embedding(embedding),
        audio_sample_path=path_text,
        duration_seconds=round(wav.size / service.settings.sample_rate, 2),
        quality_weight=round(quality.weight, 3) if quality else None,
        quality_json=json.dumps({"rms_dbfs": round(quality.rms_dbfs, 1), "clip_ratio": round(quality.clip_ratio, 4),
                                 "dynamic_range_db": round(quality.dynamic_range_db, 1)}) if quality else None,
    )
    db.add(profile)
    db.flush()  # assigns profile.id
    return profile


def _delete_sample_file(service: MonitorService, path: Optional[str]) -> None:
    """Delete a stored recording; only files inside the samples directory are ever touched."""
    if not path:
        return
    try:
        target = Path(path).resolve()
        if service.settings.samples_dir.resolve() in target.parents and target.exists():
            target.unlink()
    except OSError:
        logger.warning("Could not delete a voice sample file")


def _check_new_sample(service: MonitorService, embedding: np.ndarray, own_id: Optional[str], force: bool) -> None:
    """Refuse a sample that probably belongs to somebody else (never mix identities silently)."""
    s = service.settings
    scores = service.vector_store.score_students(embedding, s.samples_per_student, 1)
    if own_id is not None:
        own = scores.get(own_id)
        others = {k: v for k, v in scores.items() if k != own_id}
        if not force and own is not None and own < s.sample_consistency_threshold:
            raise HTTPException(
                422,
                f"This recording does not resemble the student's existing voice samples "
                f"(similarity {own:.2f} < {s.sample_consistency_threshold:.2f}). Record again, "
                f"or resend with force=true if you are sure it is the same person.",
            )
        if not force and others:
            other_id, other_score = max(others.items(), key=lambda kv: kv[1])
            if other_score >= s.duplicate_voice_threshold and other_score > (own or -1.0):
                name = service.vector_store.name_of(other_id) or "another student"
                raise HTTPException(409, f"This voice is very similar to {name}'s profile "
                                         f"(similarity {other_score:.2f}); sample not added. Use force=true to override.")
    elif scores and not force:
        other_id, other_score = max(scores.items(), key=lambda kv: kv[1])
        if other_score >= s.duplicate_voice_threshold:
            name = service.vector_store.name_of(other_id) or "another student"
            raise HTTPException(409, f"This voice is very similar to the registered student '{name}' "
                                     f"(similarity {other_score:.2f}). Add a sample to that student "
                                     f"instead, or resend with force=true.")


def _get_student(db: Session, student_id: str) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(404, "Student not found")
    return student


# ----------------------------------------------------------------------------- routes
@router.get("", response_model=List[StudentOut])
def list_students(
    db: Session = Depends(get_db), service: MonitorService = Depends(get_service)
) -> List[StudentOut]:
    """List registered students with their number of voice samples and recognition status."""
    counts = dict(
        db.execute(select(VoiceProfile.student_id, func.count(VoiceProfile.id)).group_by(VoiceProfile.student_id)).all()
    )
    minimum = service.settings.min_voice_samples
    students = db.execute(select(Student).order_by(Student.name)).scalars().all()
    return [_to_out(s, counts.get(s.id, 0), minimum) for s in students]


@router.post("/register", response_model=StudentOut, status_code=201)
def register_student(
    name: str = Form(...),
    age: Optional[int] = Form(None),
    student_code: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    group_name: Optional[str] = Form(None),
    allow_duplicate_name: bool = Form(False),
    force: bool = Form(False),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> StudentOut:
    """Create a student from a short voice recording and store its ECAPA embedding.

    The original form (name, age, student_code, audio) is unchanged; the other fields are optional.
    """
    name = name.strip()
    code = (student_code or "").strip() or None
    if not name:
        raise HTTPException(422, "Name is required")
    if code and db.execute(select(Student).where(Student.student_code == code)).scalar_one_or_none():
        raise HTTPException(409, f"Student code '{code}' already exists")
    if not allow_duplicate_name and db.execute(
        select(Student).where(func.lower(Student.name) == name.lower())
    ).first():
        raise HTTPException(
            409,
            f"A student named '{name}' already exists. Add a voice sample to that profile, or "
            f"register again with allow_duplicate_name=true (e.g. two students with the same name).",
        )

    wav, embedding, quality = _embed_upload(service, audio)
    _check_new_sample(service, embedding, None, force)
    student = Student(name=name, age=age, student_code=code,
                      notes=(notes or "").strip() or None, group_name=(group_name or "").strip() or None)
    db.add(student)
    db.flush()
    profile = _store_profile(service, db, student.id, wav, embedding, quality)
    db.commit()
    service.vector_store.add(student.id, embedding, profile.id, quality.weight)
    service.vector_store.set_name(student.id, student.name)
    return _to_out(student, 1, service.settings.min_voice_samples)


@router.post("/diagnose")
def diagnose_voice(
    audio: UploadFile = File(...),
    service: MonitorService = Depends(get_service),
) -> dict:
    """Compare a fresh recording with every registered profile and show the raw similarities.

    Nothing is stored. Use it to see *why* a registered student is not recognised: the cosine
    similarity of a genuine voice is often well below 0.75 (other microphone, distance, room).
    """
    wav, embedding, _ = _embed_upload(service, audio)
    s = service.settings
    scores = service.vector_store.score_students(embedding, s.samples_per_student, 1)
    rows = sorted(
        (
            {
                "student_id": sid,
                "name": service.vector_store.name_of(sid) or sid,
                "similarity": round(score, 3),
                "samples": service.vector_store.sample_count(sid),
                "ready": service.vector_store.sample_count(sid) >= s.min_voice_samples,
                "passes_threshold": score >= s.speaker_match_threshold,
            }
            for sid, score in scores.items()
        ),
        key=lambda r: r["similarity"], reverse=True,
    )
    best = rows[0] if rows else None
    runner = rows[1]["similarity"] if len(rows) > 1 else None
    if best is None:
        hint = "No student with a voice sample is registered."
    elif not best["ready"]:
        hint = f"{best['name']} has fewer than {s.min_voice_samples} voice samples: add more."
    elif best["passes_threshold"] and (runner is None or best["similarity"] - runner >= s.min_match_margin):
        hint = "This voice would be identified."
    elif best["passes_threshold"]:
        hint = (f"Rejected as ambiguous: the runner-up is within {s.min_match_margin} of the best score "
                f"(lower CNAI_MIN_MATCH_MARGIN or add samples).")
    else:
        hint = (f"Below the threshold {s.speaker_match_threshold:.2f}. Try a threshold of about "
                f"{max(0.2, best['similarity'] - 0.08):.2f}, and re-enroll 2-3 samples with the same microphone "
                f"and room as the monitoring.")
    return {
        "threshold": s.speaker_match_threshold,
        "min_match_margin": s.min_match_margin,
        "duration_seconds": round(wav.size / s.sample_rate, 2),
        "results": rows,
        "hint": hint,
    }


@router.get("/{student_id}", response_model=StudentProfileOut)
def get_student(
    student_id: str, db: Session = Depends(get_db), service: MonitorService = Depends(get_service)
) -> StudentProfileOut:
    """One student's profile with the list of registered voice samples."""
    student = _get_student(db, student_id)
    profiles = db.execute(
        select(VoiceProfile).where(VoiceProfile.student_id == student_id).order_by(VoiceProfile.id)
    ).scalars().all()
    minimum = service.settings.min_voice_samples
    base = _to_out(student, len(profiles), minimum)
    return StudentProfileOut(
        **base.model_dump(),
        samples=[
            VoiceSampleOut(
                id=p.id, created_at=p.created_at, duration_seconds=p.duration_seconds,
                quality_weight=p.quality_weight,
                has_audio=bool(p.audio_sample_path and Path(p.audio_sample_path).exists()),
            )
            for p in profiles
        ],
        min_samples_required=minimum,
    )


@router.patch("/{student_id}", response_model=StudentOut)
def update_student(
    student_id: str,
    body: StudentUpdate,
    db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> StudentOut:
    """Edit name, age, student code, notes or group (voice samples are untouched)."""
    student = _get_student(db, student_id)
    values = body.model_dump(exclude_unset=True)
    if "name" in values:
        new_name = (values["name"] or "").strip()
        if not new_name:
            raise HTTPException(422, "Name cannot be empty")
        student.name = new_name
    if "student_code" in values:
        code = (values["student_code"] or "").strip() or None
        if code and db.execute(
            select(Student).where(Student.student_code == code, Student.id != student_id)
        ).scalar_one_or_none():
            raise HTTPException(409, f"Student code '{code}' already exists")
        student.student_code = code
    for field in ("age", "notes", "group_name"):
        if field in values:
            value = values[field]
            setattr(student, field, (value.strip() or None) if isinstance(value, str) else value)
    db.commit()
    service.vector_store.set_name(student.id, student.name)
    return _to_out(student, _count_samples(db, student_id), service.settings.min_voice_samples)


@router.post("/{student_id}/voice", response_model=StudentOut)
def add_voice_sample(
    student_id: str,
    audio: UploadFile = File(...),
    force: bool = Form(False),
    db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> StudentOut:
    """Add another voice sample to improve recognition robustness."""
    student = _get_student(db, student_id)
    wav, embedding, quality = _embed_upload(service, audio)
    _check_new_sample(service, embedding, student.id, force)
    profile = _store_profile(service, db, student.id, wav, embedding, quality)
    db.commit()
    service.vector_store.add(student.id, embedding, profile.id, quality.weight)
    service.vector_store.set_name(student.id, student.name)
    return _to_out(student, _count_samples(db, student.id), service.settings.min_voice_samples)


@router.get("/{student_id}/voice/{sample_id}/audio")
def get_voice_sample_audio(
    student_id: str, sample_id: int, db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> FileResponse:
    """Play back a stored voice sample (local file, never sent to a third party)."""
    profile = db.get(VoiceProfile, sample_id)
    if profile is None or profile.student_id != student_id:
        raise HTTPException(404, "Voice sample not found")
    path = Path(profile.audio_sample_path or "")
    if not path.exists() or service.settings.samples_dir.resolve() not in path.resolve().parents:
        raise HTTPException(404, "Audio file no longer available")
    return FileResponse(str(path), media_type="audio/wav")


@router.delete("/{student_id}/voice/{sample_id}", status_code=204)
def delete_voice_sample(
    student_id: str, sample_id: int, db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> None:
    """Delete one voice sample (its recording and its embedding)."""
    profile = db.get(VoiceProfile, sample_id)
    if profile is None or profile.student_id != student_id:
        raise HTTPException(404, "Voice sample not found")
    path = profile.audio_sample_path
    db.delete(profile)
    db.commit()
    _delete_sample_file(service, path)
    service.vector_store.remove_sample(student_id, sample_id)


@router.delete("/{student_id}", status_code=204)
def delete_student(
    student_id: str,
    db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> None:
    """Delete a student with every voice sample, embedding and recording, and anonymise past logs."""
    student = _get_student(db, student_id)
    paths = [p.audio_sample_path for p in student.voice_profiles]
    db.delete(student)
    db.commit()
    for path in paths:
        _delete_sample_file(service, path)
    anonymize_student_in_logs(student_id)
    service.vector_store.remove_student(student_id)
    if service.engine is not None:
        service.engine.forget_student(student_id)
