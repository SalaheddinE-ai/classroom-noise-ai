# Privacy, consent and data protection checklist

This file is a **technical checklist for review by the responsible institution**; it is not legal advice and
does not state that the system complies with any law.

## What is stored (default configuration)

| Data | Stored? | Where | How to remove |
|---|---|---|---|
| Voice embeddings (192-D) | yes - biometric data | `data/classroom.db` | delete the sample / the student |
| Enrollment recordings | **no** (`CNAI_STORE_ENROLLMENT_AUDIO=false`) | `data/samples` if enabled | delete sample/student |
| Live classroom audio | **no**, processed in memory only | - | - |
| Event clips | no (`CNAI_RAW_AUDIO_RETENTION_SECONDS=0`); if > 0, short clips auto-deleted | `data/event_audio` | automatic expiry |
| Alerts / logs (times, names of confirmed students, similarity) | yes | `data/classroom.db` | student deletion anonymises; delete the DB to wipe all |

## Before deployment

1. **Authorisation.** Obtain the school's / responsible authority's approval for the purpose (noise management).
2. **Moroccan Law 09-08** (protection of individuals regarding personal data) and the CNDP: have the
   institution determine whether a declaration/authorisation is required, the legal basis, and the information
   to give to pupils and parents/guardians. Voice used to identify a person may be treated as sensitive or
   biometric data; this must be assessed by the institution.
3. **Consent / information** for pupils and legal guardians; offer a way to refuse or withdraw (delete profile).
4. **Retention.** Decide how long profiles and logs are kept; delete at the end of the school year.
5. **Access control.** The API has no login. Keep it on localhost or behind a reverse proxy with authentication;
   restrict who can open the dashboard and who can read `data/`.
6. **Disk encryption / backups** of the machine holding `data/`.
7. **Purpose limitation.** Identity estimates are not proof; do not use them for sanctions or evaluation of pupils.
8. **Warnings.** Consider `CNAI_TTS_ANNOUNCE_NAMES=false` so students are not named aloud.
9. **TTS.** Edge-TTS sends the warning *text* to an online service; prefer Piper/Kokoro (offline).

The project has no face recognition, camera, or covert recording features and none must be added.
