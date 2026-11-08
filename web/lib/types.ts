// Shapes returned by the FastAPI backend (only the fields the dashboard reads).
export interface Student {
  id: string; name: string; age?: number | null; student_code?: string | null; created_at: string;
  voice_samples: number; notes?: string | null; group_name?: string | null; recognition_status: string;
}
export interface VoiceSample { id: number; created_at: string; duration_seconds?: number | null; has_audio: boolean; quality_weight?: number | null }
export interface StudentProfile extends Student { samples: VoiceSample[]; min_samples_required: number }

export interface Waveform { t0?: string; t0_epoch?: number | null; dt_seconds: number; peaks: number[] }
export interface Segment { start: string; end: string }
export interface SpeakerEntry {
  speaker_id?: string; student_id?: string | null; student_name?: string | null; is_identified?: boolean;
  decision_status?: string; similarity_score?: number | null; runner_up_similarity?: number | null;
  rejection_reason?: string | null; identification_status?: string; signal_type?: string;
  start_time: string; end_time: string; segments?: Segment[]; waveform?: Waveform | null;
}
export interface DetectionEvent {
  event_id: string; start_time: string; end_time: string; category?: string; labels?: string[];
  speaker_count?: number; is_overlap?: boolean; overlap_duration_ms?: number | null;
  overlap_segments?: Segment[]; metadata?: Record<string, unknown>; status?: string;
  warning_triggered?: boolean; identification_method?: string; detection_method?: string;
  decision_status?: string; processing_latency_ms?: number | null; notes?: string[]; speakers: SpeakerEntry[];
}
export interface Visualization { now: string | null; history_seconds: number; waveform: Waveform; events: DetectionEvent[]; capabilities?: Record<string, boolean> }

export interface LatencyStage { median_ms: number; p95_ms: number; count: number }
export interface MonitorStatus {
  running?: boolean; ui_state: string; live_db: number; threshold_db: number; in_cooldown: boolean; cooldown_remaining: number;
  registered_voices: number; mic_alive?: boolean; mic_silent?: boolean; speech_detected?: boolean; speech_ratio?: number;
  osd_backend?: string | null; capabilities?: Record<string, boolean>; separation_error?: string | null; error?: string | null;
  models_loaded?: boolean; dropped_windows?: number; processing_errors?: number; server_time?: number;
  latency?: Record<string, LatencyStage | { median: number; p95: number } | unknown>;
  compute?: Record<string, unknown>; current_event?: DetectionEvent | null;
}
export interface Settings {
  threshold_db: number; cooldown_seconds: number; speaker_match_threshold: number; language: string;
  speaker_cooldown_seconds: number; min_identify_seconds: number; max_simultaneous_speakers: number;
  unknown_speaker_handling: string; [k: string]: unknown;
}
export interface LogRow {
  id: number; timestamp: string; event_type: string; student_name?: string | null; dB_level: number;
  message_spoken: string; match_score?: number | null; was_spoken: boolean; category?: string | null;
  speaker_count?: number | null; overlap_duration?: number | null;
}
export interface DiagnoseResult {
  threshold: number; min_match_margin: number; duration_seconds: number; hint: string;
  results: { student_id: string; name: string; similarity: number; samples: number; ready: boolean; passes_threshold: boolean }[];
}
