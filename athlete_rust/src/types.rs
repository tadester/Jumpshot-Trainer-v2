use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub enum LandmarkName {
    Nose,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

pub type PoseLandmarks = HashMap<LandmarkName, Vector3>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BallTrack {
    pub center: Vector3,
    pub radius: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameSample {
    pub frame_index: u32,
    pub timestamp_ms: u64,
    pub pose: PoseLandmarks,
    pub ball: Option<BallTrack>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShooterSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShotStage {
    Idle,
    ReadyStance,
    Load,
    SetPoint,
    Release,
    FollowThrough,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BodyProportions {
    pub torso_length: f32,
    pub femur_length: f32,
    pub tibia_length: f32,
    pub humerus_length: f32,
    pub forearm_length: f32,
    pub standing_reach: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AngleRange {
    pub min: f32,
    pub max: f32,
    pub ideal: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Optimal,
    Minor,
    Major,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricDiagnostic {
    pub metric: String,
    pub actual: f32,
    pub ideal: f32,
    pub delta: f32,
    pub severity: DiagnosticSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShotStageEvent {
    pub stage: ShotStage,
    pub frame_index: u32,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShotPhaseWindow {
    pub start_ms: u64,
    pub end_ms: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionShotSummary {
    pub shot_id: String,
    pub release_time_ms: Option<u64>,
    pub jump_height: f32,
    pub release_at_apex_offset_ms: Option<i64>,
    pub consistency_score: u8,
    pub diagnostics: Vec<MetricDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionAudit {
    pub session_id: String,
    pub attempt_count: usize,
    pub average_consistency_score: u8,
    pub top_issues: Vec<String>,
    pub shots: Vec<SessionShotSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrendPoint {
    pub date: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImprovementTrend {
    pub window_days: u32,
    pub previous_average: f32,
    pub current_average: f32,
    pub improvement_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedShotRecord {
    pub score: u8,
    pub normalized_metrics: HashMap<String, f32>,
    pub raw_diagnostics: Vec<MetricDiagnostic>,
    pub captured_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShotInput {
    pub elbow_flexion: f32,
    pub knee_load: f32,
    pub forearm_verticality: f32,
    pub elbow_flare: f32,
    pub release_height_ratio: f32,
    pub release_timing_ms: f32,
    pub release_at_apex_offset_ms: f32,
    pub jump_height: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShotQualityLabel {
    Elite,
    Strong,
    Developing,
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MlInference {
    pub label: ShotQualityLabel,
    pub confidence: f32,
    pub score: u8,
    pub nearest_neighbor: String,
    pub feedback: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationInput {
    pub body_height_m: f32,
    pub shoulder_width_m: f32,
    pub arm_span_ratio: f32,
    pub fingertip_reach_ratio: f32,
    pub camera_distance_m: f32,
    pub lens_tilt_deg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProfile {
    pub estimated_wingspan_m: f32,
    pub estimated_standing_reach_m: f32,
    pub estimated_camera_angle_deg: f32,
    pub body_height_m: f32,
    pub shoulder_width_m: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageFeedback {
    pub stage: ShotStage,
    pub score: u8,
    pub color_hint: DiagnosticSeverity,
    pub coaching_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JanitorShotRecord {
    pub athlete_id: String,
    pub shot_id: String,
    pub session_date: String,
    pub fps: u32,
    pub side_video: String,
    pub angle45_video: String,
    pub set_point_frame_side: Option<i64>,
    pub release_frame_side: Option<i64>,
    pub set_point_frame_45: Option<i64>,
    pub release_frame_45: Option<i64>,
    pub make: Option<bool>,
    pub shot_type: String,
    pub distance_ft: Option<f32>,
    pub notes: String,
    pub handedness: String,
    pub height_m: f32,
    pub wingspan_m: f32,
    pub standing_reach_m: f32,
    pub release_time_ms_side: Option<f32>,
    pub release_time_ms_45: Option<f32>,
    pub paired_view_available: bool,
    pub has_manual_stage_tags: bool,
    pub source_dataset: String,
    pub source_tier: String,
    pub annotation_quality: String,
    pub teacher_model: String,
    pub clip_uid: String,
    pub shot_start_frame_side: Option<i64>,
    pub shot_end_frame_side: Option<i64>,
    pub shot_start_frame_45: Option<i64>,
    pub shot_end_frame_45: Option<i64>,
    pub elbow_flexion: Option<f32>,
    pub knee_load: Option<f32>,
    pub forearm_verticality: Option<f32>,
    pub elbow_flare: Option<f32>,
    pub release_height_ratio: Option<f32>,
    pub release_timing_ms: Option<f32>,
    pub release_at_apex_offset_ms: Option<f32>,
    pub jump_height: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingExample {
    pub shot_id: String,
    pub features: Vec<f32>,
    pub target_score: f32,
    pub target_label: ShotQualityLabel,
    pub has_paired_view: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingDatasetSummary {
    pub example_count: usize,
    pub paired_view_examples: usize,
    pub label_balance: Vec<(ShotQualityLabel, usize)>,
    pub average_target_score: f32,
    pub feature_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelReadiness {
    pub is_ready: bool,
    pub score: u8,
    pub checklist: Vec<String>,
    pub risks: Vec<String>,
    pub recommended_next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessedSessionSummary {
    pub session_key: String,
    pub source_dataset: String,
    pub teacher_model: String,
    pub total_shots: usize,
    pub paired_shots: usize,
    pub side_only_shots: usize,
    pub angle_only_shots: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SupervisedModelSummary {
    pub trained: bool,
    pub example_count: usize,
    pub feature_count: usize,
    pub training_mae: f32,
    pub validation_mae: f32,
    pub epochs: usize,
    pub bias: f32,
    pub weights: Vec<f32>,
}
