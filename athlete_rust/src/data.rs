use crate::types::AngleRange;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EliteShotBaseline {
    pub metadata: BaselineMetadata,
    pub stages: BaselineStages,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineMetadata {
    pub name: String,
    pub capture_fps: u32,
    pub sample_size: u32,
    pub source_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineStages {
    pub ready_stance: ReadyStanceBaseline,
    pub load: LoadBaseline,
    pub set_point: SetPointBaseline,
    pub release: ReleaseBaseline,
    pub follow_through: FollowThroughBaseline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadyStanceBaseline {
    pub knee_flexion: AngleRange,
    pub hip_hinge: AngleRange,
    pub elbow_flexion: AngleRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadBaseline {
    pub knee_flexion: AngleRange,
    pub hip_hinge: AngleRange,
    pub ball_to_sternum_distance_ratio: AngleRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SetPointBaseline {
    pub elbow_flexion: AngleRange,
    pub forearm_verticality: AngleRange,
    pub elbow_flare: AngleRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseBaseline {
    pub release_height_ratio: AngleRange,
    pub release_timing_from_lift_ms: AngleRange,
    pub release_at_apex_offset_ms: AngleRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FollowThroughBaseline {
    pub wrist_freeze_ms: AngleRange,
    pub arm_arc_angle: AngleRange,
    pub landing_drift_ratio: AngleRange,
}

pub fn load_elite_shot_baseline() -> Result<EliteShotBaseline, serde_json::Error> {
    serde_json::from_str(include_str!("../data/elite-shot-baselines.json"))
}
