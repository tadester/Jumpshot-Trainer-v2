use crate::types::{
    FrameSample, LandmarkName, ShooterSide, ShotStage, ShotStageEvent, Vector3,
};

#[derive(Debug, Clone, Copy)]
pub struct StateMachineThresholds {
    pub ready_elbow_min: f32,
    pub ready_elbow_max: f32,
    pub load_knee_max: f32,
    pub set_point_velocity_threshold: f32,
    pub release_wrist_height_margin: f32,
    pub follow_through_freeze_ms: u64,
}

impl Default for StateMachineThresholds {
    fn default() -> Self {
        Self {
            ready_elbow_min: 75.0,
            ready_elbow_max: 110.0,
            load_knee_max: 125.0,
            set_point_velocity_threshold: 0.015,
            release_wrist_height_margin: 0.02,
            follow_through_freeze_ms: 180,
        }
    }
}

fn subtract(a: Vector3, b: Vector3) -> Vector3 {
    Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn magnitude(v: Vector3) -> f32 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

fn dot(a: Vector3, b: Vector3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn angle_at_joint(a: Vector3, joint: Vector3, c: Vector3) -> f32 {
    let ab = subtract(a, joint);
    let cb = subtract(c, joint);
    let denominator = magnitude(ab) * magnitude(cb);
    if denominator == 0.0 {
        return 180.0;
    }
    let cosine = (dot(ab, cb) / denominator).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

fn joint_keys(side: ShooterSide) -> (LandmarkName, LandmarkName, LandmarkName, LandmarkName, LandmarkName, LandmarkName) {
    match side {
        ShooterSide::Left => (
            LandmarkName::LeftShoulder,
            LandmarkName::LeftElbow,
            LandmarkName::LeftWrist,
            LandmarkName::LeftHip,
            LandmarkName::LeftKnee,
            LandmarkName::LeftAnkle,
        ),
        ShooterSide::Right => (
            LandmarkName::RightShoulder,
            LandmarkName::RightElbow,
            LandmarkName::RightWrist,
            LandmarkName::RightHip,
            LandmarkName::RightKnee,
            LandmarkName::RightAnkle,
        ),
    }
}

pub fn detect_shot_stage_events(
    frames: &[FrameSample],
    shooter_side: ShooterSide,
    thresholds: Option<StateMachineThresholds>,
) -> Vec<ShotStageEvent> {
    let thresholds = thresholds.unwrap_or_default();
    let (shoulder_key, elbow_key, wrist_key, hip_key, knee_key, ankle_key) = joint_keys(shooter_side);
    let mut events = Vec::new();

    let mut current_stage = ShotStage::Idle;
    let mut load_timestamp_ms: Option<u64> = None;
    let mut release_timestamp_ms: Option<u64> = None;
    let mut follow_through_stable_since: Option<u64> = None;

    for window in frames.windows(2) {
        let prev = &window[0];
        let frame = &window[1];

        let shoulder = *frame.pose.get(&shoulder_key).unwrap();
        let elbow = *frame.pose.get(&elbow_key).unwrap();
        let wrist = *frame.pose.get(&wrist_key).unwrap();
        let hip = *frame.pose.get(&hip_key).unwrap();
        let knee = *frame.pose.get(&knee_key).unwrap();
        let ankle = *frame.pose.get(&ankle_key).unwrap();
        let prev_wrist = *prev.pose.get(&wrist_key).unwrap();

        let elbow_angle = angle_at_joint(shoulder, elbow, wrist);
        let knee_angle = angle_at_joint(hip, knee, ankle);
        let wrist_velocity = magnitude(subtract(wrist, prev_wrist));

        if matches!(current_stage, ShotStage::Idle)
            && elbow_angle >= thresholds.ready_elbow_min
            && elbow_angle <= thresholds.ready_elbow_max
        {
            current_stage = ShotStage::ReadyStance;
            events.push(ShotStageEvent {
                stage: current_stage,
                frame_index: frame.frame_index,
                timestamp_ms: frame.timestamp_ms,
            });
            continue;
        }

        if matches!(current_stage, ShotStage::ReadyStance) && knee_angle <= thresholds.load_knee_max {
            current_stage = ShotStage::Load;
            load_timestamp_ms = Some(frame.timestamp_ms);
            events.push(ShotStageEvent {
                stage: current_stage,
                frame_index: frame.frame_index,
                timestamp_ms: frame.timestamp_ms,
            });
            continue;
        }

        if matches!(current_stage, ShotStage::Load) && wrist_velocity <= thresholds.set_point_velocity_threshold {
            current_stage = ShotStage::SetPoint;
            events.push(ShotStageEvent {
                stage: current_stage,
                frame_index: frame.frame_index,
                timestamp_ms: frame.timestamp_ms,
            });
            continue;
        }

        if matches!(current_stage, ShotStage::SetPoint) {
            if let Some(ball) = &frame.ball {
                if wrist.y <= ball.center.y + thresholds.release_wrist_height_margin {
                    current_stage = ShotStage::Release;
                    release_timestamp_ms = Some(frame.timestamp_ms);
                    events.push(ShotStageEvent {
                        stage: current_stage,
                        frame_index: frame.frame_index,
                        timestamp_ms: frame.timestamp_ms,
                    });
                    continue;
                }
            }
        }

        if matches!(current_stage, ShotStage::Release) {
            if wrist_velocity <= thresholds.set_point_velocity_threshold {
                if follow_through_stable_since.is_none() {
                    follow_through_stable_since = Some(frame.timestamp_ms);
                }
                if frame.timestamp_ms - follow_through_stable_since.unwrap()
                    >= thresholds.follow_through_freeze_ms
                {
                    current_stage = ShotStage::FollowThrough;
                    events.push(ShotStageEvent {
                        stage: current_stage,
                        frame_index: frame.frame_index,
                        timestamp_ms: frame.timestamp_ms,
                    });
                    continue;
                }
            } else {
                follow_through_stable_since = None;
            }
        }

        if matches!(current_stage, ShotStage::FollowThrough)
            && release_timestamp_ms.is_some()
            && load_timestamp_ms.is_some()
            && frame.timestamp_ms - release_timestamp_ms.unwrap() > 350
        {
            events.push(ShotStageEvent {
                stage: ShotStage::Complete,
                frame_index: frame.frame_index,
                timestamp_ms: frame.timestamp_ms,
            });
            break;
        }
    }

    events
}
