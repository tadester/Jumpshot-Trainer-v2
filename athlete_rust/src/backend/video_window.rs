use crate::analysis::state_machine::detect_shot_stage_events;
use crate::types::{FrameSample, ShooterSide, ShotPhaseWindow, ShotStage};

pub fn detect_shot_window(frames: &[FrameSample], shooter_side: ShooterSide) -> Option<ShotPhaseWindow> {
    let events = detect_shot_stage_events(frames, shooter_side, None);

    let start = events
        .iter()
        .find(|event| matches!(event.stage, ShotStage::ReadyStance | ShotStage::Load))?;
    let end = events
        .iter()
        .rev()
        .find(|event| matches!(event.stage, ShotStage::FollowThrough | ShotStage::Complete))?;

    Some(ShotPhaseWindow {
        start_ms: start.timestamp_ms.saturating_sub(750),
        end_ms: end.timestamp_ms + 1_000,
        reason: "Trimmed around detected shot mechanics to retain the relevant three-second rep window."
            .to_string(),
    })
}
