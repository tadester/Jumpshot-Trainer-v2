use crate::analysis::session_audit::{create_session_audit, create_shot_summary};
use crate::calibration::estimate_calibration;
use crate::ml::infer_shot_quality;
use crate::types::{
    CalibrationInput, CalibrationProfile, DiagnosticSeverity, MetricDiagnostic, MlInference,
    SessionAudit, SessionShotSummary, ShotInput, ShotStage, StageFeedback,
};

#[derive(Debug, Clone)]
pub struct TrainerSnapshot {
    pub diagnostics: Vec<MetricDiagnostic>,
    pub inference: MlInference,
    pub stage_feedback: Vec<StageFeedback>,
    pub calibration: CalibrationProfile,
}

fn metric_diagnostic(
    metric: &str,
    actual: f32,
    ideal: f32,
    tolerance: f32,
    message_minor: &str,
    message_major: &str,
) -> MetricDiagnostic {
    let delta = actual - ideal;
    let severity = if delta.abs() <= tolerance * 0.45 {
        DiagnosticSeverity::Optimal
    } else if delta.abs() <= tolerance {
        DiagnosticSeverity::Minor
    } else {
        DiagnosticSeverity::Major
    };

    let message = match severity {
        DiagnosticSeverity::Optimal => format!("{metric} is tracking inside the target window."),
        DiagnosticSeverity::Minor => message_minor.to_string(),
        DiagnosticSeverity::Major => message_major.to_string(),
    };

    MetricDiagnostic {
        metric: metric.to_string(),
        actual,
        ideal,
        delta,
        severity,
        message,
    }
}

fn stage_score(metrics: &[&MetricDiagnostic]) -> u8 {
    if metrics.is_empty() {
        return 0;
    }

    let total = metrics.iter().fold(0.0, |sum, item| {
        let weight = match item.severity {
            DiagnosticSeverity::Optimal => 1.0,
            DiagnosticSeverity::Minor => 0.65,
            DiagnosticSeverity::Major => 0.3,
        };
        sum + weight
    });
    ((total / metrics.len() as f32) * 100.0).round() as u8
}

fn stage_color(score: u8) -> DiagnosticSeverity {
    if score >= 85 {
        DiagnosticSeverity::Optimal
    } else if score >= 65 {
        DiagnosticSeverity::Minor
    } else {
        DiagnosticSeverity::Major
    }
}

fn build_stage_feedback(
    input: &ShotInput,
    diagnostics: &[MetricDiagnostic],
) -> Vec<StageFeedback> {
    let load_metrics: Vec<&MetricDiagnostic> = diagnostics
        .iter()
        .filter(|item| item.metric == "Knee Load")
        .collect();
    let set_point_metrics: Vec<&MetricDiagnostic> = diagnostics
        .iter()
        .filter(|item| item.metric == "Elbow Flexion" || item.metric == "Forearm Verticality" || item.metric == "Elbow Flare")
        .collect();
    let release_metrics: Vec<&MetricDiagnostic> = diagnostics
        .iter()
        .filter(|item| item.metric == "Release Height Ratio" || item.metric == "Release Timing")
        .collect();

    let load_score = stage_score(&load_metrics);
    let set_score = stage_score(&set_point_metrics);
    let release_score = stage_score(&release_metrics);
    let follow_score = if input.release_at_apex_offset_ms <= 25.0 { 90 } else if input.release_at_apex_offset_ms <= 45.0 { 72 } else { 48 };

    vec![
        StageFeedback {
            stage: ShotStage::ReadyStance,
            score: 84,
            color_hint: DiagnosticSeverity::Optimal,
            coaching_note: "Balanced setup. Keep eyes, chest, and ball centered before the dip.".to_string(),
        },
        StageFeedback {
            stage: ShotStage::Load,
            score: load_score,
            color_hint: stage_color(load_score),
            coaching_note: "Watch the depth and tempo of the dip so leg drive stays efficient.".to_string(),
        },
        StageFeedback {
            stage: ShotStage::SetPoint,
            score: set_score,
            color_hint: stage_color(set_score),
            coaching_note: "Stack elbow under the ball and keep the forearm vertical through the pocket.".to_string(),
        },
        StageFeedback {
            stage: ShotStage::Release,
            score: release_score,
            color_hint: stage_color(release_score),
            coaching_note: "Release closer to the apex with a higher finish for cleaner energy transfer.".to_string(),
        },
        StageFeedback {
            stage: ShotStage::FollowThrough,
            score: follow_score,
            color_hint: stage_color(follow_score),
            coaching_note: "Freeze the wrist and hold the line to improve repeatability.".to_string(),
        },
    ]
}

pub fn default_calibration_input() -> CalibrationInput {
    CalibrationInput {
        body_height_m: 1.91,
        shoulder_width_m: 0.46,
        arm_span_ratio: 1.01,
        fingertip_reach_ratio: 1.33,
        camera_distance_m: 4.7,
        lens_tilt_deg: 2.0,
    }
}

pub fn analyze_shot(input: &ShotInput, calibration_input: &CalibrationInput) -> TrainerSnapshot {
    let diagnostics = vec![
        metric_diagnostic(
            "Elbow Flexion",
            input.elbow_flexion,
            86.0,
            10.0,
            "Elbow angle is slightly off the compact set-point window.",
            "Elbow angle is costing alignment and compactness.",
        ),
        metric_diagnostic(
            "Knee Load",
            input.knee_load,
            106.0,
            14.0,
            "The dip is a little inconsistent with an efficient load.",
            "The dip depth is breaking shot rhythm.",
        ),
        metric_diagnostic(
            "Forearm Verticality",
            input.forearm_verticality,
            90.0,
            8.0,
            "Forearm line is drifting from vertical.",
            "Forearm angle is too far from vertical at set point.",
        ),
        metric_diagnostic(
            "Elbow Flare",
            input.elbow_flare,
            3.0,
            5.0,
            "The elbow is starting to flare slightly.",
            "The elbow flare is large enough to alter the release path.",
        ),
        metric_diagnostic(
            "Release Height Ratio",
            input.release_height_ratio,
            1.31,
            0.1,
            "Release height is a bit lower than ideal.",
            "Release height is well below the pro window.",
        ),
        metric_diagnostic(
            "Release Timing",
            input.release_timing_ms,
            320.0,
            55.0,
            "Release timing is close, but not fully synced to the lift.",
            "Release timing is late enough to hurt energy transfer.",
        ),
    ];

    let inference = infer_shot_quality(input);
    let stage_feedback = build_stage_feedback(input, &diagnostics);
    let calibration = estimate_calibration(calibration_input);
    TrainerSnapshot {
        diagnostics,
        inference,
        stage_feedback,
        calibration,
    }
}

fn jitter(seed: usize, amplitude: f32) -> f32 {
    let phase = seed as f32 * 0.91;
    phase.sin() * amplitude + (phase * 0.63).cos() * amplitude * 0.4
}

pub fn build_training_session(
    base: &ShotInput,
    calibration_input: &CalibrationInput,
    shot_count: usize,
) -> (Vec<SessionShotSummary>, SessionAudit) {
    let mut shots = Vec::with_capacity(shot_count);
    let calibration = estimate_calibration(calibration_input);
    let height_scalar = (calibration.estimated_standing_reach_m / 2.5).clamp(0.9, 1.15);

    for index in 0..shot_count {
        let sample = ShotInput {
            elbow_flexion: base.elbow_flexion + jitter(index, 4.0),
            knee_load: base.knee_load + jitter(index + 7, 5.5),
            forearm_verticality: base.forearm_verticality + jitter(index + 13, 3.5),
            elbow_flare: (base.elbow_flare + jitter(index + 17, 2.0)).max(0.0),
            release_height_ratio: (base.release_height_ratio * height_scalar + jitter(index + 23, 0.04)).max(0.8),
            release_timing_ms: base.release_timing_ms + jitter(index + 31, 22.0),
            release_at_apex_offset_ms: base.release_at_apex_offset_ms + jitter(index + 37, 16.0),
            jump_height: (base.jump_height + jitter(index + 41, 0.03)).max(0.1),
        };

        let snapshot = analyze_shot(&sample, calibration_input);
        let release_time = Some(sample.release_timing_ms.max(0.0).round() as u64);
        let apex_offset = Some(sample.release_at_apex_offset_ms.round() as i64);

        shots.push(create_shot_summary(
            format!("shot-{:02}", index + 1),
            snapshot.diagnostics,
            sample.jump_height,
            release_time,
            apex_offset,
        ));
    }

    let audit = create_session_audit("desktop-workout", shots.clone());
    (shots, audit)
}
