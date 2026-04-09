use crate::ml::infer_shot_quality;
use crate::trainer::default_calibration_input;
use crate::types::{
    CalibrationInput, JanitorShotRecord, ModelReadiness, ShotInput, ShotQualityLabel, TrainingDatasetSummary,
    TrainingExample,
};

pub fn shot_input_from_record(record: &JanitorShotRecord) -> ShotInput {
    let release_timing = record.release_time_ms_side.unwrap_or(320.0).clamp(180.0, 520.0);
    let distance_scale = record.distance_ft.unwrap_or(15.0) / 15.0;
    ShotInput {
        elbow_flexion: 86.0,
        knee_load: 106.0 + (distance_scale - 1.0) * 6.0,
        forearm_verticality: 90.0,
        elbow_flare: if record.paired_view_available { 4.0 } else { 6.5 },
        release_height_ratio: (record.standing_reach_m / record.height_m.max(1.0)).clamp(1.15, 1.42),
        release_timing_ms: release_timing,
        release_at_apex_offset_ms: (release_timing - 320.0) * 0.35,
        jump_height: (0.32 + (distance_scale - 1.0) * 0.06).clamp(0.2, 0.55),
    }
}

pub fn calibration_input_from_record(record: &JanitorShotRecord) -> CalibrationInput {
    let mut calibration = default_calibration_input();
    calibration.body_height_m = record.height_m;
    calibration.arm_span_ratio = record.wingspan_m / record.height_m.max(0.5);
    calibration.fingertip_reach_ratio = record.standing_reach_m / record.height_m.max(0.5);
    calibration
}

pub fn build_training_examples(records: &[JanitorShotRecord]) -> Vec<TrainingExample> {
    records
        .iter()
        .map(|record| {
            let shot = shot_input_from_record(record);
            let inference = infer_shot_quality(&shot);
            let features = vec![
                shot.elbow_flexion / 120.0,
                shot.knee_load / 135.0,
                shot.forearm_verticality / 100.0,
                shot.elbow_flare / 20.0,
                shot.release_height_ratio / 1.5,
                shot.release_timing_ms / 500.0,
                (shot.release_at_apex_offset_ms + 40.0) / 160.0,
                shot.jump_height / 0.6,
                record.height_m / 2.3,
                record.wingspan_m / 2.4,
                record.standing_reach_m / 3.0,
                record.distance_ft.unwrap_or(15.0) / 30.0,
                if record.paired_view_available { 1.0 } else { 0.0 },
            ];

            TrainingExample {
                shot_id: record.shot_id.clone(),
                target_score: inference.score as f32 / 100.0,
                target_label: inference.label,
                has_paired_view: record.paired_view_available,
                features,
            }
        })
        .collect()
}

pub fn summarize_training_dataset(examples: &[TrainingExample]) -> TrainingDatasetSummary {
    let mut elite = 0;
    let mut strong = 0;
    let mut developing = 0;
    let mut raw = 0;
    let mut paired_view_examples = 0;
    let mut total_score = 0.0;

    for example in examples {
        total_score += example.target_score;
        if example.has_paired_view {
            paired_view_examples += 1;
        }
        match example.target_label {
            ShotQualityLabel::Elite => elite += 1,
            ShotQualityLabel::Strong => strong += 1,
            ShotQualityLabel::Developing => developing += 1,
            ShotQualityLabel::Raw => raw += 1,
        }
    }

    TrainingDatasetSummary {
        example_count: examples.len(),
        paired_view_examples,
        label_balance: vec![
            (ShotQualityLabel::Elite, elite),
            (ShotQualityLabel::Strong, strong),
            (ShotQualityLabel::Developing, developing),
            (ShotQualityLabel::Raw, raw),
        ],
        average_target_score: if examples.is_empty() {
            0.0
        } else {
            total_score / examples.len() as f32
        },
        feature_count: examples.first().map(|item| item.features.len()).unwrap_or(0),
    }
}

pub fn evaluate_model_readiness(summary: &TrainingDatasetSummary) -> ModelReadiness {
    let enough_examples = summary.example_count >= 20;
    let enough_paired = summary.paired_view_examples >= 10;
    let enough_features = summary.feature_count >= 10;
    let label_variety = summary.label_balance.iter().filter(|(_, count)| *count > 0).count() >= 2;

    let mut checklist = Vec::new();
    let mut risks = Vec::new();

    checklist.push(format!("{} examples available for Rust-side training tensors.", summary.example_count));
    checklist.push(format!("{} paired-view examples are available.", summary.paired_view_examples));
    checklist.push(format!("{} normalized features are defined per example.", summary.feature_count));

    if !enough_examples {
        risks.push("Collect at least 20 tagged shots before fitting the first supervised model.".to_string());
    }
    if !enough_paired {
        risks.push("Paired side + 45-degree views are still sparse for robust elbow-flare targets.".to_string());
    }
    if !label_variety {
        risks.push("Current examples collapse into too few quality buckets, which weakens classifier training.".to_string());
    }
    if !enough_features {
        risks.push("Feature vector is too small for a stable first-pass biomechanics learner.".to_string());
    }

    let mut score = 35;
    if enough_examples {
        score += 25;
    }
    if enough_paired {
        score += 20;
    }
    if enough_features {
        score += 10;
    }
    if label_variety {
        score += 10;
    }

    let is_ready = enough_examples && enough_paired && enough_features;
    let recommended_next_step = if is_ready {
        "Dataset is ready for the first candle training run with train/validation splits.".to_string()
    } else {
        "Keep collecting tagged calibration shots, then export Parquet again before training.".to_string()
    };

    ModelReadiness {
        is_ready,
        score: score as u8,
        checklist,
        risks,
        recommended_next_step,
    }
}
