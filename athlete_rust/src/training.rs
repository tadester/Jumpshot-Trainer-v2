use crate::ml::infer_shot_quality;
use crate::trainer::default_calibration_input;
use crate::types::{
    CalibrationInput, JanitorShotRecord, ModelReadiness, ProcessedSessionSummary, ShotInput, ShotQualityLabel,
    SupervisedModelSummary, TrainingDatasetSummary, TrainingExample,
};
use std::collections::BTreeMap;

pub fn feature_vector_from_shot_input(
    shot: &ShotInput,
    height_m: f32,
    wingspan_m: f32,
    standing_reach_m: f32,
    distance_ft: f32,
    paired_view: bool,
) -> Vec<f32> {
    vec![
        shot.elbow_flexion / 120.0,
        shot.knee_load / 135.0,
        shot.forearm_verticality / 100.0,
        shot.elbow_flare / 20.0,
        shot.release_height_ratio / 1.5,
        shot.release_timing_ms / 500.0,
        (shot.release_at_apex_offset_ms + 40.0) / 160.0,
        shot.jump_height / 0.6,
        height_m / 2.3,
        wingspan_m / 2.4,
        standing_reach_m / 3.0,
        distance_ft / 30.0,
        if paired_view { 1.0 } else { 0.0 },
    ]
}

pub fn shot_input_from_record(record: &JanitorShotRecord) -> ShotInput {
    let release_timing = record
        .release_timing_ms
        .or(record.release_time_ms_side)
        .or(record.release_time_ms_45)
        .unwrap_or(320.0)
        .clamp(180.0, 520.0);
    let distance_scale = record.distance_ft.unwrap_or(15.0) / 15.0;
    ShotInput {
        elbow_flexion: record.elbow_flexion.unwrap_or(86.0).clamp(45.0, 180.0),
        knee_load: record
            .knee_load
            .unwrap_or(106.0 + (distance_scale - 1.0) * 6.0)
            .clamp(70.0, 180.0),
        forearm_verticality: record.forearm_verticality.unwrap_or(90.0).clamp(-90.0, 100.0),
        elbow_flare: record
            .elbow_flare
            .unwrap_or(if record.paired_view_available { 4.0 } else { 6.5 })
            .clamp(0.0, 90.0),
        release_height_ratio: record
            .release_height_ratio
            .unwrap_or((record.standing_reach_m / record.height_m.max(1.0)).clamp(1.15, 1.42))
            .clamp(0.3, 1.8),
        release_timing_ms: release_timing,
        release_at_apex_offset_ms: record
            .release_at_apex_offset_ms
            .unwrap_or((release_timing - 320.0) * 0.35)
            .clamp(-200.0, 200.0),
        jump_height: record
            .jump_height
            .unwrap_or((0.32 + (distance_scale - 1.0) * 0.06).clamp(0.2, 0.55))
            .clamp(0.05, 0.8),
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
            let features = feature_vector_from_shot_input(
                &shot,
                record.height_m,
                record.wingspan_m,
                record.standing_reach_m,
                record.distance_ft.unwrap_or(15.0),
                record.paired_view_available,
            );

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

pub fn summarize_processed_sessions(records: &[JanitorShotRecord]) -> Vec<ProcessedSessionSummary> {
    let mut groups: BTreeMap<String, Vec<&JanitorShotRecord>> = BTreeMap::new();
    for record in records {
        let primary_clip = if !record.side_video.is_empty() {
            record.side_video.clone()
        } else if !record.angle45_video.is_empty() {
            record.angle45_video.clone()
        } else {
            record.clip_uid.clone()
        };
        let session_key = format!("{} :: {}", record.session_date, primary_clip);
        groups.entry(session_key).or_default().push(record);
    }

    groups
        .into_iter()
        .map(|(session_key, items)| {
            let total_shots = items.len();
            let paired_shots = items.iter().filter(|item| item.paired_view_available).count();
            let side_only_shots = items
                .iter()
                .filter(|item| !item.paired_view_available && !item.side_video.is_empty())
                .count();
            let angle_only_shots = items
                .iter()
                .filter(|item| !item.paired_view_available && !item.angle45_video.is_empty())
                .count();
            let source_dataset = items.first().map(|item| item.source_dataset.clone()).unwrap_or_default();
            let teacher_model = items.first().map(|item| item.teacher_model.clone()).unwrap_or_default();

            ProcessedSessionSummary {
                session_key,
                source_dataset,
                teacher_model,
                total_shots,
                paired_shots,
                side_only_shots,
                angle_only_shots,
            }
        })
        .collect()
}

fn linear_predict(weights: &[f32], bias: f32, features: &[f32]) -> f32 {
    let dot = weights
        .iter()
        .zip(features.iter())
        .fold(bias, |acc, (weight, feature)| acc + (weight * feature));
    dot.clamp(0.0, 1.0)
}

fn mean_absolute_error(weights: &[f32], bias: f32, examples: &[TrainingExample]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let total = examples
        .iter()
        .map(|example| (linear_predict(weights, bias, &example.features) - example.target_score).abs())
        .sum::<f32>();
    total / examples.len() as f32
}

pub fn train_supervised_score_model(examples: &[TrainingExample]) -> SupervisedModelSummary {
    let feature_count = examples.first().map(|example| example.features.len()).unwrap_or(0);
    if examples.len() < 4 || feature_count == 0 {
        return SupervisedModelSummary {
            trained: false,
            example_count: examples.len(),
            feature_count,
            training_mae: 0.0,
            validation_mae: 0.0,
            epochs: 0,
            bias: 0.0,
            weights: vec![],
        };
    }

    let split_index = ((examples.len() as f32) * 0.8).round().clamp(1.0, (examples.len() - 1) as f32) as usize;
    let (train_set, validation_set) = examples.split_at(split_index);
    let mut weights = vec![0.0; feature_count];
    let mut bias = 0.0f32;
    let learning_rate = 0.08f32;
    let l2 = 0.0005f32;
    let epochs = 240usize;

    for _ in 0..epochs {
        for example in train_set {
            let prediction = linear_predict(&weights, bias, &example.features);
            let error = prediction - example.target_score;
            bias -= learning_rate * error;
            for (weight, feature) in weights.iter_mut().zip(example.features.iter()) {
                let gradient = error * feature + l2 * *weight;
                *weight -= learning_rate * gradient;
            }
        }
    }

    let training_mae = mean_absolute_error(&weights, bias, train_set);
    let validation_mae = mean_absolute_error(&weights, bias, validation_set);

    SupervisedModelSummary {
        trained: true,
        example_count: examples.len(),
        feature_count,
        training_mae,
        validation_mae,
        epochs,
        bias,
        weights,
    }
}

pub fn predict_supervised_score(model: &SupervisedModelSummary, features: &[f32]) -> Option<f32> {
    if !model.trained || model.weights.len() != features.len() {
        return None;
    }
    Some(linear_predict(&model.weights, model.bias, features))
}
