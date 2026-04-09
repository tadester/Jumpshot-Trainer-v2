use crate::data::load_elite_shot_baseline;
use crate::types::{MlInference, ShotInput, ShotQualityLabel};

#[derive(Debug, Clone)]
struct ShotPrototype {
    name: &'static str,
    features: ShotInput,
}

fn seed_prototypes() -> Vec<ShotPrototype> {
    vec![
        ShotPrototype {
            name: "compact-elite",
            features: ShotInput {
                elbow_flexion: 86.0,
                knee_load: 106.0,
                forearm_verticality: 91.0,
                elbow_flare: 3.0,
                release_height_ratio: 1.31,
                release_timing_ms: 320.0,
                release_at_apex_offset_ms: 8.0,
                jump_height: 0.42,
            },
        },
        ShotPrototype {
            name: "smooth-starter",
            features: ShotInput {
                elbow_flexion: 89.0,
                knee_load: 111.0,
                forearm_verticality: 87.0,
                elbow_flare: 6.0,
                release_height_ratio: 1.27,
                release_timing_ms: 345.0,
                release_at_apex_offset_ms: 18.0,
                jump_height: 0.36,
            },
        },
        ShotPrototype {
            name: "late-release",
            features: ShotInput {
                elbow_flexion: 97.0,
                knee_load: 118.0,
                forearm_verticality: 80.0,
                elbow_flare: 11.0,
                release_height_ratio: 1.18,
                release_timing_ms: 410.0,
                release_at_apex_offset_ms: 74.0,
                jump_height: 0.31,
            },
        },
        ShotPrototype {
            name: "wide-and-flat",
            features: ShotInput {
                elbow_flexion: 105.0,
                knee_load: 126.0,
                forearm_verticality: 71.0,
                elbow_flare: 16.0,
                release_height_ratio: 1.09,
                release_timing_ms: 470.0,
                release_at_apex_offset_ms: 102.0,
                jump_height: 0.25,
            },
        },
    ]
}

fn feature_distance(a: &ShotInput, b: &ShotInput) -> f32 {
    let elbow = ((a.elbow_flexion - b.elbow_flexion) / 20.0).powi(2);
    let knee = ((a.knee_load - b.knee_load) / 25.0).powi(2);
    let verticality = ((a.forearm_verticality - b.forearm_verticality) / 20.0).powi(2);
    let flare = ((a.elbow_flare - b.elbow_flare) / 12.0).powi(2);
    let height = ((a.release_height_ratio - b.release_height_ratio) / 0.25).powi(2);
    let timing = ((a.release_timing_ms - b.release_timing_ms) / 110.0).powi(2);
    let apex = ((a.release_at_apex_offset_ms - b.release_at_apex_offset_ms) / 60.0).powi(2);
    let jump = ((a.jump_height - b.jump_height) / 0.18).powi(2);
    (elbow + knee + verticality + flare + height + timing + apex + jump).sqrt()
}

fn baseline_score(input: &ShotInput) -> f32 {
    let Ok(baseline) = load_elite_shot_baseline() else {
        return 0.0;
    };

    let set = &baseline.stages.set_point;
    let load = &baseline.stages.load;
    let release = &baseline.stages.release;

    let scores = [
        1.0 - ((input.elbow_flexion - set.elbow_flexion.ideal).abs() / 20.0),
        1.0 - ((input.knee_load - load.knee_flexion.ideal).abs() / 25.0),
        1.0 - ((input.forearm_verticality - set.forearm_verticality.ideal).abs() / 18.0),
        1.0 - ((input.elbow_flare - set.elbow_flare.ideal).abs() / 10.0),
        1.0 - ((input.release_height_ratio - release.release_height_ratio.ideal).abs() / 0.22),
        1.0 - ((input.release_timing_ms - release.release_timing_from_lift_ms.ideal).abs() / 120.0),
        1.0 - ((input.release_at_apex_offset_ms - release.release_at_apex_offset_ms.ideal).abs() / 70.0),
        1.0 - ((input.jump_height - 0.4).abs() / 0.2),
    ];

    scores.into_iter().map(|score| score.clamp(0.0, 1.0)).sum::<f32>() / 8.0
}

fn feedback_from_input(input: &ShotInput) -> Vec<String> {
    let mut feedback = Vec::new();

    if input.elbow_flare > 8.0 {
        feedback.push("Tuck the shooting elbow closer to your hip-to-rim line.".to_string());
    }
    if input.forearm_verticality < 84.0 {
        feedback.push("Stack wrist over elbow earlier so the forearm stays more vertical.".to_string());
    }
    if input.release_timing_ms > 370.0 {
        feedback.push("The release is late. Try to let the ball go closer to the jump apex.".to_string());
    }
    if input.release_at_apex_offset_ms > 35.0 {
        feedback.push("You are releasing on the way down. Speed up the transfer from lift to snap.".to_string());
    }
    if input.knee_load > 118.0 {
        feedback.push("The dip is a little deep. Reduce wasted motion in the load phase.".to_string());
    }
    if input.release_height_ratio < 1.22 {
        feedback.push("Finish higher through the shot pocket to raise the release point.".to_string());
    }

    if feedback.is_empty() {
        feedback.push("Mechanics look compact. Focus on repeatability and shot rhythm.".to_string());
    }

    feedback
}

pub fn infer_shot_quality(input: &ShotInput) -> MlInference {
    let prototypes = seed_prototypes();
    let nearest = prototypes
        .iter()
        .map(|prototype| (prototype, feature_distance(input, &prototype.features)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("seeded prototypes are non-empty");

    let baseline = baseline_score(input);
    let neighbor_confidence = (1.0 / (1.0 + nearest.1)).clamp(0.0, 1.0);
    let confidence = ((baseline * 0.55) + (neighbor_confidence * 0.45)).clamp(0.0, 1.0);
    let score = (confidence * 100.0).round() as u8;

    let label = if score >= 87 {
        ShotQualityLabel::Elite
    } else if score >= 72 {
        ShotQualityLabel::Strong
    } else if score >= 55 {
        ShotQualityLabel::Developing
    } else {
        ShotQualityLabel::Raw
    };

    MlInference {
        label,
        confidence,
        score,
        nearest_neighbor: nearest.0.name.to_string(),
        feedback: feedback_from_input(input),
    }
}
