use crate::data::load_elite_shot_baseline;
use crate::types::{
    BodyProportions, DiagnosticSeverity, FrameSample, LandmarkName, MetricDiagnostic, ShooterSide,
    Vector3,
};

#[derive(Debug, Clone, Copy)]
pub struct DiagnosticSnapshot {
    pub elbow_flexion: f32,
    pub knee_load: f32,
    pub forearm_verticality: f32,
    pub elbow_flare: f32,
    pub release_height_ratio: f32,
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

fn keys(side: ShooterSide) -> (LandmarkName, LandmarkName, LandmarkName, LandmarkName, LandmarkName, LandmarkName) {
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

fn severity_for_delta(delta: f32, range_width: f32) -> DiagnosticSeverity {
    let absolute_delta = delta.abs();
    if absolute_delta <= range_width * 0.2 {
        DiagnosticSeverity::Optimal
    } else if absolute_delta <= range_width * 0.5 {
        DiagnosticSeverity::Minor
    } else {
        DiagnosticSeverity::Major
    }
}

fn create_diagnostic(metric: &str, actual: f32, min: f32, max: f32, ideal: f32) -> MetricDiagnostic {
    let delta = actual - ideal;
    let severity = severity_for_delta(delta, max - min);

    let message = match severity {
        DiagnosticSeverity::Optimal => format!("{metric} is within the target band."),
        DiagnosticSeverity::Minor => {
            format!("{metric} is drifting from the elite baseline by {:.1} degrees/units.", delta.abs())
        }
        DiagnosticSeverity::Major => {
            format!("{metric} is meaningfully outside the elite baseline and should be corrected first.")
        }
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

pub fn compute_diagnostic_snapshot(
    frame: &FrameSample,
    shooter_side: ShooterSide,
    proportions: &BodyProportions,
) -> DiagnosticSnapshot {
    let (shoulder_key, elbow_key, wrist_key, hip_key, knee_key, ankle_key) = keys(shooter_side);
    let shoulder = *frame.pose.get(&shoulder_key).unwrap();
    let elbow = *frame.pose.get(&elbow_key).unwrap();
    let wrist = *frame.pose.get(&wrist_key).unwrap();
    let hip = *frame.pose.get(&hip_key).unwrap();
    let knee = *frame.pose.get(&knee_key).unwrap();
    let ankle = *frame.pose.get(&ankle_key).unwrap();

    let forearm = subtract(wrist, elbow);
    let torso = subtract(shoulder, hip);

    DiagnosticSnapshot {
        elbow_flexion: angle_at_joint(shoulder, elbow, wrist),
        knee_load: angle_at_joint(hip, knee, ankle),
        forearm_verticality: 90.0 - forearm.x.atan2(-forearm.y).to_degrees().abs(),
        elbow_flare: (elbow.x - shoulder.x)
            .atan2((elbow.y - shoulder.y).abs())
            .to_degrees()
            .abs(),
        release_height_ratio: (wrist.y - ankle.y).abs()
            / proportions.standing_reach.max(magnitude(torso)),
    }
}

pub fn compare_to_elite_baseline(
    snapshot: &DiagnosticSnapshot,
    stage: &str,
) -> Result<Vec<MetricDiagnostic>, serde_json::Error> {
    let baseline = load_elite_shot_baseline()?;

    let diagnostics = match stage {
        "load" => vec![create_diagnostic(
            "kneeLoad",
            snapshot.knee_load,
            baseline.stages.load.knee_flexion.min,
            baseline.stages.load.knee_flexion.max,
            baseline.stages.load.knee_flexion.ideal,
        )],
        "set_point" => vec![
            create_diagnostic(
                "elbowFlexion",
                snapshot.elbow_flexion,
                baseline.stages.set_point.elbow_flexion.min,
                baseline.stages.set_point.elbow_flexion.max,
                baseline.stages.set_point.elbow_flexion.ideal,
            ),
            create_diagnostic(
                "forearmVerticality",
                snapshot.forearm_verticality,
                baseline.stages.set_point.forearm_verticality.min,
                baseline.stages.set_point.forearm_verticality.max,
                baseline.stages.set_point.forearm_verticality.ideal,
            ),
            create_diagnostic(
                "elbowFlare",
                snapshot.elbow_flare,
                baseline.stages.set_point.elbow_flare.min,
                baseline.stages.set_point.elbow_flare.max,
                baseline.stages.set_point.elbow_flare.ideal,
            ),
        ],
        "release" => vec![create_diagnostic(
            "releaseHeightRatio",
            snapshot.release_height_ratio,
            baseline.stages.release.release_height_ratio.min,
            baseline.stages.release.release_height_ratio.max,
            baseline.stages.release.release_height_ratio.ideal,
        )],
        _ => Vec::new(),
    };

    Ok(diagnostics)
}
