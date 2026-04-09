use crate::types::{
    BodyProportions, FrameSample, ImprovementTrend, MetricDiagnostic, NormalizedShotRecord,
    TrendPoint, Vector3,
};
use std::collections::HashMap;

fn normalize_point(point: Vector3, scale: f32) -> Vector3 {
    Vector3 {
        x: point.x / scale,
        y: point.y / scale,
        z: point.z / scale,
    }
}

pub fn normalize_frame_coordinates(frame: &FrameSample, proportions: &BodyProportions) -> FrameSample {
    let scale = proportions.standing_reach.max(1.0);

    let pose = frame
        .pose
        .iter()
        .map(|(name, point)| (*name, normalize_point(*point, scale)))
        .collect();

    let ball = frame.ball.as_ref().map(|ball| crate::types::BallTrack {
        center: normalize_point(ball.center, scale),
        radius: ball.radius / scale,
        confidence: ball.confidence,
    });

    FrameSample {
        frame_index: frame.frame_index,
        timestamp_ms: frame.timestamp_ms,
        pose,
        ball,
    }
}

pub fn to_normalized_shot_record(
    diagnostics: Vec<MetricDiagnostic>,
    captured_at: impl Into<String>,
) -> NormalizedShotRecord {
    let normalized_metrics: HashMap<String, f32> = diagnostics
        .iter()
        .map(|diagnostic| {
            let score = (1.0 - diagnostic.delta.abs() / diagnostic.ideal.abs().max(1.0)).max(0.0);
            (diagnostic.metric.clone(), score)
        })
        .collect();

    let score = if normalized_metrics.is_empty() {
        0
    } else {
        ((normalized_metrics.values().sum::<f32>() / normalized_metrics.len() as f32) * 100.0).round()
            as u8
    };

    NormalizedShotRecord {
        score,
        normalized_metrics,
        raw_diagnostics: diagnostics,
        captured_at: captured_at.into(),
    }
}

pub fn compute_improvement_trend(history: &[TrendPoint], window_days: u32) -> ImprovementTrend {
    if history.is_empty() {
        return ImprovementTrend {
            window_days,
            previous_average: 0.0,
            current_average: 0.0,
            improvement_percent: 0.0,
        };
    }

    let midpoint = (history.len() / 2).max(1);
    let previous = &history[..midpoint];
    let current = &history[midpoint..];

    let previous_average = previous.iter().map(|item| item.score).sum::<f32>() / previous.len() as f32;
    let current_average = if current.is_empty() {
        previous_average
    } else {
        current.iter().map(|item| item.score).sum::<f32>() / current.len() as f32
    };

    let improvement_percent = if previous_average == 0.0 {
        0.0
    } else {
        ((current_average - previous_average) / previous_average) * 100.0
    };

    ImprovementTrend {
        window_days,
        previous_average,
        current_average,
        improvement_percent,
    }
}
