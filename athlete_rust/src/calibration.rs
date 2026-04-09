use crate::types::{CalibrationInput, CalibrationProfile};

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

pub fn estimate_calibration(input: &CalibrationInput) -> CalibrationProfile {
    let estimated_wingspan_m = input.body_height_m * input.arm_span_ratio;
    let standing_reach_from_height = input.body_height_m * input.fingertip_reach_ratio;
    let shoulder_bonus = input.shoulder_width_m * 0.18;
    let estimated_standing_reach_m = standing_reach_from_height + shoulder_bonus;

    let geometric_angle =
        ((estimated_standing_reach_m - input.camera_height_m) / input.camera_distance_m.max(0.5))
            .atan()
            .to_degrees();
    let estimated_camera_angle_deg = geometric_angle + input.lens_tilt_deg;

    let shoulder_conf = 1.0 - ((input.shoulder_width_m - 0.44).abs() / 0.18);
    let span_conf = 1.0 - ((input.arm_span_ratio - 1.0).abs() / 0.2);
    let distance_conf = 1.0 - ((input.camera_distance_m - 4.5).abs() / 3.0);
    let confidence = clamp((shoulder_conf + span_conf + distance_conf) / 3.0, 0.35, 0.98);

    CalibrationProfile {
        estimated_wingspan_m,
        estimated_standing_reach_m,
        estimated_camera_angle_deg,
        body_height_m: input.body_height_m,
        shoulder_width_m: input.shoulder_width_m,
        confidence,
    }
}
