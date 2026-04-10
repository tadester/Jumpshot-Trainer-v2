use crate::types::JanitorShotRecord;
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

fn string_value(column: &Column, index: usize) -> String {
    match column.get(index) {
        Ok(AnyValue::String(value)) => value.to_string(),
        Ok(AnyValue::StringOwned(value)) => value.to_string(),
        Ok(other) => other.to_string(),
        Err(_) => String::new(),
    }
}

fn u32_value(column: &Column, index: usize, default: u32) -> u32 {
    match column.get(index) {
        Ok(AnyValue::UInt32(value)) => value,
        Ok(AnyValue::UInt64(value)) => value as u32,
        Ok(AnyValue::Int64(value)) => value.max(0) as u32,
        Ok(AnyValue::Int32(value)) => value.max(0) as u32,
        Ok(AnyValue::Null) | Err(_) => default,
        Ok(other) => other.to_string().parse().unwrap_or(default),
    }
}

fn bool_value(column: &Column, index: usize, default: bool) -> bool {
    match column.get(index) {
        Ok(AnyValue::Boolean(value)) => value,
        Ok(AnyValue::Null) | Err(_) => default,
        Ok(other) => other.to_string().parse().unwrap_or(default),
    }
}

fn opt_i64_value(column: &Column, index: usize) -> Option<i64> {
    match column.get(index) {
        Ok(AnyValue::Int64(value)) => Some(value),
        Ok(AnyValue::Int32(value)) => Some(value as i64),
        Ok(AnyValue::UInt64(value)) => Some(value as i64),
        Ok(AnyValue::UInt32(value)) => Some(value as i64),
        Ok(AnyValue::Float64(value)) => Some(value.round() as i64),
        Ok(AnyValue::Float32(value)) => Some(value.round() as i64),
        Ok(AnyValue::Null) | Err(_) => None,
        Ok(other) => other.to_string().parse().ok(),
    }
}

fn opt_f32_value(column: &Column, index: usize) -> Option<f32> {
    match column.get(index) {
        Ok(AnyValue::Float32(value)) => Some(value),
        Ok(AnyValue::Float64(value)) => Some(value as f32),
        Ok(AnyValue::Int64(value)) => Some(value as f32),
        Ok(AnyValue::Int32(value)) => Some(value as f32),
        Ok(AnyValue::UInt64(value)) => Some(value as f32),
        Ok(AnyValue::UInt32(value)) => Some(value as f32),
        Ok(AnyValue::Null) | Err(_) => None,
        Ok(other) => other.to_string().parse().ok(),
    }
}

pub fn load_janitor_shot_records(path: impl AsRef<Path>) -> PolarsResult<Vec<JanitorShotRecord>> {
    let file = File::open(path)?;
    let frame = ParquetReader::new(file).finish()?;

    let athlete_id = frame.column("athlete_id")?;
    let shot_id = frame.column("shot_id")?;
    let session_date = frame.column("session_date")?;
    let fps = frame.column("fps")?;
    let side_video = frame.column("side_video")?;
    let angle45_video = frame.column("angle45_video")?;
    let set_point_frame_side = frame.column("set_point_frame_side")?;
    let release_frame_side = frame.column("release_frame_side")?;
    let set_point_frame_45 = frame.column("set_point_frame_45")?;
    let release_frame_45 = frame.column("release_frame_45")?;
    let make = frame.column("make")?;
    let shot_type = frame.column("shot_type")?;
    let distance_ft = frame.column("distance_ft")?;
    let notes = frame.column("notes")?;
    let handedness = frame.column("handedness")?;
    let height_m = frame.column("height_m")?;
    let wingspan_m = frame.column("wingspan_m")?;
    let standing_reach_m = frame.column("standing_reach_m")?;
    let release_time_ms_side = frame.column("release_time_ms_side")?;
    let release_time_ms_45 = frame.column("release_time_ms_45")?;
    let paired_view_available = frame.column("paired_view_available")?;
    let source_dataset = frame.column("source_dataset")?;
    let source_tier = frame.column("source_tier")?;
    let annotation_quality = frame.column("annotation_quality")?;
    let teacher_model = frame.column("teacher_model")?;
    let clip_uid = frame.column("clip_uid")?;
    let shot_start_frame_side = frame.column("shot_start_frame_side")?;
    let shot_end_frame_side = frame.column("shot_end_frame_side")?;
    let shot_start_frame_45 = frame.column("shot_start_frame_45")?;
    let shot_end_frame_45 = frame.column("shot_end_frame_45")?;
    let elbow_flexion = frame.column("elbow_flexion").ok();
    let knee_load = frame.column("knee_load").ok();
    let forearm_verticality = frame.column("forearm_verticality").ok();
    let elbow_flare = frame.column("elbow_flare").ok();
    let release_height_ratio = frame.column("release_height_ratio").ok();
    let release_timing_ms = frame.column("release_timing_ms").ok();
    let release_at_apex_offset_ms = frame.column("release_at_apex_offset_ms").ok();
    let jump_height = frame.column("jump_height").ok();

    let mut records = Vec::with_capacity(frame.height());
    for index in 0..frame.height() {
        records.push(JanitorShotRecord {
            athlete_id: string_value(athlete_id, index),
            shot_id: string_value(shot_id, index),
            session_date: string_value(session_date, index),
            fps: u32_value(fps, index, 60),
            side_video: string_value(side_video, index),
            angle45_video: string_value(angle45_video, index),
            set_point_frame_side: opt_i64_value(set_point_frame_side, index),
            release_frame_side: opt_i64_value(release_frame_side, index),
            set_point_frame_45: opt_i64_value(set_point_frame_45, index),
            release_frame_45: opt_i64_value(release_frame_45, index),
            make: match make.get(index) {
                Ok(AnyValue::Boolean(value)) => Some(value),
                _ => None,
            },
            shot_type: string_value(shot_type, index),
            distance_ft: opt_f32_value(distance_ft, index),
            notes: string_value(notes, index),
            handedness: string_value(handedness, index),
            height_m: opt_f32_value(height_m, index).unwrap_or(1.88),
            wingspan_m: opt_f32_value(wingspan_m, index).unwrap_or(1.95),
            standing_reach_m: opt_f32_value(standing_reach_m, index).unwrap_or(2.46),
            release_time_ms_side: opt_f32_value(release_time_ms_side, index),
            release_time_ms_45: opt_f32_value(release_time_ms_45, index),
            paired_view_available: bool_value(paired_view_available, index, false),
            source_dataset: string_value(source_dataset, index),
            source_tier: string_value(source_tier, index),
            annotation_quality: string_value(annotation_quality, index),
            teacher_model: string_value(teacher_model, index),
            clip_uid: string_value(clip_uid, index),
            shot_start_frame_side: opt_i64_value(shot_start_frame_side, index),
            shot_end_frame_side: opt_i64_value(shot_end_frame_side, index),
            shot_start_frame_45: opt_i64_value(shot_start_frame_45, index),
            shot_end_frame_45: opt_i64_value(shot_end_frame_45, index),
            elbow_flexion: elbow_flexion.and_then(|column| opt_f32_value(column, index)),
            knee_load: knee_load.and_then(|column| opt_f32_value(column, index)),
            forearm_verticality: forearm_verticality.and_then(|column| opt_f32_value(column, index)),
            elbow_flare: elbow_flare.and_then(|column| opt_f32_value(column, index)),
            release_height_ratio: release_height_ratio.and_then(|column| opt_f32_value(column, index)),
            release_timing_ms: release_timing_ms.and_then(|column| opt_f32_value(column, index)),
            release_at_apex_offset_ms: release_at_apex_offset_ms.and_then(|column| opt_f32_value(column, index)),
            jump_height: jump_height.and_then(|column| opt_f32_value(column, index)),
        });
    }
    Ok(records)
}
