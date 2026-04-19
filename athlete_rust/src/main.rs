use biomech_ai::ingest::load_janitor_shot_records;
use biomech_ai::trainer::{analyze_shot, TrainerSnapshot};
use biomech_ai::training::{
    build_training_examples, calibration_input_from_record, feature_vector_from_shot_input,
    predict_supervised_score, shot_input_from_record, summarize_training_dataset, train_supervised_score_model,
};
use biomech_ai::types::{
    CalibrationInput, DiagnosticSeverity, JanitorShotRecord, ShotInput, ShotStage, StageFeedback,
    SupervisedModelSummary, TrainingDatasetSummary,
};
use eframe::egui::{self, Align2, Color32, FontId, RichText, Stroke, TextureHandle, Vec2};
use image::io::Reader as ImageReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::thread;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("JumpShot Trainer"),
        ..Default::default()
    };

    eframe::run_native(
        "JumpShot Trainer",
        options,
        Box::new(|cc| Ok(Box::new(JumpshotTrainerApp::new(cc)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClipView {
    Side,
    Angle45,
}

impl ClipView {
    fn as_cli(self) -> &'static str {
        match self {
            Self::Side => "side",
            Self::Angle45 => "angle45",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Side => "Side View",
            Self::Angle45 => "Front Quarter",
        }
    }

}

#[derive(Clone)]
struct LoadedCorpus {
    supervised_model: SupervisedModelSummary,
    dataset_summary: TrainingDatasetSummary,
}

#[derive(Clone)]
struct AnalysisRunResult {
    clip_path: PathBuf,
    manifest_path: PathBuf,
    session_json: PathBuf,
    shot_records: Vec<JanitorShotRecord>,
    corpus: LoadedCorpus,
    selected_view: ClipView,
}

enum WorkerEvent {
    Status(String),
    Completed(AnalysisRunResult),
    Failed(String),
}

struct JumpshotTrainerApp {
    project_root: PathBuf,
    selected_clip_path: String,
    selected_view: ClipView,
    athlete_name: String,
    athlete_handedness: String,
    athlete_height_m: String,
    athlete_wingspan_m: String,
    athlete_standing_reach_m: String,
    analysis_receiver: Option<Receiver<WorkerEvent>>,
    is_processing: bool,
    status_message: String,
    error_message: Option<String>,
    loaded_corpus: LoadedCorpus,
    analysis_result: Option<AnalysisRunResult>,
    selected_shot_index: usize,
    show_engine_details: bool,
    preview_image_path: Option<PathBuf>,
    release_image_path: Option<PathBuf>,
    preview_texture: Option<TextureHandle>,
    release_texture: Option<TextureHandle>,
    shot_thumbnail_paths: Vec<Option<PathBuf>>,
    shot_thumbnail_textures: Vec<Option<TextureHandle>>,
    texture_revision: u64,
}

impl JumpshotTrainerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        apply_theme(&cc.egui_ctx);

        let athlete_rust_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let project_root = athlete_rust_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| athlete_rust_dir.clone());

        let mut athlete_name = "Guest Athlete".to_string();
        let mut athlete_handedness = "right".to_string();
        let mut athlete_height_m = "1.88".to_string();
        let mut athlete_wingspan_m = "1.95".to_string();
        let mut athlete_standing_reach_m = "2.40".to_string();

        let generated_profile_path = project_root.join("datasets/uploads/app_athlete.json");
        if let Ok(content) = std::fs::read_to_string(&generated_profile_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(n) = json.get("name").and_then(|v| v.as_str()) { athlete_name = n.to_string(); }
                if let Some(h) = json.get("handedness").and_then(|v| v.as_str()) { athlete_handedness = h.to_string(); }
                if let Some(h) = json.get("height_m").and_then(|v| v.as_f64()) { athlete_height_m = format!("{:.2}", h); }
                if let Some(w) = json.get("wingspan_m").and_then(|v| v.as_f64()) { athlete_wingspan_m = format!("{:.2}", w); }
                if let Some(r) = json.get("standing_reach_m").and_then(|v| v.as_f64()) { athlete_standing_reach_m = format!("{:.2}", r); }
            }
        }

        Self {
            project_root: project_root.clone(),
            selected_clip_path: String::new(),
            selected_view: ClipView::Side,
            athlete_name,
            athlete_handedness,
            athlete_height_m,
            athlete_wingspan_m,
            athlete_standing_reach_m,
            analysis_receiver: None,
            is_processing: false,
            status_message: "Drop a shooting clip into the window or paste a path to start.".to_string(),
            error_message: None,
            loaded_corpus: load_corpus_state(&project_root),
            analysis_result: None,
            selected_shot_index: 0,
            show_engine_details: false,
            preview_image_path: None,
            release_image_path: None,
            preview_texture: None,
            release_texture: None,
            shot_thumbnail_paths: Vec::new(),
            shot_thumbnail_textures: Vec::new(),
            texture_revision: 0,
        }
    }

    fn poll_background_worker(&mut self, ctx: &egui::Context) {
        let mut should_clear = false;
        if let Some(receiver) = self.analysis_receiver.take() {
            while let Ok(event) = receiver.try_recv() {
                match event {
                    WorkerEvent::Status(message) => {
                        self.status_message = message;
                    }
                    WorkerEvent::Completed(result) => {
                        self.loaded_corpus = result.corpus.clone();
                        self.analysis_result = Some(result.clone());
                        self.selected_shot_index = 0;
                        self.is_processing = false;
                        self.error_message = None;

                        self.rebuild_shot_thumbnails(ctx, &result);
                        if let Err(error) = self.refresh_release_image(ctx) {
                            eprintln!("Release snapshot extraction failed: {error}");
                            self.release_image_path = None;
                            self.release_texture = None;
                        }
                        self.log_all_model_scores();
                        self.log_selected_shot_score();

                        self.status_message = "Analysis complete. Review the shot constraints and image data above.".to_string();
                        should_clear = true;
                    }
                    WorkerEvent::Failed(message) => {
                        self.is_processing = false;
                        self.error_message = Some(message.clone());
                        self.status_message = "Analysis failed. Check the message below and try a cleaner clip or a different view."
                            .to_string();
                        should_clear = true;
                    }
                }
            }
            if !should_clear {
                self.analysis_receiver = Some(receiver);
            }
        }
    }

    fn start_analysis(&mut self, ctx: &egui::Context) {
        let clip_path = PathBuf::from(self.selected_clip_path.trim());
        if self.selected_clip_path.trim().is_empty() {
            self.error_message = Some("Choose a video first. Drag one into the app or paste the full file path.".to_string());
            return;
        }
        if !clip_path.exists() {
            self.error_message = Some("That video path does not exist. Check the file path and try again.".to_string());
            return;
        }

        self.error_message = None;
        self.is_processing = true;
        self.status_message = "Starting analysis pipeline...".to_string();
        self.analysis_result = None;
        self.selected_shot_index = 0;
        self.set_release_image(ctx, None);
        self.shot_thumbnail_paths.clear();
        self.shot_thumbnail_textures.clear();

        let project_root = self.project_root.clone();
        let preview_out = project_root.join("datasets/uploads/app_preview.jpg");
        match extract_video_frame(&project_root, &clip_path, 0, &preview_out) {
            Ok(()) => {
                self.set_preview_image(ctx, Some(preview_out));
            }
            Err(error) => {
                eprintln!("Preview snapshot extraction failed: {error}");
                self.set_preview_image(ctx, None);
            }
        }

        
        let generated_profile_path = project_root.join("datasets/uploads/app_athlete.json");
        std::fs::create_dir_all(generated_profile_path.parent().unwrap()).ok();
        let height_val: f32 = self.athlete_height_m.parse().unwrap_or(1.88);
        let wingspan_val: f32 = self.athlete_wingspan_m.parse().unwrap_or(1.95);
        let reach_val: f32 = self.athlete_standing_reach_m.parse().unwrap_or(2.40);
        
        let json_content = format!(
            r#"{{ "athlete_id": "app_user", "name": "{}", "handedness": "{}", "height_m": {}, "wingspan_m": {}, "standing_reach_m": {} }}"#,
            self.athlete_name.replace("\"", ""),
            if self.athlete_handedness == "right" { "right" } else { "left" },
            height_val,
            wingspan_val,
            reach_val
        );
        std::fs::write(&generated_profile_path, json_content).ok();
        
        let athlete_profile = generated_profile_path;
        let selected_view = self.selected_view;
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            let outcome = run_analysis_pipeline(
                &project_root,
                &clip_path,
                selected_view,
                &athlete_profile,
                &sender,
            );
            if let Err(error) = outcome {
                let _ = sender.send(WorkerEvent::Failed(error));
            }
        });

        self.analysis_receiver = Some(receiver);
    }

    fn selected_record(&self) -> Option<&JanitorShotRecord> {
        let result = self.analysis_result.as_ref()?;
        result.shot_records.get(self.selected_shot_index)
    }

    fn supervised_score_for_record(&self, record: &JanitorShotRecord) -> Option<f32> {
        predict_supervised_score(
            &self.loaded_corpus.supervised_model,
            &feature_vector_from_shot_input(
                &shot_input_from_record(record),
                record.height_m,
                record.wingspan_m,
                record.standing_reach_m,
                record.distance_ft.unwrap_or(15.0),
                record.paired_view_available,
            ),
        )
    }

    fn selected_shot_view(&self) -> Option<(ShotInput, CalibrationInput, TrainerSnapshot)> {
        let record = self.selected_record()?;
        let input = shot_input_from_record(record);
        let calibration = calibration_input_from_record(record);
        let snapshot = analyze_shot(&input, &calibration);
        Some((input, calibration, snapshot))
    }

    fn refresh_release_image(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let Some(result) = &self.analysis_result else {
            self.set_release_image(ctx, None);
            return Ok(());
        };
        let Some(record) = result.shot_records.get(self.selected_shot_index) else {
            self.set_release_image(ctx, None);
            return Ok(());
        };

        let release_frame = match result.selected_view {
            ClipView::Side => record.release_frame_side,
            ClipView::Angle45 => record.release_frame_45,
        };

        let Some(release_frame) = release_frame else {
            self.set_release_image(ctx, None);
            return Ok(());
        };

        let release_out = self.project_root.join("datasets/uploads/app_release.jpg");
        extract_video_frame(&self.project_root, &result.clip_path, release_frame.max(0), &release_out)?;
        self.set_release_image(ctx, Some(release_out));
        Ok(())
    }

    fn rebuild_shot_thumbnails(&mut self, ctx: &egui::Context, result: &AnalysisRunResult) {
        self.shot_thumbnail_paths.clear();
        self.shot_thumbnail_textures.clear();

        for (index, record) in result.shot_records.iter().enumerate() {
            let thumb_path = self
                .project_root
                .join(format!("datasets/uploads/app_rep_thumb_{index}.jpg"));

            let frame_index = preferred_thumbnail_frame(record, result.selected_view);
            let texture = frame_index
                .and_then(|frame| {
                    extract_video_frame(&self.project_root, &result.clip_path, frame.max(0), &thumb_path).ok()?;
                    load_texture_from_path(ctx, &thumb_path, &self.next_texture_name("rep-thumb")).ok()
                });

            self.shot_thumbnail_paths.push(texture.as_ref().map(|_| thumb_path));
            self.shot_thumbnail_textures.push(texture);
        }
    }

    fn next_texture_name(&mut self, prefix: &str) -> String {
        self.texture_revision += 1;
        format!("{prefix}-{}", self.texture_revision)
    }

    fn set_preview_image(&mut self, ctx: &egui::Context, path: Option<PathBuf>) {
        self.preview_texture = path
            .as_ref()
            .and_then(|path| load_texture_from_path(ctx, path, &self.next_texture_name("preview")).ok());
        self.preview_image_path = path;
    }

    fn set_release_image(&mut self, ctx: &egui::Context, path: Option<PathBuf>) {
        self.release_texture = path
            .as_ref()
            .and_then(|path| load_texture_from_path(ctx, path, &self.next_texture_name("release")).ok());
        self.release_image_path = path;
    }

    fn log_all_model_scores(&self) {
        let Some(result) = &self.analysis_result else {
            return;
        };

        for (index, record) in result.shot_records.iter().enumerate() {
            if let Some(score) = self.supervised_score_for_record(record) {
                println!("Model score [{}]: {:.3}", index + 1, score);
            }
        }
    }

    fn log_selected_shot_score(&self) {
        let Some(result) = &self.analysis_result else {
            return;
        };
        let Some(record) = result.shot_records.get(self.selected_shot_index) else {
            return;
        };
        if let Some(score) = self.supervised_score_for_record(record) {
            println!(
                "Selected shot {} from {}",
                self.selected_shot_index + 1,
                display_file_name(&result.clip_path)
            );
            println!("Model score: {:.3}", score);
        }
    }
}

impl eframe::App for JumpshotTrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_dropped_file(ctx, &mut self.selected_clip_path);
        self.poll_background_worker(ctx);
        paint_background(ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(26, 22)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    hero_header(ui);
                    ui.add_space(18.0);

                    shell_card(ui, |ui| {
                        upload_panel(ui, self);
                    });

                    if self.is_processing {
                        ui.add_space(14.0);
                        shell_card(ui, |ui| {
                            processing_panel(ui, &self.status_message, self.preview_texture.as_ref());
                        });
                    }

                    if let Some(error) = &self.error_message {
                        ui.add_space(14.0);
                        error_card(ui, error);
                    }

                    let analysis_result = self.analysis_result.clone();
                    if let Some(result) = analysis_result.as_ref() {
                        ui.add_space(18.0);
                        if let Some((input, calibration, snapshot)) = self.selected_shot_view() {
                            shell_card(ui, |ui| {
                                analysis_overview(
                                    ui,
                                    result,
                                    self.selected_shot_index,
                                    &snapshot,
                                    self.preview_texture.as_ref(),
                                );
                            });

                            ui.add_space(14.0);
                            ui.columns(2, |columns| {
                                let (left_cols, right_cols) = columns.split_at_mut(1);
                                let left = &mut left_cols[0];
                                let right = &mut right_cols[0];

                                shell_card(left, |ui| {
                                    shot_selector(ui, self, result);
                                });

                                shell_card(right, |ui| {
                                    adjustments_panel(ui, &snapshot);
                                });
                            });

                            ui.add_space(14.0);
                            ui.columns(2, |columns| {
                                let (left_cols, right_cols) = columns.split_at_mut(1);
                                let left = &mut left_cols[0];
                                let right = &mut right_cols[0];

                                shell_card(left, |ui| {
                                    metric_summary(ui, &input, &snapshot);
                                });
                                shell_card(right, |ui| {
                                    stage_panel(ui, &snapshot.stage_feedback, self.release_texture.as_ref());
                                });
                            });

                            ui.add_space(14.0);
                            shell_card(ui, |ui| {
                                overlay_panel(ui, &input, &snapshot.stage_feedback, &calibration);
                            });
                        }
                    }

                    ui.add_space(16.0);
                    engine_footer(ui, self);
                });
            });
    }
}

fn extract_video_frame(
    project_root: &Path,
    clip_path: &Path,
    frame_index: i64,
    output_path: &Path,
) -> Result<(), String> {
    let janitor_python = project_root.join("janitor_python/.venv/bin/python");
    if !janitor_python.exists() {
        return Err(format!(
            "Frame extraction python not found at {}",
            janitor_python.display()
        ));
    }

    let frame_script = r#"
import pathlib
import sys

import cv2

clip_path = sys.argv[1]
frame_index = max(int(sys.argv[2]), 0)
output_path = pathlib.Path(sys.argv[3])

capture = cv2.VideoCapture(clip_path)
if not capture.isOpened():
    raise SystemExit(f"Could not open video: {clip_path}")

if frame_index:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ok, frame = capture.read()
capture.release()

if not ok or frame is None:
    raise SystemExit(f"Could not read frame {frame_index} from {clip_path}")

output_path.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(output_path), frame):
    raise SystemExit(f"Could not write image to {output_path}")
"#;

    run_command(
        Command::new(&janitor_python)
            .arg("-c")
            .arg(frame_script)
            .arg(clip_path)
            .arg(frame_index.to_string())
            .arg(output_path),
    )?;
    Ok(())
}

fn run_analysis_pipeline(
    project_root: &Path,
    clip_path: &Path,
    selected_view: ClipView,
    athlete_profile: &Path,
    sender: &mpsc::Sender<WorkerEvent>,
) -> Result<(), String> {
    let janitor = project_root.join("janitor_python/.venv/bin/jumpshot-janitor");
    if !janitor.exists() {
        return Err(format!("Janitor CLI not found at {}", janitor.display()));
    }
    if !athlete_profile.exists() {
        return Err(format!("Athlete profile not found at {}", athlete_profile.display()));
    }

    let _ = sender.send(WorkerEvent::Status("Copying clip into the workspace...".to_string()));
    let intake_output = run_command(
        Command::new(&janitor)
            .current_dir(project_root)
            .arg("intake-video")
            .arg("--project-root")
            .arg(project_root)
            .arg("--clip")
            .arg(clip_path)
            .arg("--view")
            .arg(selected_view.as_cli()),
    )?;
    let manifest_path = parse_labeled_path(&intake_output, "Wrote intake manifest: ")
        .ok_or_else(|| format!("Could not find manifest path in janitor output:\n{intake_output}"))?;

    let _ = sender.send(WorkerEvent::Status("Running pose, ball, and shot analysis...".to_string()));
    let strong_output = run_command(
        Command::new(&janitor)
            .current_dir(project_root)
            .arg("strong-process")
            .arg("--project-root")
            .arg(project_root)
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--athlete-profile")
            .arg(athlete_profile)
            .arg("--source-dataset")
            .arg("uploaded_session")
            .arg("--teacher-model")
            .arg("mediapipe_yolov8_teacher")
            .arg("--frame-stride")
            .arg("1")
            .arg("--yolo-weights")
            .arg(project_root.join("yolov8n.pt"))
            .arg("--pose-weights")
            .arg(project_root.join("yolov8n-pose.pt"))
            .arg("--mediapipe-model")
            .arg(project_root.join("datasets/models/mediapipe/pose_landmarker_lite.task")),
    )?;

    let shots_parquet = parse_labeled_path(&strong_output, "Wrote shots_parquet: ")
        .ok_or_else(|| format!("Could not find shot parquet path in janitor output:\n{strong_output}"))?;
    let session_json = parse_labeled_path(&strong_output, "Wrote session_json: ")
        .ok_or_else(|| format!("Could not find session json path in janitor output:\n{strong_output}"))?;

    let _ = sender.send(WorkerEvent::Status("Refreshing the shared model corpus...".to_string()));
    let _ = run_command(
        Command::new(&janitor)
            .current_dir(project_root)
            .arg("build-corpus")
            .arg("--project-root")
            .arg(project_root),
    )?;

    let shot_records = load_janitor_shot_records(&shots_parquet)
        .map_err(|error| format!("Failed to load processed shots: {error}"))?;
    if shot_records.is_empty() {
        return Err("The clip finished processing, but no usable shots were detected. Try a clearer angle, tighter framing, or a steadier clip.".to_string());
    }

    let corpus = load_corpus_state(project_root);
    let result = AnalysisRunResult {
        clip_path: clip_path.to_path_buf(),
        manifest_path,
        session_json,
        shot_records,
        corpus,
        selected_view,
    };
    let _ = sender.send(WorkerEvent::Completed(result));
    Ok(())
}

fn load_corpus_state(project_root: &Path) -> LoadedCorpus {
    let empty_summary = TrainingDatasetSummary {
        example_count: 0,
        paired_view_examples: 0,
        label_balance: vec![],
        average_target_score: 0.0,
        feature_count: 0,
    };
    let corpus_path = project_root.join("datasets/shared/processed/training_corpus.parquet");
    let records = load_janitor_shot_records(&corpus_path).unwrap_or_default();
    if records.is_empty() {
        return LoadedCorpus {
            supervised_model: train_supervised_score_model(&[]),
            dataset_summary: empty_summary,
        };
    }

    let examples = build_training_examples(&records);
    LoadedCorpus {
        supervised_model: train_supervised_score_model(&examples),
        dataset_summary: summarize_training_dataset(&examples),
    }
}

fn run_command(command: &mut Command) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("Failed to start command: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn parse_labeled_path(output: &str, prefix: &str) -> Option<PathBuf> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(prefix).map(|rest| PathBuf::from(rest.trim())))
}

fn load_texture_from_path(
    ctx: &egui::Context,
    path: &Path,
    texture_name: &str,
) -> Result<TextureHandle, String> {
    let image = ImageReader::open(path)
        .map_err(|error| format!("Failed to open image {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("Failed to guess image format for {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("Failed to decode image {}: {error}", path.display()))?
        .to_rgba8();

    let size = [image.width() as usize, image.height() as usize];
    let pixels = image.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Ok(ctx.load_texture(
        texture_name.to_string(),
        color_image,
        egui::TextureOptions::LINEAR,
    ))
}

fn preferred_thumbnail_frame(record: &JanitorShotRecord, view: ClipView) -> Option<i64> {
    match view {
        ClipView::Side => record
            .release_frame_side
            .or(record.set_point_frame_side)
            .or(record.shot_start_frame_side),
        ClipView::Angle45 => record
            .release_frame_45
            .or(record.set_point_frame_45)
            .or(record.shot_start_frame_45),
    }
}

fn choose_video_file() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("osascript")
            .arg("-e")
            .arg(r#"POSIX path of (choose file with prompt "Choose a jump-shot video")"#)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return (!path.is_empty()).then(|| PathBuf::from(path));
    }

    #[cfg(target_os = "windows")]
    {
        let script = r#"
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = "Video Files|*.mp4;*.mov;*.avi;*.mkv;*.m4v|All Files|*.*"
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  Write-Output $dialog.FileName
}
"#;
        let output = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg(script)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return (!path.is_empty()).then(|| PathBuf::from(path));
    }

    #[cfg(target_os = "linux")]
    {
        for (command, args) in [
            ("zenity", vec!["--file-selection", "--title=Choose a jump-shot video"]),
            ("kdialog", vec!["--getopenfilename", ".", "*.mp4 *.mov *.avi *.mkv *.m4v"]),
        ] {
            let output = Command::new(command).args(args).output().ok()?;
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
        None
    }
}

fn apply_dropped_file(ctx: &egui::Context, selected_clip_path: &mut String) {
    let dropped = ctx.input(|input| input.raw.dropped_files.clone());
    if let Some(file) = dropped.into_iter().find(|file| file.path.is_some()) {
        if let Some(path) = file.path {
            *selected_clip_path = path.display().to_string();
        }
    }
}

fn hero_header(ui: &mut egui::Ui) {
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("JumpShot Trainer").size(40.0).strong().color(Color32::from_rgb(122, 20, 32)));
        ui.add_space(6.0);
        ui.label(
            RichText::new("Drop in a clip, set up your profile, and get instant mechanical feedback.")
                .size(18.0)
                .color(Color32::from_rgb(111, 77, 82)),
        );
    });
}

fn upload_panel(ui: &mut egui::Ui, app: &mut JumpshotTrainerApp) {
    egui::Frame::new()
        .fill(Color32::from_rgb(255, 244, 245))
        .stroke(Stroke::new(1.0, Color32::from_rgb(238, 198, 203)))
        .corner_radius(20.0)
        .inner_margin(egui::Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.label(RichText::new("Build A Clear Review").size(18.0).strong().color(Color32::from_rgb(122, 20, 32)));
            ui.add_space(4.0);
            ui.label(
                RichText::new("Use the setup panel to load the clip, then review the generated snapshots and red-flag mechanics below.")
                    .color(Color32::from_rgb(122, 93, 98)),
            );
        });

    ui.add_space(18.0);
    ui.columns(2, |columns| {
        columns[0].vertical(|ui| {
            ui.label(RichText::new("Video Setup").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
            ui.add_space(8.0);
            ui.label(RichText::new("Choose a video file and select the camera angle.").color(Color32::from_rgb(118, 98, 102)));
            ui.add_space(16.0);
            ui.label(RichText::new("Video Path").color(Color32::from_rgb(118, 98, 102)));
            ui.add(egui::TextEdit::singleline(&mut app.selected_clip_path).hint_text("Absolute path...").desired_width(f32::INFINITY));
            ui.add_space(10.0);
            if secondary_button(ui, "Choose File").clicked() {
                if let Some(path) = choose_video_file() {
                    app.selected_clip_path = path.display().to_string();
                }
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                toggle_chip(ui, &mut app.selected_view, ClipView::Side);
                toggle_chip(ui, &mut app.selected_view, ClipView::Angle45);
            });
        });

        columns[1].vertical(|ui| {
            ui.label(RichText::new("Athlete Form").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
            ui.add_space(8.0);
            egui::Grid::new("athlete_profile_grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label(RichText::new("Name").color(Color32::from_rgb(118, 98, 102)));
                ui.add(egui::TextEdit::singleline(&mut app.athlete_name).desired_width(f32::INFINITY));
                ui.end_row();

                ui.label(RichText::new("Height (m)").color(Color32::from_rgb(118, 98, 102)));
                ui.add(egui::TextEdit::singleline(&mut app.athlete_height_m).desired_width(f32::INFINITY));
                ui.end_row();

                ui.label(RichText::new("Wingspan (m)").color(Color32::from_rgb(118, 98, 102)));
                ui.add(egui::TextEdit::singleline(&mut app.athlete_wingspan_m).desired_width(f32::INFINITY));
                ui.end_row();

                ui.label(RichText::new("Reach (m)").color(Color32::from_rgb(118, 98, 102)));
                ui.add(egui::TextEdit::singleline(&mut app.athlete_standing_reach_m).desired_width(f32::INFINITY));
                ui.end_row();

                ui.label(RichText::new("Handedness").color(Color32::from_rgb(118, 98, 102)));
                ui.horizontal(|ui| {
                    if ui.selectable_label(app.athlete_handedness == "right", "Right").clicked() { app.athlete_handedness = "right".to_string(); }
                    if ui.selectable_label(app.athlete_handedness == "left", "Left").clicked() { app.athlete_handedness = "left".to_string(); }
                });
                ui.end_row();
            });
        });
    });

    ui.add_space(18.0);
    ui.horizontal(|ui| {
        let button_text = if app.is_processing { "Analyzing..." } else { "Analyze Video" };
        if primary_button(ui, button_text).clicked() && !app.is_processing {
            app.start_analysis(ui.ctx());
        }
    });
}

fn processing_panel(ui: &mut egui::Ui, status_message: &str, preview_texture: Option<&TextureHandle>) {
    ui.horizontal(|ui| {
        ui.add(egui::Spinner::new().size(22.0));
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.label(RichText::new("Running analysis").size(22.0).strong().color(Color32::from_rgb(122, 20, 32)));
            ui.label(RichText::new(status_message).color(Color32::from_rgb(118, 98, 102)));
        });

        if let Some(texture) = preview_texture {
            ui.add_space(30.0);
            ui.add(
                egui::Image::new(texture)
                    .maintain_aspect_ratio(true)
                    .fit_to_exact_size(egui::vec2(260.0, 180.0))
                    .corner_radius(12.0)
            );
        }
    });
}

fn error_card(ui: &mut egui::Ui, message: &str) {
    shell_card(ui, |ui| {
        ui.label(RichText::new("Analysis Error").size(20.0).strong().color(Color32::from_rgb(164, 28, 44)));
        ui.add_space(6.0);
        ui.label(RichText::new(message).color(Color32::from_rgb(130, 71, 80)));
    });
}

fn analysis_overview(
    ui: &mut egui::Ui,
    result: &AnalysisRunResult,
    selected_shot_index: usize,
    snapshot: &TrainerSnapshot,
    preview_texture: Option<&TextureHandle>,
) {
    ui.label(RichText::new("Shot Analysis").size(30.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.label(
        RichText::new(format!(
            "{} • {} detected shots • {}",
            display_file_name(&result.clip_path),
            result.shot_records.len(),
            result.selected_view.label()
        ))
        .color(Color32::from_rgb(121, 99, 103)),
    );

    ui.add_space(14.0);
    if let Some(texture) = preview_texture {
        ui.add(
            egui::Image::new(texture)
                .maintain_aspect_ratio(true)
                .max_width(ui.available_width())
                .max_height(420.0)
                .corner_radius(18.0),
        );
    } else {
        ui.label(RichText::new("Preview snapshot unavailable for this clip.").color(Color32::from_rgb(121, 99, 103)));
    }

    ui.add_space(14.0);
    ui.columns(2, |columns| {
        columns[0].vertical(|ui| {
            stat_card(
                ui,
                "Shot Label",
                &format!("{:?}", snapshot.inference.label),
                "Best-fit style bucket based on your extracted mechanics.",
            );
        });
        columns[1].vertical(|ui| {
            stat_card(
                ui,
                "Selected Shot",
                &format!("{} / {}", selected_shot_index + 1, result.shot_records.len()),
                "Switch reps below to compare mechanics against a visual baseline.",
            );
        });
    });

    ui.add_space(10.0);
    ui.label(
        RichText::new("Explicit model scores now print to terminal output only.")
            .italics()
            .color(Color32::from_rgb(131, 104, 108)),
    );
}

fn shot_selector(ui: &mut egui::Ui, app: &mut JumpshotTrainerApp, result: &AnalysisRunResult) {
    ui.label(RichText::new("Pick The Rep To Review").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.add_space(8.0);
    ui.label(
        RichText::new("Each detected shot can be selected below. The coaching cards update instantly.")
            .color(Color32::from_rgb(118, 98, 102)),
    );
    ui.add_space(12.0);

    egui::ScrollArea::horizontal().show(ui, |ui| {
        ui.horizontal(|ui| {
            for (index, record) in result.shot_records.iter().enumerate() {
                let selected = app.selected_shot_index == index;
                let label = format!("Shot {} • {:.0} ms", index + 1, record.release_timing_ms.unwrap_or(0.0));
                let fill = if selected {
                    Color32::from_rgb(188, 31, 53)
                } else {
                    Color32::from_rgb(255, 247, 247)
                };
                let mut button = if let Some(texture) = app
                    .shot_thumbnail_textures
                    .get(index)
                    .and_then(|texture| texture.as_ref())
                {
                    egui::Button::image_and_text(
                        egui::Image::new(texture)
                            .fit_to_exact_size(egui::vec2(112.0, 84.0))
                            .corner_radius(10.0),
                        RichText::new(label).color(if selected {
                            Color32::from_rgb(255, 250, 250)
                        } else {
                            Color32::from_rgb(108, 48, 56)
                        }),
                    )
                } else {
                    egui::Button::new(
                        RichText::new(label).color(if selected {
                            Color32::from_rgb(255, 250, 250)
                        } else {
                            Color32::from_rgb(108, 48, 56)
                        }),
                    )
                };

                button = button
                    .fill(fill)
                    .corner_radius(12.0)
                    .stroke(Stroke::new(1.0, Color32::from_rgb(226, 187, 192)));

                if ui.add_sized([240.0, 112.0], button).clicked()
                {
                    app.selected_shot_index = index;
                    if let Err(error) = app.refresh_release_image(ui.ctx()) {
                        eprintln!("Release snapshot extraction failed: {error}");
                    }
                    app.log_selected_shot_score();
                }
            }
        });
    });

    ui.add_space(12.0);
    ui.label(
        RichText::new(format!(
            "Manifest: {}",
            display_file_name(&result.manifest_path)
        ))
        .color(Color32::from_rgb(134, 108, 112)),
    );
}

fn adjustments_panel(ui: &mut egui::Ui, snapshot: &TrainerSnapshot) {
    ui.label(RichText::new("What To Adjust").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.add_space(8.0);
    for (title, body) in coaching_actions(snapshot).into_iter().take(3) {
        advice_card(ui, &title, &body);
        ui.add_space(8.0);
    }
}

fn metric_summary(ui: &mut egui::Ui, input: &ShotInput, snapshot: &TrainerSnapshot) {
    ui.label(RichText::new("Mechanical Snapshot").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.add_space(10.0);

    ui.columns(2, |columns| {
        stat_card_compact(&mut columns[0], "Prototype", snapshot.inference.nearest_neighbor.as_str());
        stat_card_compact(&mut columns[1], "Elbow Flare", &format!("{:.1}°", input.elbow_flare));
        stat_card_compact(&mut columns[0], "Forearm", &format!("{:.1}°", input.forearm_verticality));
        stat_card_compact(&mut columns[1], "Release Timing", &format!("{:.0} ms", input.release_timing_ms));
        stat_card_compact(&mut columns[0], "Release Height", &format!("{:.2}x", input.release_height_ratio));
        stat_card_compact(&mut columns[1], "Knee Load", &format!("{:.1}°", input.knee_load));
        stat_card_compact(&mut columns[0], "Jump Height", &format!("{:.2} m", input.jump_height));
    });

    ui.add_space(10.0);
    ui.label(RichText::new("Quick model cue").color(Color32::from_rgb(118, 98, 102)));
    if let Some(feedback) = snapshot.inference.feedback.first() {
        ui.label(RichText::new(feedback).size(16.0).color(Color32::from_rgb(87, 49, 55)));
    }
}

fn stage_panel(ui: &mut egui::Ui, stages: &[StageFeedback], release_texture: Option<&TextureHandle>) {
    ui.label(RichText::new("Shot Phases").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.add_space(10.0);
    for stage in stages {
        stage_row(ui, stage);
        ui.add_space(8.0);
    }

    if let Some(texture) = release_texture {
        ui.add_space(8.0);
        ui.label(RichText::new("Release Snapshot").size(18.0).strong().color(Color32::from_rgb(93, 18, 30)));
        ui.label(
            RichText::new("The extracted release frame gives the coaching notes a concrete visual anchor.")
                .color(Color32::from_rgb(118, 98, 102)),
        );
        ui.add_space(8.0);
        ui.add(
            egui::Image::new(texture)
                .maintain_aspect_ratio(true)
                .max_width(ui.available_width())
                .max_height(320.0)
                .corner_radius(16.0),
        );
    }
}

fn overlay_panel(
    ui: &mut egui::Ui,
    input: &ShotInput,
    stage_feedback: &[StageFeedback],
    calibration: &CalibrationInput,
) {
    ui.label(RichText::new("Visual Review").size(22.0).strong().color(Color32::from_rgb(93, 18, 30)));
    ui.add_space(6.0);
    ui.label(
        RichText::new(format!(
            "Estimated athlete setup: {:.2} m height • {:.2} m reach • {:.1}° lens tilt",
            calibration.body_height_m,
            calibration.body_height_m * calibration.fingertip_reach_ratio,
            calibration.lens_tilt_deg
        ))
        .color(Color32::from_rgb(118, 98, 102)),
    );
    ui.add_space(12.0);
    draw_overlay_review(ui, input, stage_feedback);
}

fn engine_footer(ui: &mut egui::Ui, app: &mut JumpshotTrainerApp) {
    ui.add_space(8.0);
    if ui
        .button(if app.show_engine_details {
            "Hide Engine Details"
        } else {
            "Show Engine Details"
        })
        .clicked()
    {
        app.show_engine_details = !app.show_engine_details;
    }

    if app.show_engine_details {
        ui.add_space(10.0);
        shell_card(ui, |ui| {
            ui.label(RichText::new("Engine Status").size(18.0).strong());
            ui.add_space(8.0);
            ui.label(format!(
                "{} training examples available in the shared corpus.",
                app.loaded_corpus.dataset_summary.example_count
            ));
            ui.label(format!(
                "{} paired-view examples currently support the background score model.",
                app.loaded_corpus.dataset_summary.paired_view_examples
            ));
            if let Some(result) = &app.analysis_result {
                ui.add_space(6.0);
                ui.label(format!("Latest processed session: {}", result.session_json.display()));
            }
        });
    }
}

fn coaching_actions(snapshot: &TrainerSnapshot) -> Vec<(String, String)> {
    let mut issues = snapshot.diagnostics.clone();
    issues.sort_by(|a, b| {
        severity_rank(b.severity)
            .cmp(&severity_rank(a.severity))
            .then_with(|| b.delta.abs().partial_cmp(&a.delta.abs()).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut actions = Vec::new();
    for issue in issues.into_iter().filter(|issue| issue.severity != DiagnosticSeverity::Optimal) {
        let body = match issue.metric.as_str() {
            "Elbow Flare" => "Keep the shooting elbow tucked closer to your shot line so the release stays compact.".to_string(),
            "Forearm Verticality" => "Get the wrist stacked over the elbow earlier so the forearm stays more vertical at the set point.".to_string(),
            "Release Timing" => "Let the ball go sooner so the release happens closer to the top of the jump.".to_string(),
            "Release Height Ratio" => "Raise the finish and get into the shot pocket earlier so the release point climbs.".to_string(),
            "Knee Load" => "Smooth out the dip so the lower body loads without extra wasted motion.".to_string(),
            "Elbow Flexion" => "Keep the set point more compact so the elbow stays in a tighter window.".to_string(),
            _ => issue.message.clone(),
        };
        actions.push((issue.metric.clone(), body));
    }

    if actions.is_empty() {
        actions.push((
            "Good Base".to_string(),
            "The shot is sitting in a healthy window right now. Focus on repeating the same rhythm rep after rep.".to_string(),
        ));
    }

    actions
}

fn severity_rank(severity: DiagnosticSeverity) -> u8 {
    match severity {
        DiagnosticSeverity::Major => 3,
        DiagnosticSeverity::Minor => 2,
        DiagnosticSeverity::Optimal => 1,
    }
}

fn display_file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();
    visuals.override_text_color = Some(Color32::from_rgb(74, 37, 43));
    visuals.panel_fill = Color32::from_rgb(255, 251, 251);
    visuals.window_fill = Color32::from_rgb(255, 251, 251);
    visuals.extreme_bg_color = Color32::from_rgb(255, 255, 255);
    visuals.faint_bg_color = Color32::from_rgb(255, 245, 246);
    visuals.selection.bg_fill = Color32::from_rgb(193, 35, 57);
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(255, 245, 246));
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(255, 251, 251);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(237, 212, 215));
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(255, 255, 255);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(231, 205, 209));
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(255, 241, 243);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.2, Color32::from_rgb(200, 64, 84));
    visuals.widgets.active.bg_fill = Color32::from_rgb(193, 35, 57);
    visuals.widgets.active.bg_stroke = Stroke::new(1.2, Color32::from_rgb(193, 35, 57));
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(16.0, 16.0);
    style.spacing.button_padding = egui::vec2(18.0, 12.0);
    style.visuals.window_corner_radius = 16.0.into();
    ctx.set_style(style);
}

fn paint_background(ctx: &egui::Context) {
    let painter = ctx.layer_painter(egui::LayerId::background());
    let rect = ctx.content_rect();
    painter.rect_filled(rect, 0.0, Color32::from_rgb(255, 249, 249));
    painter.circle_filled(
        rect.left_top() + egui::vec2(rect.width() * 0.18, rect.height() * 0.14),
        rect.width() * 0.18,
        Color32::from_rgba_unmultiplied(215, 46, 74, 18),
    );
    painter.circle_filled(
        rect.right_bottom() - egui::vec2(rect.width() * 0.12, rect.height() * 0.18),
        rect.width() * 0.22,
        Color32::from_rgba_unmultiplied(140, 12, 35, 14),
    );
}

fn shell_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(Color32::from_rgb(255, 255, 255))
        .stroke(Stroke::new(1.0, Color32::from_rgb(237, 210, 214)))
        .shadow(egui::epaint::Shadow {
            offset: [0, 10],
            blur: 24,
            spread: 0,
            color: Color32::from_rgba_unmultiplied(122, 20, 32, 18),
        })
        .corner_radius(24.0)
        .inner_margin(egui::Margin::symmetric(22, 20))
        .show(ui, add_contents);
}

fn primary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(
            RichText::new(label)
                .size(16.0)
                .strong()
                .color(Color32::from_rgb(255, 249, 249)),
        )
            .fill(Color32::from_rgb(193, 35, 57))
            .stroke(Stroke::new(1.0, Color32::from_rgb(146, 23, 40)))
            .corner_radius(16.0),
    )
}

fn secondary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(
            RichText::new(label)
                .size(15.0)
                .strong()
                .color(Color32::from_rgb(146, 23, 40)),
        )
        .fill(Color32::from_rgb(255, 244, 245))
        .stroke(Stroke::new(1.0, Color32::from_rgb(220, 172, 180)))
        .corner_radius(14.0),
    )
}

fn toggle_chip(ui: &mut egui::Ui, selected_view: &mut ClipView, option: ClipView) {
    let selected = *selected_view == option;
    let fill = if selected {
        Color32::from_rgb(193, 35, 57)
    } else {
        Color32::from_rgb(255, 247, 247)
    };
    if ui
        .add(
            egui::Button::new(
                RichText::new(option.label())
                    .color(if selected {
                        Color32::from_rgb(255, 249, 249)
                    } else {
                        Color32::from_rgb(108, 48, 56)
                    }),
            )
                .fill(fill)
                .corner_radius(999.0)
                .stroke(Stroke::new(1.0, Color32::from_rgb(226, 187, 192))),
        )
        .clicked()
    {
        *selected_view = option;
    }
}

fn stat_card(ui: &mut egui::Ui, title: &str, value: &str, caption: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(255, 246, 247))
        .corner_radius(18.0)
        .stroke(Stroke::new(1.0, Color32::from_rgb(237, 208, 212)))
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_min_height(128.0);
            ui.label(RichText::new(title).color(Color32::from_rgb(141, 87, 95)));
            ui.add_space(8.0);
            ui.label(RichText::new(value).size(28.0).strong().color(Color32::from_rgb(93, 18, 30)));
            ui.add_space(10.0);
            ui.label(RichText::new(caption).size(14.0).color(Color32::from_rgb(121, 99, 103)));
        });
}

fn stat_card_compact(ui: &mut egui::Ui, title: &str, value: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(255, 248, 249))
        .corner_radius(14.0)
        .stroke(Stroke::new(1.0, Color32::from_rgb(237, 208, 212)))
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_min_height(84.0);
            ui.label(RichText::new(title).size(12.0).color(Color32::from_rgb(141, 87, 95)));
            ui.add_space(4.0);
            ui.label(RichText::new(value).size(22.0).strong().color(Color32::from_rgb(87, 49, 55)));
        });
}

fn advice_card(ui: &mut egui::Ui, title: &str, body: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(255, 243, 244))
        .corner_radius(16.0)
        .stroke(Stroke::new(1.0, Color32::from_rgb(236, 198, 204)))
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.label(RichText::new(title).size(18.0).strong().color(Color32::from_rgb(122, 20, 32)));
            ui.add_space(6.0);
            ui.label(RichText::new(body).color(Color32::from_rgb(110, 79, 84)));
        });
}

fn stage_row(ui: &mut egui::Ui, stage: &StageFeedback) {
    let color = match stage.color_hint {
        DiagnosticSeverity::Optimal => Color32::from_rgb(100, 196, 119),
        DiagnosticSeverity::Minor => Color32::from_rgb(224, 184, 90),
        DiagnosticSeverity::Major => Color32::from_rgb(224, 108, 88),
    };

    egui::Frame::new()
        .fill(Color32::from_rgb(255, 247, 248))
        .corner_radius(14.0)
        .stroke(Stroke::new(1.0, Color32::from_rgb(237, 208, 212)))
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(color, RichText::new(format!("{:?}", stage.stage)).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{} / 100", stage.score))
                            .strong()
                            .color(Color32::from_rgb(122, 20, 32)),
                    );
                });
            });
            ui.add_space(6.0);
            ui.label(RichText::new(stage.coaching_note.as_str()).color(Color32::from_rgb(110, 79, 84)));
        });
}

fn draw_overlay_review(ui: &mut egui::Ui, input: &ShotInput, stages: &[StageFeedback]) {
    let desired_size = Vec2::new(ui.available_width(), 280.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 22.0, Color32::from_rgb(255, 244, 245));

    let center_x = rect.center().x - 24.0;
    let floor_y = rect.bottom() - 28.0;
    let hip_y = floor_y - 98.0;
    let shoulder_y = hip_y - 78.0;
    let knee_x = center_x - 12.0;
    let ankle_x = center_x - 4.0;
    let elbow_x = center_x + (input.elbow_flare * 3.2).clamp(0.0, 48.0);
    let wrist_x = elbow_x + 14.0;
    let wrist_y = floor_y - (input.release_height_ratio * 94.0);

    let load_color = stage_color_from_feedback(stages, ShotStage::Load);
    let set_color = stage_color_from_feedback(stages, ShotStage::SetPoint);
    let release_color = stage_color_from_feedback(stages, ShotStage::Release);

    painter.line_segment(
        [egui::pos2(center_x, shoulder_y), egui::pos2(center_x, hip_y)],
        Stroke::new(7.0, Color32::from_rgb(235, 201, 153)),
    );
    painter.line_segment(
        [egui::pos2(center_x, hip_y), egui::pos2(knee_x, floor_y - 44.0)],
        Stroke::new(7.0, load_color),
    );
    painter.line_segment(
        [egui::pos2(knee_x, floor_y - 44.0), egui::pos2(ankle_x, floor_y)],
        Stroke::new(7.0, load_color),
    );
    painter.line_segment(
        [egui::pos2(center_x, shoulder_y), egui::pos2(elbow_x, shoulder_y + 34.0)],
        Stroke::new(7.0, set_color),
    );
    painter.line_segment(
        [egui::pos2(elbow_x, shoulder_y + 34.0), egui::pos2(wrist_x, wrist_y)],
        Stroke::new(7.0, release_color),
    );

    painter.line_segment(
        [egui::pos2(center_x, shoulder_y - 34.0), egui::pos2(center_x + 2.0, shoulder_y - 6.0)],
        Stroke::new(7.0, Color32::from_rgb(235, 201, 153)),
    );
    painter.circle_filled(egui::pos2(center_x, shoulder_y - 52.0), 18.0, Color32::from_rgb(248, 226, 186));

    painter.text(
        rect.left_top() + egui::vec2(18.0, 16.0),
        Align2::LEFT_TOP,
        "Color guide: green = solid, amber = slight issue, red = clear fix",
        FontId::proportional(15.0),
        Color32::from_rgb(120, 83, 89),
    );
}

fn stage_color_from_feedback(stages: &[StageFeedback], stage: ShotStage) -> Color32 {
    let Some(stage_feedback) = stages.iter().find(|item| item.stage == stage) else {
        return Color32::from_rgb(155, 163, 170);
    };
    match stage_feedback.color_hint {
        DiagnosticSeverity::Optimal => Color32::from_rgb(108, 201, 125),
        DiagnosticSeverity::Minor => Color32::from_rgb(234, 190, 95),
        DiagnosticSeverity::Major => Color32::from_rgb(230, 114, 91),
    }
}
