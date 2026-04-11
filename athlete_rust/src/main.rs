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
use eframe::egui::{self, Align2, Color32, FontId, RichText, Stroke, Vec2};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::thread;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 960.0])
            .with_min_inner_size([1120.0, 760.0])
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
    athlete_profile_path: String,
    analysis_receiver: Option<Receiver<WorkerEvent>>,
    is_processing: bool,
    status_message: String,
    error_message: Option<String>,
    loaded_corpus: LoadedCorpus,
    analysis_result: Option<AnalysisRunResult>,
    selected_shot_index: usize,
    show_engine_details: bool,
}

impl JumpshotTrainerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);

        let athlete_rust_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let project_root = athlete_rust_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| athlete_rust_dir.clone());
        let athlete_profile_path = project_root
            .join("datasets/calibration_20_shot/annotations/athlete_profile.json")
            .display()
            .to_string();

        Self {
            project_root: project_root.clone(),
            selected_clip_path: String::new(),
            selected_view: ClipView::Side,
            athlete_profile_path,
            analysis_receiver: None,
            is_processing: false,
            status_message: "Drop a shooting clip into the window or paste a path to start.".to_string(),
            error_message: None,
            loaded_corpus: load_corpus_state(&project_root),
            analysis_result: None,
            selected_shot_index: 0,
            show_engine_details: false,
        }
    }

    fn poll_background_worker(&mut self) {
        let mut should_clear = false;
        if let Some(receiver) = &self.analysis_receiver {
            while let Ok(event) = receiver.try_recv() {
                match event {
                    WorkerEvent::Status(message) => {
                        self.status_message = message;
                    }
                    WorkerEvent::Completed(result) => {
                        self.loaded_corpus = result.corpus.clone();
                        self.analysis_result = Some(result);
                        self.selected_shot_index = 0;
                        self.is_processing = false;
                        self.error_message = None;
                        self.status_message = "Analysis complete. Review the shot and make the adjustment cues below.".to_string();
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
        }
        if should_clear {
            self.analysis_receiver = None;
        }
    }

    fn start_analysis(&mut self) {
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

        let project_root = self.project_root.clone();
        let athlete_profile = PathBuf::from(self.athlete_profile_path.trim());
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

    fn selected_shot_view(&self) -> Option<(ShotInput, CalibrationInput, TrainerSnapshot, Option<f32>)> {
        let record = self.selected_record()?;
        let input = shot_input_from_record(record);
        let calibration = calibration_input_from_record(record);
        let snapshot = analyze_shot(&input, &calibration);
        let supervised = predict_supervised_score(
            &self.loaded_corpus.supervised_model,
            &feature_vector_from_shot_input(
                &input,
                calibration.body_height_m,
                calibration.body_height_m * calibration.arm_span_ratio,
                calibration.body_height_m * calibration.fingertip_reach_ratio,
                record.distance_ft.unwrap_or(15.0),
                record.paired_view_available,
            ),
        );
        Some((input, calibration, snapshot, supervised))
    }
}

impl eframe::App for JumpshotTrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_dropped_file(ctx, &mut self.selected_clip_path);
        self.poll_background_worker();
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
                            processing_panel(ui, &self.status_message);
                        });
                    }

                    if let Some(error) = &self.error_message {
                        ui.add_space(14.0);
                        error_card(ui, error);
                    }

                    let analysis_result = self.analysis_result.clone();
                    if let Some(result) = analysis_result.as_ref() {
                        ui.add_space(18.0);
                        if let Some((input, calibration, snapshot, supervised_score)) = self.selected_shot_view() {
                            shell_card(ui, |ui| {
                                analysis_overview(
                                    ui,
                                    result,
                                    self.selected_shot_index,
                                    &snapshot,
                                    supervised_score,
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
                                    metric_summary(ui, &input, supervised_score, &snapshot);
                                });
                                shell_card(right, |ui| {
                                    stage_panel(ui, &snapshot.stage_feedback);
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
            .arg("30")
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
        ui.label(RichText::new("JumpShot Trainer").size(42.0).strong().color(Color32::from_rgb(245, 239, 228)));
        ui.add_space(6.0);
        ui.label(
            RichText::new("Drop in a clip, let the model break down your form, and get the next adjustment to make.")
                .size(17.0)
                .color(Color32::from_rgb(196, 204, 208)),
        );
    });
}

fn upload_panel(ui: &mut egui::Ui, app: &mut JumpshotTrainerApp) {
    ui.label(RichText::new("Analyze A Jump Shot").size(24.0).strong());
    ui.add_space(6.0);
    ui.label(
        RichText::new("Use a side view or front-quarter clip. The app will process the video in the background and return coaching feedback.")
            .color(Color32::from_rgb(176, 185, 191)),
    );
    ui.add_space(16.0);

    egui::Frame::new()
        .fill(Color32::from_rgb(19, 26, 31))
        .stroke(Stroke::new(1.0, Color32::from_rgb(53, 72, 80)))
        .corner_radius(18.0)
        .inner_margin(egui::Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.label(RichText::new("Drop a video here or paste a path below").size(18.0).strong());
            ui.add_space(10.0);
            ui.add(
                egui::TextEdit::singleline(&mut app.selected_clip_path)
                    .hint_text("/absolute/path/to/video.mp4")
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                toggle_chip(ui, &mut app.selected_view, ClipView::Side);
                toggle_chip(ui, &mut app.selected_view, ClipView::Angle45);
            });

            ui.add_space(12.0);
            ui.collapsing("Advanced", |ui| {
                ui.label(RichText::new("Athlete profile path").color(Color32::from_rgb(176, 185, 191)));
                ui.add(
                    egui::TextEdit::singleline(&mut app.athlete_profile_path)
                        .desired_width(f32::INFINITY),
                );
            });

            ui.add_space(14.0);
            let button_text = if app.is_processing { "Analyzing..." } else { "Analyze Video" };
            if primary_button(ui, button_text).clicked() && !app.is_processing {
                app.start_analysis();
            }
        });
}

fn processing_panel(ui: &mut egui::Ui, status_message: &str) {
    ui.horizontal(|ui| {
        ui.add(egui::Spinner::new().size(22.0));
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.label(RichText::new("Running analysis").size(20.0).strong());
            ui.label(RichText::new(status_message).color(Color32::from_rgb(176, 185, 191)));
        });
    });
}

fn error_card(ui: &mut egui::Ui, message: &str) {
    shell_card(ui, |ui| {
        ui.label(RichText::new("Analysis Error").size(20.0).strong().color(Color32::from_rgb(255, 214, 214)));
        ui.add_space(6.0);
        ui.label(RichText::new(message).color(Color32::from_rgb(255, 188, 188)));
    });
}

fn analysis_overview(
    ui: &mut egui::Ui,
    result: &AnalysisRunResult,
    selected_shot_index: usize,
    snapshot: &TrainerSnapshot,
    supervised_score: Option<f32>,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new("Shot Analysis").size(26.0).strong());
            ui.label(
                RichText::new(format!(
                    "{} • {} detected shots • {}",
                    display_file_name(&result.clip_path),
                    result.shot_records.len(),
                    result.selected_view.label()
                ))
                .color(Color32::from_rgb(176, 185, 191)),
            );
        });
        ui.add_space(ui.available_width() - 260.0);
        score_pill(
            ui,
            &format!("{} / 100", snapshot.inference.score),
            score_color(snapshot.inference.score),
        );
    });

    ui.add_space(14.0);
    ui.columns(3, |columns| {
        let score_text = supervised_score
            .map(|score| format!("{:.0} / 100", score * 100.0))
            .unwrap_or_else(|| "Not ready".to_string());
        stat_card(&mut columns[0], "Model Score", &score_text, "Overall confidence from the current Rust scoring model.");
        stat_card(
            &mut columns[1],
            "Shot Label",
            &format!("{:?}", snapshot.inference.label),
            "Best-fit style bucket based on your extracted mechanics.",
        );
        stat_card(
            &mut columns[2],
            "Selected Shot",
            &format!("{} / {}", selected_shot_index + 1, result.shot_records.len()),
            "If the clip contains multiple reps, you can switch between them below.",
        );
    });
}

fn shot_selector(ui: &mut egui::Ui, app: &mut JumpshotTrainerApp, result: &AnalysisRunResult) {
    ui.label(RichText::new("Pick The Rep To Review").size(20.0).strong());
    ui.add_space(8.0);
    ui.label(
        RichText::new("Each detected shot can be selected below. The coaching cards update instantly.")
            .color(Color32::from_rgb(176, 185, 191)),
    );
    ui.add_space(12.0);

    egui::ScrollArea::horizontal().show(ui, |ui| {
        ui.horizontal(|ui| {
            for (index, record) in result.shot_records.iter().enumerate() {
                let selected = app.selected_shot_index == index;
                let label = format!("Shot {} • {:.0} ms", index + 1, record.release_timing_ms.unwrap_or(0.0));
                let fill = if selected {
                    Color32::from_rgb(215, 119, 64)
                } else {
                    Color32::from_rgb(25, 33, 39)
                };
                if ui
                    .add(
                        egui::Button::new(label)
                            .fill(fill)
                            .corner_radius(12.0)
                            .stroke(Stroke::new(1.0, Color32::from_rgb(72, 90, 98))),
                    )
                    .clicked()
                {
                    app.selected_shot_index = index;
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
        .color(Color32::from_rgb(124, 142, 150)),
    );
}

fn adjustments_panel(ui: &mut egui::Ui, snapshot: &TrainerSnapshot) {
    ui.label(RichText::new("What To Adjust").size(20.0).strong());
    ui.add_space(8.0);
    for (title, body) in coaching_actions(snapshot).into_iter().take(3) {
        advice_card(ui, &title, &body);
        ui.add_space(8.0);
    }
}

fn metric_summary(
    ui: &mut egui::Ui,
    input: &ShotInput,
    supervised_score: Option<f32>,
    snapshot: &TrainerSnapshot,
) {
    ui.label(RichText::new("Mechanical Snapshot").size(20.0).strong());
    ui.add_space(10.0);

    let score_text = supervised_score
        .map(|score| format!("{:.0} / 100", score * 100.0))
        .unwrap_or_else(|| format!("{} / 100", snapshot.inference.score));

    ui.columns(2, |columns| {
        stat_card_compact(&mut columns[0], "Overall", &score_text);
        stat_card_compact(&mut columns[1], "Prototype", snapshot.inference.nearest_neighbor.as_str());
        stat_card_compact(&mut columns[0], "Elbow Flare", &format!("{:.1}°", input.elbow_flare));
        stat_card_compact(&mut columns[1], "Forearm", &format!("{:.1}°", input.forearm_verticality));
        stat_card_compact(&mut columns[0], "Release Timing", &format!("{:.0} ms", input.release_timing_ms));
        stat_card_compact(&mut columns[1], "Release Height", &format!("{:.2}x", input.release_height_ratio));
        stat_card_compact(&mut columns[0], "Knee Load", &format!("{:.1}°", input.knee_load));
        stat_card_compact(&mut columns[1], "Jump Height", &format!("{:.2} m", input.jump_height));
    });

    ui.add_space(10.0);
    ui.label(RichText::new("Quick model cue").color(Color32::from_rgb(176, 185, 191)));
    if let Some(feedback) = snapshot.inference.feedback.first() {
        ui.label(RichText::new(feedback).size(15.0));
    }
}

fn stage_panel(ui: &mut egui::Ui, stages: &[StageFeedback]) {
    ui.label(RichText::new("Shot Phases").size(20.0).strong());
    ui.add_space(10.0);
    for stage in stages {
        stage_row(ui, stage);
        ui.add_space(8.0);
    }
}

fn overlay_panel(
    ui: &mut egui::Ui,
    input: &ShotInput,
    stage_feedback: &[StageFeedback],
    calibration: &CalibrationInput,
) {
    ui.label(RichText::new("Visual Review").size(20.0).strong());
    ui.add_space(6.0);
    ui.label(
        RichText::new(format!(
            "Estimated athlete setup: {:.2} m height • {:.2} m reach • {:.1}° lens tilt",
            calibration.body_height_m,
            calibration.body_height_m * calibration.fingertip_reach_ratio,
            calibration.lens_tilt_deg
        ))
        .color(Color32::from_rgb(176, 185, 191)),
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
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(240, 236, 226));
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(11, 16, 20);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(20, 28, 34);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(33, 45, 52);
    visuals.widgets.active.bg_fill = Color32::from_rgb(215, 119, 64);
    visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(240, 236, 226);
    visuals.selection.bg_fill = Color32::from_rgb(215, 119, 64);
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(255, 230, 210));
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 12.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.visuals.window_corner_radius = 20.0.into();
    style.visuals.panel_fill = Color32::from_rgb(10, 15, 18);
    ctx.set_style(style);
}

fn paint_background(ctx: &egui::Context) {
    let painter = ctx.layer_painter(egui::LayerId::background());
    let rect = ctx.content_rect();
    painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 12, 15));
    painter.circle_filled(
        rect.left_top() + egui::vec2(rect.width() * 0.18, rect.height() * 0.16),
        rect.width() * 0.18,
        Color32::from_rgba_unmultiplied(212, 110, 51, 34),
    );
    painter.circle_filled(
        rect.right_bottom() - egui::vec2(rect.width() * 0.14, rect.height() * 0.18),
        rect.width() * 0.22,
        Color32::from_rgba_unmultiplied(45, 122, 120, 24),
    );
}

fn shell_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(Color32::from_rgb(13, 19, 24))
        .stroke(Stroke::new(1.0, Color32::from_rgb(42, 57, 65)))
        .corner_radius(22.0)
        .inner_margin(egui::Margin::symmetric(20, 18))
        .show(ui, add_contents);
}

fn primary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(16.0).strong())
            .fill(Color32::from_rgb(215, 119, 64))
            .stroke(Stroke::new(1.0, Color32::from_rgb(244, 201, 172)))
            .corner_radius(14.0),
    )
}

fn toggle_chip(ui: &mut egui::Ui, selected_view: &mut ClipView, option: ClipView) {
    let selected = *selected_view == option;
    let fill = if selected {
        Color32::from_rgb(56, 132, 130)
    } else {
        Color32::from_rgb(25, 33, 39)
    };
    if ui
        .add(
            egui::Button::new(option.label())
                .fill(fill)
                .corner_radius(999.0)
                .stroke(Stroke::new(1.0, Color32::from_rgb(68, 88, 95))),
        )
        .clicked()
    {
        *selected_view = option;
    }
}

fn stat_card(ui: &mut egui::Ui, title: &str, value: &str, caption: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(18, 25, 30))
        .corner_radius(18.0)
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_min_height(128.0);
            ui.label(RichText::new(title).color(Color32::from_rgb(170, 180, 186)));
            ui.add_space(8.0);
            ui.label(RichText::new(value).size(24.0).strong());
            ui.add_space(10.0);
            ui.label(RichText::new(caption).size(13.0).color(Color32::from_rgb(143, 156, 163)));
        });
}

fn stat_card_compact(ui: &mut egui::Ui, title: &str, value: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(18, 25, 30))
        .corner_radius(14.0)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_min_height(74.0);
            ui.label(RichText::new(title).size(12.0).color(Color32::from_rgb(170, 180, 186)));
            ui.add_space(4.0);
            ui.label(RichText::new(value).size(20.0).strong());
        });
}

fn advice_card(ui: &mut egui::Ui, title: &str, body: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(19, 27, 31))
        .corner_radius(16.0)
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.label(RichText::new(title).size(16.0).strong());
            ui.add_space(6.0);
            ui.label(RichText::new(body).color(Color32::from_rgb(185, 193, 198)));
        });
}

fn score_pill(ui: &mut egui::Ui, value: &str, fill: Color32) {
    egui::Frame::new()
        .fill(fill)
        .corner_radius(999.0)
        .inner_margin(egui::Margin::symmetric(18, 10))
        .show(ui, |ui| {
            ui.label(RichText::new(value).strong().size(18.0).color(Color32::from_rgb(12, 15, 18)));
        });
}

fn score_color(score: u8) -> Color32 {
    if score >= 85 {
        Color32::from_rgb(108, 201, 125)
    } else if score >= 65 {
        Color32::from_rgb(234, 190, 95)
    } else {
        Color32::from_rgb(230, 114, 91)
    }
}

fn stage_row(ui: &mut egui::Ui, stage: &StageFeedback) {
    let color = match stage.color_hint {
        DiagnosticSeverity::Optimal => Color32::from_rgb(100, 196, 119),
        DiagnosticSeverity::Minor => Color32::from_rgb(224, 184, 90),
        DiagnosticSeverity::Major => Color32::from_rgb(224, 108, 88),
    };

    egui::Frame::new()
        .fill(Color32::from_rgb(18, 24, 29))
        .corner_radius(14.0)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(color, RichText::new(format!("{:?}", stage.stage)).strong());
                ui.add_space(ui.available_width() - 130.0);
                ui.label(RichText::new(format!("{} / 100", stage.score)).strong());
            });
            ui.add_space(6.0);
            ui.label(RichText::new(stage.coaching_note.as_str()).color(Color32::from_rgb(177, 187, 193)));
        });
}

fn draw_overlay_review(ui: &mut egui::Ui, input: &ShotInput, stages: &[StageFeedback]) {
    let desired_size = Vec2::new(ui.available_width(), 280.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 22.0, Color32::from_rgb(18, 24, 29));

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
        Color32::from_rgb(196, 204, 208),
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
