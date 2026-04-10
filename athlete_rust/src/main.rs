use biomech_ai::ingest::load_janitor_shot_records;
use biomech_ai::trainer::{analyze_shot, build_training_session, default_calibration_input};
use biomech_ai::training::{
    build_training_examples, calibration_input_from_record, evaluate_model_readiness, shot_input_from_record,
    summarize_processed_sessions, summarize_training_dataset,
};
use biomech_ai::types::{
    CalibrationInput, DiagnosticSeverity, ModelReadiness, ProcessedSessionSummary, SessionAudit,
    SessionShotSummary, ShotInput, ShotQualityLabel, ShotStage, StageFeedback, TrainingDatasetSummary,
};
use eframe::egui::{self, Align2, Color32, FontFamily, FontId, RichText, Stroke, Vec2};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1540.0, 980.0])
            .with_min_inner_size([1280.0, 820.0])
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
enum AppScreen {
    Calibration,
    Dashboard,
}

struct JumpshotTrainerApp {
    calibration_input: CalibrationInput,
    input: ShotInput,
    session_audit: SessionAudit,
    shots: Vec<SessionShotSummary>,
    screen: AppScreen,
    dataset_status: String,
    dataset_summary: TrainingDatasetSummary,
    model_readiness: ModelReadiness,
    processed_sessions: Vec<ProcessedSessionSummary>,
}

impl JumpshotTrainerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);

        let default_input = ShotInput {
            elbow_flexion: 87.0,
            knee_load: 108.0,
            forearm_verticality: 89.0,
            elbow_flare: 4.0,
            release_height_ratio: 1.29,
            release_timing_ms: 332.0,
            release_at_apex_offset_ms: 16.0,
            jump_height: 0.39,
        };

        let empty_summary = TrainingDatasetSummary {
            example_count: 0,
            paired_view_examples: 0,
            label_balance: vec![],
            average_target_score: 0.0,
            feature_count: 0,
        };

        let corpus_path = std::path::Path::new("../datasets/shared/processed/training_corpus.parquet");
        let parquet_path = std::path::Path::new("../datasets/shared/processed/calibration_20_shot_shot_records.parquet");
        let preferred_path = if corpus_path.exists() { corpus_path } else { parquet_path };
        let (input, calibration_input, dataset_status, dataset_summary, model_readiness, processed_sessions) =
            if preferred_path.exists() {
            match load_janitor_shot_records(preferred_path) {
                Ok(records) if !records.is_empty() => {
                    let examples = build_training_examples(&records);
                    let summary = summarize_training_dataset(&examples);
                    let readiness = evaluate_model_readiness(&summary);
                    let processed_sessions = summarize_processed_sessions(&records);
                    let first = &records[0];
                    (
                        shot_input_from_record(first),
                        calibration_input_from_record(first),
                        format!("Linked to janitor export at {}", preferred_path.display()),
                        summary,
                        readiness,
                        processed_sessions,
                    )
                }
                Ok(_) => (
                    default_input,
                    default_calibration_input(),
                    "Janitor parquet found but empty. Using local demo state.".to_string(),
                    empty_summary.clone(),
                    evaluate_model_readiness(&empty_summary),
                    vec![],
                ),
                Err(error) => (
                    default_input,
                    default_calibration_input(),
                    format!("Failed to load janitor parquet: {error}"),
                    empty_summary.clone(),
                    evaluate_model_readiness(&empty_summary),
                    vec![],
                ),
            }
        } else {
            (
                default_input,
                default_calibration_input(),
                format!("No janitor parquet yet at {}. Using local demo state.", preferred_path.display()),
                empty_summary.clone(),
                evaluate_model_readiness(&empty_summary),
                vec![],
            )
        };

        let (shots, session_audit) = build_training_session(&input, &calibration_input, 8);

        Self {
            calibration_input,
            input,
            session_audit,
            shots,
            screen: AppScreen::Calibration,
            dataset_status,
            dataset_summary,
            model_readiness,
            processed_sessions,
        }
    }

    fn regenerate_session(&mut self) {
        let (shots, audit) = build_training_session(&self.input, &self.calibration_input, 8);
        self.shots = shots;
        self.session_audit = audit;
    }
}

impl eframe::App for JumpshotTrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        paint_background(ctx);
        let snapshot = analyze_shot(&self.input, &self.calibration_input);

        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(13, 18, 22))
                    .inner_margin(egui::Margin::symmetric(22, 18)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("JumpShot Trainer").size(34.0).strong());
                        ui.label(
                            RichText::new("Rust biomechanics lab for calibration, diagnostics, and trainable shot data")
                                .size(15.0)
                                .color(Color32::from_rgb(177, 190, 196)),
                        );
                    });
                    ui.add_space(ui.available_width() - 320.0);
                    pill_label(
                        ui,
                        if self.model_readiness.is_ready {
                            "Training Ready"
                        } else {
                            "Collecting Dataset"
                        },
                        readiness_color(self.model_readiness.score),
                    );
                });
                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    nav_button(ui, &mut self.screen, AppScreen::Calibration, "Calibration");
                    nav_button(ui, &mut self.screen, AppScreen::Dashboard, "Performance Dashboard");
                    status_chip(ui, &self.dataset_status);
                });
            });

        match self.screen {
            AppScreen::Calibration => self.render_calibration(ctx, &snapshot),
            AppScreen::Dashboard => self.render_dashboard(ctx, &snapshot),
        }
    }
}

impl JumpshotTrainerApp {
    fn render_calibration(&mut self, ctx: &egui::Context, snapshot: &biomech_ai::trainer::TrainerSnapshot) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(24, 22)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    hero_metrics(ui, &self.dataset_summary, &self.model_readiness, snapshot.inference.score);
                    ui.add_space(16.0);

                    ui.columns(2, |columns| {
                        let (left_cols, right_cols) = columns.split_at_mut(1);
                        let left = &mut left_cols[0];
                        let right = &mut right_cols[0];

                        section_card(left, "Calibration Deck", "Dial in athlete geometry and camera placement before the first session.", |ui| {
                            slider(ui, &mut self.calibration_input.body_height_m, 1.45..=2.25, "Body Height");
                            slider(ui, &mut self.calibration_input.shoulder_width_m, 0.32..=0.62, "Shoulder Width");
                            slider(ui, &mut self.calibration_input.arm_span_ratio, 0.92..=1.12, "Arm Span Ratio");
                            slider(ui, &mut self.calibration_input.fingertip_reach_ratio, 1.2..=1.46, "Standing Reach Ratio");
                            slider(ui, &mut self.calibration_input.camera_distance_m, 2.0..=8.0, "Camera Distance");
                            slider(ui, &mut self.calibration_input.lens_tilt_deg, -10.0..=15.0, "Lens Tilt");
                            ui.add_space(10.0);
                            if accent_button(ui, "Lock Calibration + Open Dashboard").clicked() {
                                self.regenerate_session();
                                self.screen = AppScreen::Dashboard;
                            }
                        });

                        section_card(right, "Athlete Geometry", "Estimated values that will be reused by the feature and training pipeline.", |ui| {
                            metric_pair(ui, "Estimated Wingspan", &format!("{:.2} m", snapshot.calibration.estimated_wingspan_m));
                            metric_pair(ui, "Standing Reach", &format!("{:.2} m", snapshot.calibration.estimated_standing_reach_m));
                            metric_pair(ui, "Camera Angle", &format!("{:.1} deg", snapshot.calibration.estimated_camera_angle_deg));
                            metric_pair(ui, "Calibration Confidence", &format!("{:.0}%", snapshot.calibration.confidence * 100.0));
                            ui.add_space(12.0);
                            draw_calibration_preview(ui, &self.calibration_input);
                        });
                    });

                    ui.add_space(16.0);
                    ui.columns(2, |columns| {
                        let (left_cols, right_cols) = columns.split_at_mut(1);
                        let left = &mut left_cols[0];
                        let right = &mut right_cols[0];

                        section_card(left, "Training Readiness", "This checks whether the current dataset is strong enough for the first Rust-side training pass.", |ui| {
                            readiness_panel(ui, &self.model_readiness);
                        });

                        section_card(right, "Dataset Summary", "Shared Parquet exported by the Python janitor and read directly by the Rust athlete app.", |ui| {
                            metric_pair(ui, "Examples", &self.dataset_summary.example_count.to_string());
                            metric_pair(ui, "Paired Views", &self.dataset_summary.paired_view_examples.to_string());
                            metric_pair(ui, "Feature Count", &self.dataset_summary.feature_count.to_string());
                            metric_pair(ui, "Avg Target Score", &format!("{:.0}%", self.dataset_summary.average_target_score * 100.0));
                            ui.add_space(10.0);
                            for (label, count) in &self.dataset_summary.label_balance {
                                label_row(ui, *label, *count);
                            }
                        });
                    });
                });
            });
    }

    fn render_dashboard(&mut self, ctx: &egui::Context, snapshot: &biomech_ai::trainer::TrainerSnapshot) {
        egui::SidePanel::left("control_panel")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(18, 24, 29))
                    .inner_margin(egui::Margin::symmetric(18, 18)),
            )
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.label(RichText::new("Shot Controls").size(24.0).strong());
                ui.label(RichText::new("Tune mechanics and inspect the live training signal.").size(14.0));
                ui.add_space(10.0);
                control_slider(ui, &mut self.input.elbow_flexion, 60.0..=120.0, "Elbow Flexion", "Compact pocket");
                control_slider(ui, &mut self.input.knee_load, 85.0..=135.0, "Knee Load", "Efficient dip");
                control_slider(ui, &mut self.input.forearm_verticality, 65.0..=100.0, "Forearm Verticality", "Stacked line");
                control_slider(ui, &mut self.input.elbow_flare, 0.0..=20.0, "Elbow Flare", "Alignment drift");
                control_slider(ui, &mut self.input.release_height_ratio, 0.95..=1.5, "Release Height Ratio", "High finish");
                control_slider(ui, &mut self.input.release_timing_ms, 220.0..=500.0, "Release Timing (ms)", "Lift to snap");
                control_slider(ui, &mut self.input.release_at_apex_offset_ms, -40.0..=120.0, "Release vs Apex (ms)", "Apex sync");
                control_slider(ui, &mut self.input.jump_height, 0.15..=0.6, "Jump Height (m)", "Vertical pop");

                ui.add_space(12.0);
                if accent_button(ui, "Rebuild Session Audit").clicked() {
                    self.regenerate_session();
                }

                ui.add_space(14.0);
                section_mini(ui, "Calibration Carryover");
                metric_pair(ui, "Reach", &format!("{:.2} m", snapshot.calibration.estimated_standing_reach_m));
                metric_pair(ui, "Wingspan", &format!("{:.2} m", snapshot.calibration.estimated_wingspan_m));
                metric_pair(ui, "Camera Angle", &format!("{:.1} deg", snapshot.calibration.estimated_camera_angle_deg));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(22, 18)))
            .show(ctx, |ui| {
                hero_metrics(ui, &self.dataset_summary, &self.model_readiness, snapshot.inference.score);
                ui.add_space(16.0);

                ui.columns(2, |columns| {
                    let (left_cols, right_cols) = columns.split_at_mut(1);
                    let left = &mut left_cols[0];
                    let right = &mut right_cols[0];

                    section_card(left, "Shot Intelligence", "Live ML readout, nearest motion prototype, and coaching prompts.", |ui| {
                        score_badge(ui, snapshot.inference.score, snapshot.inference.label);
                        ui.add_space(10.0);
                        ui.label(format!("Nearest learned pattern: {}", snapshot.inference.nearest_neighbor));
                        ui.label(format!("Model confidence: {:.0}%", snapshot.inference.confidence * 100.0));
                        ui.add_space(10.0);
                        for item in &snapshot.inference.feedback {
                            ui.label(format!("• {item}"));
                        }
                    });

                    section_card(right, "Training Readiness", "Model-facing summary of what the current Parquet dataset can support.", |ui| {
                        readiness_panel(ui, &self.model_readiness);
                    });
                });

                ui.add_space(16.0);

                section_card(ui, "Processed Sessions", "Uploaded sessions currently in the Rust review corpus, including paired-view coverage and teacher provenance.", |ui| {
                    processed_sessions_panel(ui, &self.processed_sessions);
                });

                ui.add_space(16.0);

                ui.columns(2, |columns| {
                    let (left_cols, right_cols) = columns.split_at_mut(1);
                    let left = &mut left_cols[0];
                    let right = &mut right_cols[0];

                    section_card(left, "Mechanical Diagnostics", "Color-coded score bars against the elite baseline window.", |ui| {
                        for diagnostic in &snapshot.diagnostics {
                            diagnostic_row(ui, diagnostic.metric.as_str(), diagnostic.actual, diagnostic.ideal, diagnostic.severity);
                        }
                    });

                    section_card(right, "Shot Audit", "Session consistency across generated attempts and top recurring corrections.", |ui| {
                        let top_fixes = if self.session_audit.top_issues.is_empty() {
                            "None".to_string()
                        } else {
                            self.session_audit.top_issues.join(", ")
                        };
                        metric_pair(ui, "Attempts", &self.session_audit.attempt_count.to_string());
                        metric_pair(ui, "Average Consistency", &format!("{}", self.session_audit.average_consistency_score));
                        metric_pair(ui, "Top Fixes", &top_fixes);
                        ui.add_space(10.0);
                        mini_timeline(ui, &self.shots);
                    });
                });

                ui.add_space(16.0);

                ui.columns(2, |columns| {
                    let (left_cols, right_cols) = columns.split_at_mut(1);
                    let left = &mut left_cols[0];
                    let right = &mut right_cols[0];

                    section_card(left, "Kinetic Chain Stages", "Stage-by-stage cues for the load, set point, release, and finish.", |ui| {
                        stage_cards(ui, &snapshot.stage_feedback);
                    });

                    section_card(right, "Visual Review", "Overlay and release-path panels designed to mirror the eventual camera review experience.", |ui| {
                        draw_overlay_review(ui, &self.input, &snapshot.stage_feedback);
                        ui.add_space(10.0);
                        draw_release_panel(ui, &self.input);
                    });
                });
            });
    }
}

fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(14.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(12);
    style.visuals = egui::Visuals::dark();
    style.visuals.override_text_color = Some(Color32::from_rgb(240, 236, 228));
    style.visuals.panel_fill = Color32::from_rgb(12, 18, 22);
    style.visuals.window_fill = Color32::from_rgb(12, 18, 22);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(19, 25, 30);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(28, 37, 43);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(44, 57, 64);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(190, 113, 58);
    style.visuals.selection.bg_fill = Color32::from_rgb(190, 113, 58);
    style.visuals.hyperlink_color = Color32::from_rgb(238, 183, 92);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(30.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(16.0, FontFamily::Proportional),
    );
    ctx.set_style(style);
}

fn paint_background(ctx: &egui::Context) {
    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    painter.rect_filled(rect, 0.0, Color32::from_rgb(10, 14, 18));
    painter.circle_filled(rect.left_top() + egui::vec2(220.0, 180.0), 260.0, Color32::from_rgba_premultiplied(168, 88, 41, 22));
    painter.circle_filled(rect.right_top() + egui::vec2(-180.0, 220.0), 220.0, Color32::from_rgba_premultiplied(51, 108, 89, 28));
    painter.circle_filled(rect.center_bottom() + egui::vec2(-120.0, -60.0), 280.0, Color32::from_rgba_premultiplied(35, 57, 73, 24));
}

fn hero_metrics(ui: &mut egui::Ui, summary: &TrainingDatasetSummary, readiness: &ModelReadiness, shot_score: u8) {
    ui.horizontal(|ui| {
        metric_tile(ui, "Dataset", &summary.example_count.to_string(), "examples");
        metric_tile(ui, "Paired Views", &summary.paired_view_examples.to_string(), "linked shots");
        metric_tile(ui, "Readiness", &readiness.score.to_string(), "training score");
        metric_tile(ui, "Live Shot", &shot_score.to_string(), "current inference");
    });
}

fn metric_tile(ui: &mut egui::Ui, title: &str, value: &str, subtitle: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(19, 25, 30))
        .stroke(Stroke::new(1.0, Color32::from_rgb(46, 60, 68)))
        .corner_radius(18.0)
        .inner_margin(egui::Margin::symmetric(16, 16))
        .show(ui, |ui| {
            ui.set_min_size(Vec2::new((ui.available_width() / 4.0).max(150.0), 92.0));
            ui.label(RichText::new(title).size(14.0).color(Color32::from_rgb(165, 178, 184)));
            ui.label(RichText::new(value).size(30.0).strong());
            ui.label(RichText::new(subtitle).size(13.0).color(Color32::from_rgb(122, 136, 145)));
        });
}

fn section_card<R>(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    egui::Frame::new()
        .fill(Color32::from_rgb(18, 24, 29))
        .stroke(Stroke::new(1.0, Color32::from_rgb(43, 56, 63)))
        .corner_radius(22.0)
        .inner_margin(egui::Margin::symmetric(18, 18))
        .show(ui, |ui| {
            ui.label(RichText::new(title).size(24.0).strong());
            ui.label(RichText::new(subtitle).size(14.0).color(Color32::from_rgb(162, 176, 183)));
            ui.add_space(12.0);
            add_contents(ui)
        })
        .inner
}

fn section_mini(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).size(18.0).strong());
    ui.add_space(6.0);
}

fn metric_pair(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(Color32::from_rgb(164, 176, 182)).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).size(16.0));
        });
    });
}

fn slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) {
    ui.add(egui::Slider::new(value, range).text(label));
}

fn control_slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    label: &str,
    hint: &str,
) {
    ui.label(RichText::new(label).strong());
    ui.add(egui::Slider::new(value, range).show_value(true));
    ui.label(RichText::new(hint).size(12.0).color(Color32::from_rgb(140, 154, 161)));
    ui.add_space(6.0);
}

fn nav_button(ui: &mut egui::Ui, current: &mut AppScreen, target: AppScreen, label: &str) {
    let selected = *current == target;
    let button = egui::Button::new(label)
        .fill(if selected {
            Color32::from_rgb(190, 113, 58)
        } else {
            Color32::from_rgb(27, 36, 42)
        })
        .corner_radius(999.0);
    if ui.add(button).clicked() {
        *current = target;
    }
}

fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).strong())
            .fill(Color32::from_rgb(190, 113, 58))
            .corner_radius(14.0)
            .min_size(Vec2::new(ui.available_width().min(320.0), 42.0)),
    )
}

fn pill_label(ui: &mut egui::Ui, label: &str, color: Color32) {
    egui::Frame::new()
        .fill(color)
        .corner_radius(999.0)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.label(RichText::new(label).strong().size(13.0));
        });
}

fn status_chip(ui: &mut egui::Ui, text: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(21, 28, 33))
        .stroke(Stroke::new(1.0, Color32::from_rgb(48, 61, 68)))
        .corner_radius(999.0)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(12.5).color(Color32::from_rgb(187, 198, 203)));
        });
}

fn score_badge(ui: &mut egui::Ui, score: u8, label: ShotQualityLabel) {
    let fill = match label {
        ShotQualityLabel::Elite => Color32::from_rgb(61, 135, 88),
        ShotQualityLabel::Strong => Color32::from_rgb(170, 121, 52),
        ShotQualityLabel::Developing => Color32::from_rgb(176, 95, 58),
        ShotQualityLabel::Raw => Color32::from_rgb(152, 61, 55),
    };

    egui::Frame::new()
        .fill(fill)
        .corner_radius(18.0)
        .inner_margin(egui::Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.label(
                RichText::new(format!("{}  •  {}", score, label_text(label)))
                    .size(30.0)
                    .strong(),
            );
        });
}

fn label_text(label: ShotQualityLabel) -> &'static str {
    match label {
        ShotQualityLabel::Elite => "Elite Window",
        ShotQualityLabel::Strong => "Strong Base",
        ShotQualityLabel::Developing => "Developing",
        ShotQualityLabel::Raw => "Needs Rebuild",
    }
}

fn label_row(ui: &mut egui::Ui, label: ShotQualityLabel, count: usize) {
    let color = match label {
        ShotQualityLabel::Elite => Color32::from_rgb(79, 162, 109),
        ShotQualityLabel::Strong => Color32::from_rgb(214, 161, 65),
        ShotQualityLabel::Developing => Color32::from_rgb(201, 118, 70),
        ShotQualityLabel::Raw => Color32::from_rgb(194, 77, 62),
    };
    ui.horizontal(|ui| {
        ui.colored_label(color, label_text(label));
        ui.label(format!("{count} examples"));
    });
}

fn readiness_panel(ui: &mut egui::Ui, readiness: &ModelReadiness) {
    let color = readiness_color(readiness.score);
    ui.horizontal(|ui| {
        pill_label(
            ui,
            if readiness.is_ready {
                "Ready for first supervised run"
            } else {
                "Still collecting trainable coverage"
            },
            color,
        );
        ui.label(format!("{} / 100", readiness.score));
    });
    ui.add_space(8.0);
    for item in &readiness.checklist {
        ui.label(format!("• {item}"));
    }
    if !readiness.risks.is_empty() {
        ui.add_space(10.0);
        ui.label(RichText::new("Open Risks").strong().color(Color32::from_rgb(234, 176, 84)));
        for risk in &readiness.risks {
            ui.label(format!("• {risk}"));
        }
    }
    ui.add_space(10.0);
    ui.label(RichText::new(readiness.recommended_next_step.as_str()).italics());
}

fn readiness_color(score: u8) -> Color32 {
    if score >= 80 {
        Color32::from_rgb(61, 135, 88)
    } else if score >= 55 {
        Color32::from_rgb(170, 121, 52)
    } else {
        Color32::from_rgb(152, 61, 55)
    }
}

fn diagnostic_row(ui: &mut egui::Ui, metric: &str, actual: f32, ideal: f32, severity: DiagnosticSeverity) {
    let color = severity_color(severity);
    let pct = (1.0 - ((actual - ideal).abs() / ideal.max(1.0))).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.colored_label(color, metric);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(format!("{actual:.1} / target {ideal:.1}"));
        });
    });
    ui.add(egui::ProgressBar::new(pct).desired_width(ui.available_width()).fill(color));
    ui.add_space(8.0);
}

fn severity_color(severity: DiagnosticSeverity) -> Color32 {
    match severity {
        DiagnosticSeverity::Optimal => Color32::from_rgb(79, 162, 109),
        DiagnosticSeverity::Minor => Color32::from_rgb(214, 161, 65),
        DiagnosticSeverity::Major => Color32::from_rgb(194, 77, 62),
    }
}

fn stage_cards(ui: &mut egui::Ui, stages: &[StageFeedback]) {
    for stage in stages {
        egui::Frame::new()
            .fill(Color32::from_rgb(23, 31, 36))
            .stroke(Stroke::new(1.0, Color32::from_rgb(46, 60, 68)))
            .corner_radius(16.0)
            .inner_margin(egui::Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(severity_color(stage.color_hint), stage_name(stage.stage));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{} / 100", stage.score));
                    });
                });
                ui.label(stage.coaching_note.as_str());
            });
        ui.add_space(8.0);
    }
}

fn stage_name(stage: ShotStage) -> &'static str {
    match stage {
        ShotStage::Idle => "Idle",
        ShotStage::ReadyStance => "Ready Stance",
        ShotStage::Load => "Load",
        ShotStage::SetPoint => "Set Point",
        ShotStage::Release => "Release",
        ShotStage::FollowThrough => "Follow Through",
        ShotStage::Complete => "Complete",
    }
}

fn mini_timeline(ui: &mut egui::Ui, shots: &[SessionShotSummary]) {
    let width = ui.available_width();
    let desired_size = Vec2::new(width, 128.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 14.0, Color32::from_rgb(22, 29, 34));

    if shots.is_empty() {
        return;
    }

    let step = rect.width() / shots.len() as f32;
    for (index, shot) in shots.iter().enumerate() {
        let x = rect.left() + step * index as f32 + step * 0.5;
        let h = (rect.height() - 24.0) * (shot.consistency_score as f32 / 100.0);
        let color = if shot.consistency_score >= 85 {
            Color32::from_rgb(76, 161, 101)
        } else if shot.consistency_score >= 65 {
            Color32::from_rgb(208, 155, 66)
        } else {
            Color32::from_rgb(184, 76, 66)
        };
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x - step * 0.28, rect.bottom() - h - 10.0),
                egui::pos2(x + step * 0.28, rect.bottom() - 10.0),
            ),
            6.0,
            color,
        );
    }
}

fn processed_sessions_panel(ui: &mut egui::Ui, sessions: &[ProcessedSessionSummary]) {
    if sessions.is_empty() {
        ui.label("No processed sessions have been folded into the corpus yet.");
        return;
    }

    for session in sessions {
        egui::Frame::new()
            .fill(Color32::from_rgb(23, 31, 36))
            .stroke(Stroke::new(1.0, Color32::from_rgb(46, 60, 68)))
            .corner_radius(16.0)
            .inner_margin(egui::Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(session.session_key.as_str()).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        pill_label(ui, &format!("{} shots", session.total_shots), Color32::from_rgb(36, 86, 98));
                    });
                });
                ui.label(format!(
                    "Source: {}   |   Teacher: {}",
                    session.source_dataset, session.teacher_model
                ));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    metric_tile_compact(ui, "Paired", session.paired_shots);
                    metric_tile_compact(ui, "Side Only", session.side_only_shots);
                    metric_tile_compact(ui, "Angle Only", session.angle_only_shots);
                });
            });
        ui.add_space(8.0);
    }
}

fn metric_tile_compact(ui: &mut egui::Ui, title: &str, value: usize) {
    egui::Frame::new()
        .fill(Color32::from_rgb(18, 24, 29))
        .corner_radius(12.0)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_min_size(Vec2::new(120.0, 54.0));
            ui.label(RichText::new(title).size(12.0).color(Color32::from_rgb(164, 176, 182)));
            ui.label(RichText::new(value.to_string()).size(22.0).strong());
        });
}

fn draw_calibration_preview(ui: &mut egui::Ui, calibration_input: &CalibrationInput) {
    let desired_size = Vec2::new(ui.available_width(), 260.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 20.0, Color32::from_rgb(23, 31, 36));

    let athlete_x = rect.left() + rect.width() * 0.35;
    let floor_y = rect.bottom() - 24.0;
    let body_h = (calibration_input.body_height_m * 90.0).clamp(110.0, 180.0);
    let shoulder_y = floor_y - body_h * 0.72;
    let head_y = floor_y - body_h;
    let shoulder_half = calibration_input.shoulder_width_m * 90.0;
    let cam_x = rect.right() - 74.0;
    let estimated_mount_height_m = calibration_input.body_height_m * 0.58;
    let cam_y = floor_y - estimated_mount_height_m * 72.0;

    painter.line_segment(
        [egui::pos2(athlete_x, floor_y), egui::pos2(athlete_x, head_y)],
        Stroke::new(8.0, Color32::from_rgb(186, 121, 69)),
    );
    painter.line_segment(
        [egui::pos2(athlete_x - shoulder_half, shoulder_y), egui::pos2(athlete_x + shoulder_half, shoulder_y)],
        Stroke::new(7.0, Color32::from_rgb(233, 173, 94)),
    );
    painter.circle_filled(egui::pos2(athlete_x, head_y - 16.0), 18.0, Color32::from_rgb(243, 216, 151));
    painter.circle_filled(egui::pos2(cam_x, cam_y), 16.0, Color32::from_rgb(120, 139, 154));
    painter.line_segment(
        [egui::pos2(cam_x, cam_y), egui::pos2(athlete_x, shoulder_y)],
        Stroke::new(2.5, Color32::from_rgb(76, 161, 101)),
    );
    painter.text(
        rect.left_top() + egui::vec2(16.0, 14.0),
        Align2::LEFT_TOP,
        "Athlete and camera geometry",
        FontId::proportional(16.0),
        Color32::from_rgb(240, 236, 226),
    );
}

fn draw_overlay_review(ui: &mut egui::Ui, input: &ShotInput, stages: &[StageFeedback]) {
    let desired_size = Vec2::new(ui.available_width(), 250.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 20.0, Color32::from_rgb(23, 31, 36));

    let center_x = rect.center().x - 20.0;
    let floor_y = rect.bottom() - 26.0;
    let hip_y = floor_y - 92.0;
    let shoulder_y = hip_y - 74.0;
    let knee_x = center_x - 12.0;
    let ankle_x = center_x - 5.0;
    let elbow_x = center_x + (input.elbow_flare * 3.2).clamp(0.0, 44.0);
    let wrist_x = elbow_x + 12.0;
    let wrist_y = floor_y - (input.release_height_ratio * 90.0);

    let load_color = stage_color_from_feedback(stages, ShotStage::Load);
    let set_color = stage_color_from_feedback(stages, ShotStage::SetPoint);
    let release_color = stage_color_from_feedback(stages, ShotStage::Release);

    painter.line_segment([egui::pos2(center_x, shoulder_y), egui::pos2(center_x, hip_y)], Stroke::new(7.0, Color32::from_rgb(181, 121, 70)));
    painter.line_segment([egui::pos2(center_x, hip_y), egui::pos2(knee_x, hip_y + 58.0)], Stroke::new(6.0, load_color));
    painter.line_segment([egui::pos2(knee_x, hip_y + 58.0), egui::pos2(ankle_x, floor_y)], Stroke::new(6.0, load_color));
    painter.line_segment([egui::pos2(center_x, shoulder_y), egui::pos2(elbow_x, shoulder_y + 44.0)], Stroke::new(6.0, set_color));
    painter.line_segment([egui::pos2(elbow_x, shoulder_y + 44.0), egui::pos2(wrist_x, wrist_y)], Stroke::new(5.0, release_color));
    painter.circle_filled(egui::pos2(wrist_x + 18.0, wrist_y - 8.0), 12.0, release_color);
}

fn stage_color_from_feedback(stages: &[StageFeedback], stage: ShotStage) -> Color32 {
    stages
        .iter()
        .find(|item| item.stage == stage)
        .map(|item| severity_color(item.color_hint))
        .unwrap_or(Color32::from_rgb(120, 139, 154))
}

fn draw_release_panel(ui: &mut egui::Ui, input: &ShotInput) {
    let desired_size = Vec2::new(ui.available_width(), 186.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 20.0, Color32::from_rgb(23, 31, 36));
    let center_x = rect.center().x;
    let floor_y = rect.bottom() - 24.0;
    let shoulder_y = rect.top() + 58.0;
    let elbow_offset = (input.elbow_flare * 3.2).clamp(0.0, 44.0);
    let release_y = floor_y - (input.release_height_ratio * 78.0);

    painter.line_segment([egui::pos2(center_x, floor_y), egui::pos2(center_x, shoulder_y)], Stroke::new(7.0, Color32::from_rgb(181, 121, 70)));
    painter.line_segment([egui::pos2(center_x, shoulder_y), egui::pos2(center_x + elbow_offset, shoulder_y + 44.0)], Stroke::new(6.0, Color32::from_rgb(223, 160, 74)));
    painter.line_segment([egui::pos2(center_x + elbow_offset, shoulder_y + 44.0), egui::pos2(center_x + elbow_offset + 12.0, release_y)], Stroke::new(5.0, Color32::from_rgb(239, 206, 126)));
    let ball_color = if input.release_at_apex_offset_ms <= 20.0 {
        Color32::from_rgb(76, 161, 101)
    } else if input.release_at_apex_offset_ms <= 50.0 {
        Color32::from_rgb(208, 155, 66)
    } else {
        Color32::from_rgb(184, 76, 66)
    };
    painter.circle_filled(egui::pos2(center_x + elbow_offset + 20.0, release_y - 8.0), 14.0, ball_color);
}
