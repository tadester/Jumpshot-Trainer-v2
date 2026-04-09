use crate::types::{DiagnosticSeverity, MetricDiagnostic, SessionAudit, SessionShotSummary};
use std::collections::HashMap;

fn severity_weight(diagnostic: &MetricDiagnostic) -> f32 {
    match diagnostic.severity {
        DiagnosticSeverity::Optimal => 1.0,
        DiagnosticSeverity::Minor => 0.6,
        DiagnosticSeverity::Major => 0.25,
    }
}

pub fn compute_consistency_score(diagnostics: &[MetricDiagnostic]) -> u8 {
    if diagnostics.is_empty() {
        return 0;
    }

    let weighted: f32 = diagnostics.iter().map(severity_weight).sum();
    ((weighted / diagnostics.len() as f32) * 100.0).round() as u8
}

pub fn create_shot_summary(
    shot_id: impl Into<String>,
    diagnostics: Vec<MetricDiagnostic>,
    jump_height: f32,
    release_time_ms: Option<u64>,
    release_at_apex_offset_ms: Option<i64>,
) -> SessionShotSummary {
    SessionShotSummary {
        shot_id: shot_id.into(),
        consistency_score: compute_consistency_score(&diagnostics),
        diagnostics,
        jump_height,
        release_time_ms,
        release_at_apex_offset_ms,
    }
}

pub fn create_session_audit(session_id: impl Into<String>, shots: Vec<SessionShotSummary>) -> SessionAudit {
    let attempt_count = shots.len();
    let average_consistency_score = if attempt_count == 0 {
        0
    } else {
        (shots
            .iter()
            .map(|shot| shot.consistency_score as f32)
            .sum::<f32>()
            / attempt_count as f32)
            .round() as u8
    };

    let mut issue_counts: HashMap<String, usize> = HashMap::new();
    for shot in &shots {
        for diagnostic in &shot.diagnostics {
            if !matches!(diagnostic.severity, DiagnosticSeverity::Optimal) {
                *issue_counts.entry(diagnostic.metric.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut top_issues: Vec<(String, usize)> = issue_counts.into_iter().collect();
    top_issues.sort_by(|a, b| b.1.cmp(&a.1));

    SessionAudit {
        session_id: session_id.into(),
        attempt_count,
        average_consistency_score,
        top_issues: top_issues.into_iter().take(3).map(|(metric, _)| metric).collect(),
        shots,
    }
}
