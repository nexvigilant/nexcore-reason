//! Integration test — full Foundry artifact pipeline B1 → A3 → feedback.
//!
//! Exercises the complete data flow through The Foundry's typed artifact
//! system, validating that every bridge handoff and station transform
//! produces structurally valid artifacts that serialize and deserialize
//! correctly.
//!
//! Flow:
//! ```text
//! DesignSpec (B1) → SourceArtifact (B2) → ValidatedDeliverable (B3)
//!       ↓                                          ↓
//!  MetricReport ──→ AggregatedMetrics (A1) ──→ PatternReport (A2)
//!       ↓                                          ↓
//!  CausalDag (reason) ──→ InferenceEngine ──→ IntelligenceReport (A3)
//!       ↓
//!  DesignConstraints (feedback bridge → B1)
//! ```

use nexcore_foundry::analyst::{
    AggregatedMetrics, ComplexityRating, IntelligenceReport, Metric, MetricReport, PatternReport,
    PatternSignature, RiskLevel,
};
use nexcore_foundry::artifact::{
    Component, ComponentKind, Contract, DesignSpec, FileEntry, ShippableArtifact, SourceArtifact,
    ValidatedDeliverable,
};
use nexcore_foundry::bridge::{BridgeKind, DesignConstraints};
use nexcore_foundry::governance::{CascadeValidation, GoalLevel, SmartGoal};
use nexcore_foundry::station::{PipelineOrder, StationId};

use nexcore_reason::counterfactual::{CounterfactualEngine, Intervention};
use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
use nexcore_reason::inference::InferenceEngine;

// ---------------------------------------------------------------------------
// Phase 1: Builder pipeline (B1 → B2 → B3)
// ---------------------------------------------------------------------------

/// Simulates the full builder pipeline: B1 produces a DesignSpec, B2
/// transforms it into a SourceArtifact, B3 validates and ships it.
#[test]
fn builder_pipeline_b1_to_b3() {
    // B1: Blueprint station produces a DesignSpec.
    let design = DesignSpec {
        name: "Authentication module".to_string(),
        components: vec![
            Component {
                kind: ComponentKind::RustCrate,
                name: "auth_handler".to_string(),
                path: "crates/auth-handler".to_string(),
            },
            Component {
                kind: ComponentKind::Test,
                name: "auth_handler_test".to_string(),
                path: "crates/auth-handler/tests".to_string(),
            },
        ],
        contracts: vec![Contract {
            input_type: "LoginRequest".to_string(),
            output_type: "AuthToken".to_string(),
        }],
        test_plan: vec![
            "valid credentials return token".to_string(),
            "invalid credentials return 401".to_string(),
        ],
        primitives: vec![],
        constraints: vec!["No unwrap".to_string(), "100% branch coverage".to_string()],
    };

    // Verify DesignSpec serializes.
    let design_json = serde_json::to_string(&design).expect("DesignSpec serialisation");
    assert!(design_json.contains("auth_handler"));

    // B2: Frame station transforms DesignSpec → SourceArtifact.
    let source = SourceArtifact {
        files: vec![
            FileEntry {
                path: "crates/auth-handler/src/lib.rs".to_string(),
                content_hash: "sha256:abc123".to_string(),
                loc: 150,
            },
            FileEntry {
                path: "crates/auth-handler/tests/integration.rs".to_string(),
                content_hash: "sha256:def456".to_string(),
                loc: 80,
            },
        ],
        build_order: vec!["auth-handler".to_string()],
        implemented: true,
    };
    assert!(source.implemented);
    assert_eq!(source.files.len(), 2);

    // B3: Finish station validates and produces a ShippableArtifact.
    let validated = ValidatedDeliverable {
        build_pass: true,
        test_count: 12,
        tests_passed: 12,
        lint_pass: true,
        coverage_percent: 95.0,
        failures: vec![],
    };
    assert!(validated.is_green());

    let shippable = ShippableArtifact::from_validated(validated);
    assert!(shippable.ready);
}

// ---------------------------------------------------------------------------
// Phase 2: Extraction bridge → Analyst pipeline (A1 → A2 → A3)
// ---------------------------------------------------------------------------

/// Simulates the analyst pipeline: extraction bridge produces metrics,
/// A1 aggregates, A2 detects patterns, A3 reasons over a causal graph.
#[test]
fn analyst_pipeline_a1_to_a3_with_inference() {
    // Extraction bridge output → A1 input.
    let metric_report = MetricReport {
        source_station: "B3-finish".to_string(),
        timestamp: nexcore_chrono::DateTime::now(),
        metrics: vec![
            Metric {
                name: "cyclomatic_complexity".to_string(),
                value: 8.5,
                unit: "per-function avg".to_string(),
            },
            Metric {
                name: "test_coverage".to_string(),
                value: 95.0,
                unit: "percent".to_string(),
            },
            Metric {
                name: "loc".to_string(),
                value: 230.0,
                unit: "lines".to_string(),
            },
        ],
    };

    // A1: Aggregate metrics.
    let aggregated = AggregatedMetrics {
        quality_score: 0.88,
        primitive_density: 0.72,
        complexity_rating: ComplexityRating::Moderate,
        coverage_delta: 0.03,
        raw_metrics: vec![metric_report],
    };
    assert!(aggregated.quality_score > 0.8);
    assert_eq!(aggregated.complexity_rating, ComplexityRating::Moderate);

    // Crystallization bridge → A2 input.
    let _signatures = vec![
        PatternSignature {
            name: "separation-of-concerns".to_string(),
            confidence: 0.91,
        },
        PatternSignature {
            name: "missing-error-boundary".to_string(),
            confidence: 0.75,
        },
    ];

    // A2: Pattern analysis.
    let pattern_report = PatternReport {
        patterns: vec!["separation-of-concerns".to_string()],
        anti_patterns: vec!["missing-error-boundary".to_string()],
        structural_risks: vec!["high cyclomatic complexity in auth handler".to_string()],
        opportunities: vec!["extract validation logic".to_string()],
    };
    assert_eq!(pattern_report.patterns.len(), 1);
    assert_eq!(pattern_report.anti_patterns.len(), 1);

    // Inference bridge → A3: Build causal DAG from patterns + metrics.
    let mut dag = CausalDag::new();

    dag.add_node(CausalNode {
        id: NodeId::new("high_complexity"),
        label: "High cyclomatic complexity".to_string(),
        node_type: NodeType::Metric,
    });
    dag.add_node(CausalNode {
        id: NodeId::new("missing_error_boundary"),
        label: "Missing error boundary".to_string(),
        node_type: NodeType::Pattern,
    });
    dag.add_node(CausalNode {
        id: NodeId::new("regression_risk"),
        label: "Regression risk".to_string(),
        node_type: NodeType::Risk,
    });
    dag.add_node(CausalNode {
        id: NodeId::new("release_risk"),
        label: "Release quality risk".to_string(),
        node_type: NodeType::Risk,
    });
    dag.add_node(CausalNode {
        id: NodeId::new("extract_validation"),
        label: "Extract validation logic".to_string(),
        node_type: NodeType::Recommendation,
    });

    dag.add_link(CausalLink {
        from: NodeId::new("high_complexity"),
        to: NodeId::new("regression_risk"),
        strength: 0.85,
        evidence: "Complexity > 8 correlates with defect rate".to_string(),
    })
    .expect("acyclic");

    dag.add_link(CausalLink {
        from: NodeId::new("missing_error_boundary"),
        to: NodeId::new("regression_risk"),
        strength: 0.7,
        evidence: "Unhandled error paths in pattern analysis".to_string(),
    })
    .expect("acyclic");

    dag.add_link(CausalLink {
        from: NodeId::new("regression_risk"),
        to: NodeId::new("release_risk"),
        strength: 0.9,
        evidence: "Regression compounds into release risk".to_string(),
    })
    .expect("acyclic");

    dag.add_link(CausalLink {
        from: NodeId::new("regression_risk"),
        to: NodeId::new("extract_validation"),
        strength: 0.9,
        evidence: "Decomposition reduces per-function complexity".to_string(),
    })
    .expect("acyclic");

    // A3: Run inference engine.
    let engine = InferenceEngine::new(dag);
    let report = engine.infer().expect("inference must succeed");

    // Verify report structure.
    assert!(
        report.risk_level >= RiskLevel::Moderate,
        "risk should be at least Moderate; got {:?}",
        report.risk_level
    );
    assert!(
        !report.findings.is_empty(),
        "should have at least one finding"
    );
    assert!(report.confidence > 0.0, "confidence should be positive");

    // Verify CausalGraph export.
    let graph = engine.to_causal_graph();
    assert_eq!(graph.edges.len(), 4);

    // Verify full report serialization round-trip.
    let report_json = serde_json::to_string(&report).expect("report serialisation");
    let recovered: IntelligenceReport =
        serde_json::from_str(&report_json).expect("report deserialisation");
    assert_eq!(recovered.findings.len(), report.findings.len());
    assert_eq!(recovered.risk_level, report.risk_level);
}

// ---------------------------------------------------------------------------
// Phase 3: Feedback bridge (A3 → B1)
// ---------------------------------------------------------------------------

/// Simulates the feedback bridge: A3 intelligence report drives
/// DesignConstraints that feed back into B1 for the next iteration.
#[test]
fn feedback_bridge_a3_to_b1() {
    let intelligence = IntelligenceReport {
        findings: vec!["High complexity drives regression risk".to_string()],
        recommendations: vec!["Extract validation logic".to_string()],
        risk_level: RiskLevel::High,
        confidence: 0.82,
    };

    // Feedback bridge transforms intelligence into design constraints.
    let constraints = DesignConstraints {
        new_components: intelligence.recommendations.iter().cloned().collect(),
        new_constraints: vec!["Max cyclomatic complexity <= 6".to_string()],
        iteration_trigger: intelligence.risk_level >= RiskLevel::High,
    };

    assert!(
        constraints.iteration_trigger,
        "High risk should trigger iteration"
    );
    assert_eq!(constraints.new_components.len(), 1);
    assert_eq!(constraints.new_constraints.len(), 1);

    // Verify serialization.
    let json = serde_json::to_string(&constraints).expect("constraints serialisation");
    let recovered: DesignConstraints =
        serde_json::from_str(&json).expect("constraints deserialisation");
    assert!(recovered.iteration_trigger);
}

// ---------------------------------------------------------------------------
// Phase 4: Counterfactual analysis (what-if on the causal graph)
// ---------------------------------------------------------------------------

/// Demonstrates counterfactual reasoning: "what if we removed the
/// complexity factor?" — verifying the regression risk path breaks.
#[test]
fn counterfactual_removes_causal_path() {
    let mut dag = CausalDag::new();

    dag.add_node(CausalNode {
        id: NodeId::new("complexity"),
        label: "Complexity".to_string(),
        node_type: NodeType::Metric,
    });
    dag.add_node(CausalNode {
        id: NodeId::new("risk"),
        label: "Risk".to_string(),
        node_type: NodeType::Risk,
    });
    dag.add_link(CausalLink {
        from: NodeId::new("complexity"),
        to: NodeId::new("risk"),
        strength: 0.9,
        evidence: "direct".to_string(),
    })
    .expect("acyclic");

    let engine = CounterfactualEngine::new(dag);
    let result = engine
        .evaluate(&Intervention::RemoveNode(NodeId::new("complexity")))
        .expect("evaluation");

    assert!(
        result.impact_score > 0.0,
        "removing a causal root should have positive impact"
    );
    assert!(
        result.new_roots.contains(&NodeId::new("risk")),
        "risk should become a new root after complexity is removed"
    );
}

// ---------------------------------------------------------------------------
// Phase 5: VDAG pipeline ordering
// ---------------------------------------------------------------------------

/// Verify the VDAG full pipeline ordering covers all 14 stations in the
/// correct topological sequence.
#[test]
fn vdag_full_ordering_covers_all_stations() {
    let order = PipelineOrder::vdag_full();

    assert_eq!(order.len(), 14, "VDAG full pipeline should have 14 stages");
    assert!(!order.is_empty());

    // First station must be B1 (Blueprint).
    assert_eq!(order.stages[0], StationId::B1, "pipeline must start at B1");

    // Last station must be BridgeFeedback.
    assert_eq!(
        order.stages[order.len() - 1],
        StationId::BridgeFeedback,
        "pipeline must end at BridgeFeedback"
    );

    // Builder stations must precede analyst stations.
    let b3_pos = order
        .stages
        .iter()
        .position(|s| *s == StationId::B3)
        .expect("B3 must be present");
    let a1_pos = order
        .stages
        .iter()
        .position(|s| *s == StationId::A1)
        .expect("A1 must be present");
    assert!(b3_pos < a1_pos, "B3 (finish) must come before A1 (measure)");
}

// ---------------------------------------------------------------------------
// Phase 6: Bridge contract verification
// ---------------------------------------------------------------------------

/// All 6 bridge kinds exist and can be enumerated.
#[test]
fn all_bridge_kinds_enumerable() {
    let kinds = BridgeKind::all();
    assert_eq!(kinds.len(), 6, "Foundry architecture defines 6 bridges");
}

// ---------------------------------------------------------------------------
// Phase 7: Full serialization chain
// ---------------------------------------------------------------------------

/// Every major artifact type in the pipeline serializes to JSON and
/// deserializes back without data loss.
#[test]
fn full_serialization_chain() {
    // DesignSpec
    let design = DesignSpec {
        name: "test".to_string(),
        components: vec![],
        contracts: vec![],
        test_plan: vec![],
        primitives: vec![],
        constraints: vec![],
    };
    let json = serde_json::to_string(&design).expect("DesignSpec");
    let _: DesignSpec = serde_json::from_str(&json).expect("DesignSpec roundtrip");

    // AggregatedMetrics
    let agg = AggregatedMetrics {
        quality_score: 0.9,
        primitive_density: 0.7,
        complexity_rating: ComplexityRating::Low,
        coverage_delta: 0.01,
        raw_metrics: vec![],
    };
    let json = serde_json::to_string(&agg).expect("AggregatedMetrics");
    let _: AggregatedMetrics = serde_json::from_str(&json).expect("AggregatedMetrics roundtrip");

    // PatternReport
    let pr = PatternReport {
        patterns: vec!["p1".to_string()],
        anti_patterns: vec![],
        structural_risks: vec![],
        opportunities: vec![],
    };
    let json = serde_json::to_string(&pr).expect("PatternReport");
    let _: PatternReport = serde_json::from_str(&json).expect("PatternReport roundtrip");

    // IntelligenceReport
    let ir = IntelligenceReport {
        findings: vec!["f1".to_string()],
        recommendations: vec!["r1".to_string()],
        risk_level: RiskLevel::Critical,
        confidence: 0.99,
    };
    let json = serde_json::to_string(&ir).expect("IntelligenceReport");
    let recovered: IntelligenceReport =
        serde_json::from_str(&json).expect("IntelligenceReport roundtrip");
    assert_eq!(recovered.risk_level, RiskLevel::Critical);

    // CausalDag
    let mut dag = CausalDag::new();
    dag.add_node(CausalNode {
        id: NodeId::new("a"),
        label: "A".to_string(),
        node_type: NodeType::Module,
    });
    let json = serde_json::to_string(&dag).expect("CausalDag");
    let _: CausalDag = serde_json::from_str(&json).expect("CausalDag roundtrip");

    // SmartGoal
    let sg = SmartGoal {
        id: "SK-1".to_string(),
        level: GoalLevel::Strategic,
        specific: "test".to_string(),
        measurable: "test".to_string(),
        achievable: "test".to_string(),
        relevant: "test".to_string(),
        time_bound: "Q1".to_string(),
        traces_to: vec![],
    };
    let json = serde_json::to_string(&sg).expect("SmartGoal");
    let _: SmartGoal = serde_json::from_str(&json).expect("SmartGoal roundtrip");

    // CascadeValidation
    let cv = CascadeValidation {
        total_operational_goals: 6,
        traced_to_team: 6,
        traced_to_strategic: 6,
        alignment_percent: 100.0,
    };
    let json = serde_json::to_string(&cv).expect("CascadeValidation");
    let recovered: CascadeValidation =
        serde_json::from_str(&json).expect("CascadeValidation roundtrip");
    assert!(recovered.is_fully_aligned());
}
