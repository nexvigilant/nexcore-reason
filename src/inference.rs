//! Causal inference engine for The Foundry's REASON station (A3).
//!
//! The engine consumes a [`CausalDag`] and produces an [`IntelligenceReport`]
//! by traversing all root-to-leaf causal chains, scoring each path by the
//! product of its link strengths, and synthesising findings and recommendations
//! from the highest-scoring paths.
//!
//! # Design
//!
//! Path scoring uses the multiplicative composition of causal strengths along a
//! chain. A chain `A →(0.9)→ B →(0.8)→ C` has a composite score of `0.72`.
//! This models the intuition that evidence weakens as it passes through
//! intermediate factors — each link in the chain attenuates the overall signal.
//!
//! The [`RiskLevel`] is mapped from the highest observed path score, so a
//! single strong chain is sufficient to escalate the overall risk classification
//! even if most other chains are weak.
//!
//! # Pipeline position
//!
//! ```text
//! Inference bridge  →  A3 (InferenceEngine::infer)  →  IntelligenceReport
//! ```
//!
//! # Examples
//!
//! ```
//! use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
//! use nexcore_reason::inference::InferenceEngine;
//!
//! let mut dag = CausalDag::new();
//!
//! dag.add_node(CausalNode {
//!     id: NodeId::new("missing_tests"),
//!     label: "Missing test coverage".to_string(),
//!     node_type: NodeType::Metric,
//! });
//! dag.add_node(CausalNode {
//!     id: NodeId::new("regression_risk"),
//!     label: "Regression risk".to_string(),
//!     node_type: NodeType::Risk,
//! });
//! dag.add_link(CausalLink {
//!     from: NodeId::new("missing_tests"),
//!     to: NodeId::new("regression_risk"),
//!     strength: 0.85,
//!     evidence: "historical data".to_string(),
//! }).expect("acyclic");
//!
//! let engine = InferenceEngine::new(dag);
//! let report = engine.infer().expect("inference must succeed");
//!
//! assert!(!report.findings.is_empty());
//! assert!(report.confidence > 0.0);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
use nexcore_foundry::analyst::{CausalEdge, CausalGraph, IntelligenceReport, RiskLevel};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration that governs inference behaviour.
///
/// All thresholds operate on normalised `[0.0, 1.0]` scores derived from
/// the product of causal link strengths along a path.
///
/// # Examples
///
/// ```
/// use nexcore_reason::inference::InferenceConfig;
///
/// let cfg = InferenceConfig::default();
/// assert_eq!(cfg.risk_threshold, 0.7);
/// assert_eq!(cfg.confidence_floor, 0.5);
/// assert_eq!(cfg.max_findings, 10);
/// assert_eq!(cfg.max_recommendations, 5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Composite path score above which a path is treated as a risk finding.
    ///
    /// Default: `0.7`.
    pub risk_threshold: f64,

    /// Minimum composite path score required for a path to be included in the
    /// report at all.
    ///
    /// Default: `0.5`.
    pub confidence_floor: f64,

    /// Maximum number of findings to include in the [`IntelligenceReport`].
    ///
    /// Default: `10`.
    pub max_findings: usize,

    /// Maximum number of recommendations to include in the
    /// [`IntelligenceReport`].
    ///
    /// Default: `5`.
    pub max_recommendations: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            risk_threshold: 0.7,
            confidence_floor: 0.5,
            max_findings: 10,
            max_recommendations: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Finding (intermediate representation)
// ---------------------------------------------------------------------------

/// An intermediate finding synthesised from high-scoring causal paths before
/// the final [`IntelligenceReport`] is assembled.
///
/// # Examples
///
/// ```
/// use nexcore_reason::dag::NodeId;
/// use nexcore_reason::inference::Finding;
///
/// let finding = Finding {
///     description: "High complexity drives regression risk".to_string(),
///     severity: 0.82,
///     supporting_paths: vec![vec![NodeId::new("complexity"), NodeId::new("regression_risk")]],
/// };
/// assert!(finding.severity > 0.5);
/// assert_eq!(finding.supporting_paths.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Human-readable description of the causal finding.
    pub description: String,

    /// Severity in `[0.0, 1.0]` derived from the composite path score.
    pub severity: f64,

    /// One or more causal chains (sequences of [`NodeId`]) that provide
    /// evidence for this finding.
    pub supporting_paths: Vec<Vec<NodeId>>,
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// Causal inference engine for the REASON station (A3).
///
/// Traverses a [`CausalDag`], scores all root-to-leaf paths, and synthesises
/// an [`IntelligenceReport`] with risk classification, findings, and
/// recommendations.
///
/// # Examples
///
/// ```
/// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
/// use nexcore_reason::inference::{InferenceConfig, InferenceEngine};
///
/// let mut dag = CausalDag::new();
/// dag.add_node(CausalNode {
///     id: NodeId::new("root"),
///     label: "Root cause".to_string(),
///     node_type: NodeType::Metric,
/// });
/// dag.add_node(CausalNode {
///     id: NodeId::new("risk"),
///     label: "Risk outcome".to_string(),
///     node_type: NodeType::Risk,
/// });
/// dag.add_link(CausalLink {
///     from: NodeId::new("root"),
///     to: NodeId::new("risk"),
///     strength: 0.9,
///     evidence: String::new(),
/// }).expect("acyclic");
///
/// let engine = InferenceEngine::new(dag);
/// let report = engine.infer().expect("inference must succeed");
/// assert_eq!(report.risk_level, nexcore_foundry::analyst::RiskLevel::Critical);
/// ```
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    dag: CausalDag,
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Creates a new [`InferenceEngine`] with default [`InferenceConfig`].
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::CausalDag;
    /// use nexcore_reason::inference::InferenceEngine;
    ///
    /// let engine = InferenceEngine::new(CausalDag::new());
    /// let report = engine.infer().expect("empty DAG produces a valid report");
    /// assert!(report.findings.is_empty());
    /// ```
    #[must_use]
    pub fn new(dag: CausalDag) -> Self {
        Self {
            dag,
            config: InferenceConfig::default(),
        }
    }

    /// Creates a new [`InferenceEngine`] with a custom [`InferenceConfig`].
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::CausalDag;
    /// use nexcore_reason::inference::{InferenceConfig, InferenceEngine};
    ///
    /// let config = InferenceConfig {
    ///     risk_threshold: 0.6,
    ///     confidence_floor: 0.3,
    ///     max_findings: 5,
    ///     max_recommendations: 3,
    /// };
    /// let engine = InferenceEngine::with_config(CausalDag::new(), config);
    /// let report = engine.infer().expect("inference must succeed");
    /// assert!(report.findings.is_empty());
    /// ```
    #[must_use]
    pub fn with_config(dag: CausalDag, config: InferenceConfig) -> Self {
        Self { dag, config }
    }

    /// Runs causal inference over the DAG and produces an [`IntelligenceReport`].
    ///
    /// # Algorithm
    ///
    /// 1. Enumerate all root-to-leaf causal chains via depth-first traversal.
    /// 2. Score each chain as the product of its link strengths.
    /// 3. Discard chains whose score falls below [`InferenceConfig::confidence_floor`].
    /// 4. Classify overall [`RiskLevel`] from the maximum observed chain score.
    /// 5. Generate findings from chains above [`InferenceConfig::risk_threshold`]
    ///    that terminate at [`NodeType::Risk`] nodes.
    /// 6. Generate recommendations from [`NodeType::Recommendation`] nodes
    ///    reachable via above-threshold chains.
    /// 7. Compute `confidence` as the arithmetic mean of retained chain scores.
    ///
    /// # Errors
    ///
    /// Returns an [`nexcore_error::NexError`] if the DAG contains a link that references
    /// a node id not present in the node list.
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::CausalDag;
    /// use nexcore_reason::inference::InferenceEngine;
    ///
    /// let engine = InferenceEngine::new(CausalDag::new());
    /// let report = engine.infer().expect("empty DAG must produce Low risk");
    /// assert_eq!(report.risk_level, nexcore_foundry::analyst::RiskLevel::Low);
    /// assert_eq!(report.confidence, 0.0);
    /// ```
    pub fn infer(&self) -> Result<IntelligenceReport, nexcore_error::NexError> {
        self.validate_links()?;

        let chains = self.find_causal_chains();

        let retained: Vec<(Vec<&NodeId>, f64)> = chains
            .into_iter()
            .filter(|(_, score)| *score >= self.config.confidence_floor)
            .collect();

        let max_score = retained.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
        let risk_level = self.classify_risk(max_score);

        let findings = self.build_findings(&retained);
        let recommendations = self.build_recommendations(&retained);

        let confidence = if retained.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count = retained.len() as f64;
            retained.iter().map(|(_, s)| *s).sum::<f64>() / count
        };

        Ok(IntelligenceReport {
            findings,
            recommendations,
            risk_level,
            confidence,
        })
    }

    /// Maps a maximum path score to a [`RiskLevel`].
    ///
    /// | Score range   | [`RiskLevel`]           |
    /// |---------------|-------------------------|
    /// | `[0.0, 0.3)`  | [`RiskLevel::Low`]      |
    /// | `[0.3, 0.6)`  | [`RiskLevel::Moderate`] |
    /// | `[0.6, 0.8)`  | [`RiskLevel::High`]     |
    /// | `[0.8, 1.0]`  | [`RiskLevel::Critical`] |
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::CausalDag;
    /// use nexcore_reason::inference::InferenceEngine;
    /// use nexcore_foundry::analyst::RiskLevel;
    ///
    /// let engine = InferenceEngine::new(CausalDag::new());
    /// assert_eq!(engine.classify_risk(0.0),  RiskLevel::Low);
    /// assert_eq!(engine.classify_risk(0.3),  RiskLevel::Moderate);
    /// assert_eq!(engine.classify_risk(0.6),  RiskLevel::High);
    /// assert_eq!(engine.classify_risk(0.8),  RiskLevel::Critical);
    /// assert_eq!(engine.classify_risk(1.0),  RiskLevel::Critical);
    /// ```
    #[must_use]
    pub fn classify_risk(&self, max_path_score: f64) -> RiskLevel {
        if max_path_score < 0.3 {
            RiskLevel::Low
        } else if max_path_score < 0.6 {
            RiskLevel::Moderate
        } else if max_path_score < 0.8 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    /// Enumerates all root-to-leaf causal chains in the DAG with their
    /// composite scores.
    ///
    /// The composite score of a chain is the product of all link strengths
    /// along the path.  A chain `A →(0.9)→ B →(0.8)→ C` scores `0.72`.
    ///
    /// Roots are defined structurally: nodes with no incoming edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    /// use nexcore_reason::inference::InferenceEngine;
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("a"), label: "A".to_string(), node_type: NodeType::Metric });
    /// dag.add_node(CausalNode { id: NodeId::new("b"), label: "B".to_string(), node_type: NodeType::Risk });
    /// dag.add_link(CausalLink { from: NodeId::new("a"), to: NodeId::new("b"), strength: 0.8, evidence: String::new() }).unwrap();
    ///
    /// let engine = InferenceEngine::new(dag);
    /// let chains = engine.find_causal_chains();
    /// assert_eq!(chains.len(), 1);
    ///
    /// let (path, score) = &chains[0];
    /// assert_eq!(path.len(), 2);
    /// assert!((score - 0.8).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn find_causal_chains(&self) -> Vec<(Vec<&NodeId>, f64)> {
        let mut results = Vec::new();

        // Build a forward adjacency map once rather than scanning the link
        // slice on every DFS step.
        let adj = self.build_adjacency();

        for root_id in self.dag.roots() {
            let mut path: Vec<&NodeId> = vec![root_id];
            self.dfs_paths(root_id, &mut path, 1.0_f64, &adj, &mut results);
        }

        results
    }

    /// Converts the DAG's link set into a [`CausalGraph`] in the format
    /// expected by the `nexcore_foundry` analyst pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    /// use nexcore_reason::inference::InferenceEngine;
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("a"), label: "Factor A".to_string(), node_type: NodeType::Metric });
    /// dag.add_node(CausalNode { id: NodeId::new("b"), label: "Risk B".to_string(), node_type: NodeType::Risk });
    /// dag.add_link(CausalLink { from: NodeId::new("a"), to: NodeId::new("b"), strength: 0.75, evidence: String::new() }).unwrap();
    ///
    /// let engine = InferenceEngine::new(dag);
    /// let graph = engine.to_causal_graph();
    /// assert_eq!(graph.edges.len(), 1);
    /// assert_eq!(graph.edges[0].from, "Factor A");
    /// assert_eq!(graph.edges[0].to, "Risk B");
    /// assert!((graph.edges[0].strength - 0.75).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn to_causal_graph(&self) -> CausalGraph {
        let label_map = self.build_label_map();
        let edges = self
            .dag
            .links
            .iter()
            .map(|link| self.link_to_edge(link, &label_map))
            .collect();
        CausalGraph { edges }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Validates that every link endpoint resolves to a known node id.
    fn validate_links(&self) -> Result<(), nexcore_error::NexError> {
        // Collect ids as owned values to avoid double-reference confusion.
        let ids: std::collections::HashSet<NodeId> =
            self.dag.nodes.iter().map(|n| n.id.clone()).collect();

        for link in &self.dag.links {
            nexcore_error::ensure!(
                ids.contains(&link.from),
                "causal link references unknown source node `{}`",
                link.from
            );
            nexcore_error::ensure!(
                ids.contains(&link.to),
                "causal link references unknown target node `{}`",
                link.to
            );
        }
        Ok(())
    }

    /// Builds a forward adjacency map: `NodeId → [(target NodeId, strength)]`.
    fn build_adjacency(&self) -> HashMap<&NodeId, Vec<(&NodeId, f64)>> {
        let mut adj: HashMap<&NodeId, Vec<(&NodeId, f64)>> = HashMap::new();
        for link in &self.dag.links {
            adj.entry(&link.from)
                .or_default()
                .push((&link.to, link.strength));
        }
        adj
    }

    /// Builds an owned `NodeId → label` lookup map.
    fn build_label_map(&self) -> HashMap<NodeId, String> {
        self.dag
            .nodes
            .iter()
            .map(|n| (n.id.clone(), n.label.clone()))
            .collect()
    }

    /// Depth-first traversal accumulating all root-to-leaf paths.
    fn dfs_paths<'dag>(
        &'dag self,
        current_id: &'dag NodeId,
        path: &mut Vec<&'dag NodeId>,
        running_score: f64,
        adj: &HashMap<&'dag NodeId, Vec<(&'dag NodeId, f64)>>,
        results: &mut Vec<(Vec<&'dag NodeId>, f64)>,
    ) {
        let children = adj.get(current_id).map(Vec::as_slice).unwrap_or(&[]);

        if children.is_empty() {
            results.push((path.clone(), running_score));
            return;
        }

        for &(child_id, strength) in children {
            let new_score = running_score * strength;
            path.push(child_id);
            self.dfs_paths(child_id, path, new_score, adj, results);
            path.pop();
        }
    }

    /// Converts a [`CausalLink`] to a [`CausalEdge`] using labels where available.
    fn link_to_edge(&self, link: &CausalLink, label_map: &HashMap<NodeId, String>) -> CausalEdge {
        let from_label = label_map
            .get(&link.from)
            .cloned()
            .unwrap_or_else(|| link.from.as_str().to_string());
        let to_label = label_map
            .get(&link.to)
            .cloned()
            .unwrap_or_else(|| link.to.as_str().to_string());
        CausalEdge {
            from: from_label,
            to: to_label,
            strength: link.strength,
        }
    }

    /// Returns the [`CausalNode`] for `id`, or `None`.
    fn node_by_id(&self, id: &NodeId) -> Option<&CausalNode> {
        self.dag.nodes.iter().find(|n| &n.id == id)
    }

    /// Builds [`Finding`] descriptions from paths terminating at
    /// [`NodeType::Risk`] nodes that score above the risk threshold.
    fn build_findings(&self, chains: &[(Vec<&NodeId>, f64)]) -> Vec<String> {
        let mut findings: Vec<Finding> = Vec::new();

        for (path, score) in chains {
            let score = *score;
            if score < self.config.risk_threshold {
                continue;
            }

            let terminal_id = match path.last() {
                Some(id) => *id,
                None => continue,
            };

            let terminal_node = match self.node_by_id(terminal_id) {
                Some(n) => n,
                None => continue,
            };

            if terminal_node.node_type != NodeType::Risk {
                continue;
            }

            let description = self.describe_finding(path, score, terminal_node);
            let owned_path: Vec<NodeId> = path.iter().map(|id| (*id).clone()).collect();

            if let Some(existing) = findings.iter_mut().find(|f| f.description == description) {
                existing.supporting_paths.push(owned_path);
                if score > existing.severity {
                    existing.severity = score;
                }
            } else {
                findings.push(Finding {
                    description,
                    severity: score,
                    supporting_paths: vec![owned_path],
                });
            }
        }

        findings.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        findings.truncate(self.config.max_findings);
        findings.into_iter().map(|f| f.description).collect()
    }

    /// Builds recommendation strings from [`NodeType::Recommendation`] nodes
    /// in above-threshold chains.
    fn build_recommendations(&self, chains: &[(Vec<&NodeId>, f64)]) -> Vec<String> {
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut recommendations: Vec<(String, f64)> = Vec::new();

        for (path, score) in chains {
            let score = *score;
            if score < self.config.risk_threshold {
                continue;
            }

            for node_id in path.iter().copied() {
                let node = match self.node_by_id(node_id) {
                    Some(n) => n,
                    None => continue,
                };

                if node.node_type != NodeType::Recommendation {
                    continue;
                }

                if seen.insert(node_id.clone()) {
                    recommendations.push((node.label.clone(), score));
                }
            }
        }

        recommendations
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        recommendations.truncate(self.config.max_recommendations);
        recommendations
            .into_iter()
            .map(|(label, _)| label)
            .collect()
    }

    /// Formats a human-readable description for a single causal finding.
    fn describe_finding(&self, path: &[&NodeId], score: f64, terminal: &CausalNode) -> String {
        let chain_labels: Vec<String> = path
            .iter()
            .map(|id| {
                // path.iter() yields &&NodeId; deref once to pass &NodeId.
                self.node_by_id(id)
                    .map_or_else(|| id.as_str().to_string(), |n| n.label.clone())
            })
            .collect();

        format!(
            "Causal chain ({score:.2}) leads to risk \"{risk}\": {chain}",
            score = score,
            risk = terminal.label,
            chain = chain_labels.join(" \u{2192} "),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use nexcore_foundry::analyst::RiskLevel;

    use crate::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};

    use super::{InferenceConfig, InferenceEngine};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn node(id: &str, label: &str, node_type: NodeType) -> CausalNode {
        CausalNode {
            id: NodeId::new(id),
            label: label.to_string(),
            node_type,
        }
    }

    fn link(from: &str, to: &str, strength: f64) -> CausalLink {
        CausalLink {
            from: NodeId::new(from),
            to: NodeId::new(to),
            strength,
            evidence: format!("{from}->{to}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: empty DAG
    // -----------------------------------------------------------------------

    #[test]
    fn empty_dag_produces_low_risk_report() {
        let engine = InferenceEngine::new(CausalDag::new());
        let report = engine.infer().expect("empty DAG inference must succeed");

        assert!(report.findings.is_empty());
        assert!(report.recommendations.is_empty());
        assert_eq!(report.risk_level, RiskLevel::Low);
        assert_eq!(report.confidence, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 2: single root → risk path
    // -----------------------------------------------------------------------

    /// Strength 0.85 ≥ 0.8 → Critical.
    #[test]
    fn single_path_dag_produces_critical_risk() {
        let mut dag = CausalDag::new();
        dag.add_node(node("root", "Root cause", NodeType::Metric));
        dag.add_node(node("risk", "Risk outcome", NodeType::Risk));
        dag.add_link(link("root", "risk", 0.85)).expect("acyclic");

        let engine = InferenceEngine::new(dag);
        let report = engine.infer().expect("single-path inference must succeed");

        assert_eq!(report.risk_level, RiskLevel::Critical);
        assert!(!report.findings.is_empty());
        assert!((report.confidence - 0.85).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Test 3: multi-path risk classification
    // -----------------------------------------------------------------------

    /// Strong path (0.75) → High; weak path (0.40) dropped by confidence floor.
    #[test]
    fn multi_path_risk_classification_uses_highest_score() {
        let mut dag = CausalDag::new();
        dag.add_node(node("root1", "High complexity", NodeType::Metric));
        dag.add_node(node("risk1", "Regression risk", NodeType::Risk));
        dag.add_link(link("root1", "risk1", 0.75)).expect("acyclic");

        dag.add_node(node("root2", "Missing docs", NodeType::Metric));
        dag.add_node(node("risk2", "Onboarding delay", NodeType::Risk));
        dag.add_link(link("root2", "risk2", 0.40)).expect("acyclic");

        let engine = InferenceEngine::new(dag);
        let report = engine.infer().expect("multi-path inference must succeed");

        assert_eq!(report.risk_level, RiskLevel::High);
        assert_eq!(report.findings.len(), 1);
        assert!(
            report.findings[0].contains("Regression risk"),
            "finding should mention the risk label; got: {:?}",
            report.findings
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: config overrides
    // -----------------------------------------------------------------------

    #[test]
    fn config_overrides_affect_report() {
        let mut dag = CausalDag::new();
        dag.add_node(node("root", "Weak root", NodeType::Metric));
        dag.add_node(node("risk", "Minor risk", NodeType::Risk));
        dag.add_link(link("root", "risk", 0.3)).expect("acyclic");

        let config = InferenceConfig {
            confidence_floor: 0.2,
            risk_threshold: 0.25,
            max_findings: 10,
            max_recommendations: 5,
        };
        let engine = InferenceEngine::with_config(dag, config);
        let report = engine
            .infer()
            .expect("config override inference must succeed");

        assert_eq!(report.risk_level, RiskLevel::Moderate);
        assert!(!report.findings.is_empty());
        assert!((report.confidence - 0.3).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Test 5: causal graph export
    // -----------------------------------------------------------------------

    #[test]
    fn causal_graph_export_uses_labels_and_preserves_strength() {
        let mut dag = CausalDag::new();
        dag.add_node(node("n1", "Factor Alpha", NodeType::Metric));
        dag.add_node(node("n2", "Risk Beta", NodeType::Risk));
        dag.add_link(link("n1", "n2", 0.62)).expect("acyclic");

        let engine = InferenceEngine::new(dag);
        let graph = engine.to_causal_graph();

        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].from, "Factor Alpha");
        assert_eq!(graph.edges[0].to, "Risk Beta");
        assert!((graph.edges[0].strength - 0.62).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Test 6: recommendation nodes
    // -----------------------------------------------------------------------

    #[test]
    fn recommendation_nodes_appear_in_report() {
        let mut dag = CausalDag::new();
        dag.add_node(node("root", "Missing coverage", NodeType::Metric));
        dag.add_node(node("factor", "Untested paths", NodeType::Pattern));
        dag.add_node(node(
            "rec",
            "Add integration tests",
            NodeType::Recommendation,
        ));
        dag.add_link(link("root", "factor", 0.9)).expect("acyclic");
        dag.add_link(link("factor", "rec", 0.9)).expect("acyclic");

        let engine = InferenceEngine::new(dag);
        let report = engine
            .infer()
            .expect("recommendation inference must succeed");

        assert!(
            report
                .recommendations
                .iter()
                .any(|r| r == "Add integration tests"),
            "recommendation label should appear; got: {:?}",
            report.recommendations
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: dangling link integrity error
    // -----------------------------------------------------------------------

    #[test]
    fn dangling_link_produces_error() {
        let mut dag = CausalDag::new();
        dag.add_node(node("exists", "Existing node", NodeType::Metric));
        // Push directly to bypass add_link's cycle check so we can test
        // validate_links independently.
        dag.links.push(CausalLink {
            from: NodeId::new("exists"),
            to: NodeId::new("missing"),
            strength: 0.5,
            evidence: String::new(),
        });

        let engine = InferenceEngine::new(dag);
        let result = engine.infer();
        assert!(result.is_err(), "dangling link should produce an error");
    }

    // -----------------------------------------------------------------------
    // Test 8: multi-hop path score
    // -----------------------------------------------------------------------

    /// A → B → C → D: 0.9 × 0.8 × 0.9 = 0.648.
    #[test]
    fn multi_hop_path_score_is_product_of_strengths() {
        let mut dag = CausalDag::new();
        dag.add_node(node("a", "A", NodeType::Metric));
        dag.add_node(node("b", "B", NodeType::Pattern));
        dag.add_node(node("c", "C", NodeType::Pattern));
        dag.add_node(node("d", "D", NodeType::Risk));
        dag.add_link(link("a", "b", 0.9)).expect("acyclic");
        dag.add_link(link("b", "c", 0.8)).expect("acyclic");
        dag.add_link(link("c", "d", 0.9)).expect("acyclic");

        let engine = InferenceEngine::new(dag);
        let chains = engine.find_causal_chains();

        assert_eq!(chains.len(), 1);
        let expected = 0.9_f64 * 0.8 * 0.9;
        assert!(
            (chains[0].1 - expected).abs() < 1e-9,
            "score was {}, expected {}",
            chains[0].1,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: classify_risk boundary values
    // -----------------------------------------------------------------------

    #[test]
    fn classify_risk_boundary_values() {
        let engine = InferenceEngine::new(CausalDag::new());

        assert_eq!(engine.classify_risk(0.0), RiskLevel::Low);
        assert_eq!(engine.classify_risk(0.299), RiskLevel::Low);
        assert_eq!(engine.classify_risk(0.3), RiskLevel::Moderate);
        assert_eq!(engine.classify_risk(0.599), RiskLevel::Moderate);
        assert_eq!(engine.classify_risk(0.6), RiskLevel::High);
        assert_eq!(engine.classify_risk(0.799), RiskLevel::High);
        assert_eq!(engine.classify_risk(0.8), RiskLevel::Critical);
        assert_eq!(engine.classify_risk(1.0), RiskLevel::Critical);
    }
}
