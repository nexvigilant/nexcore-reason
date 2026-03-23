//! Causal DAG for The Foundry's REASON station (A3).
//!
//! Provides a directed acyclic graph whose nodes are named entities (code
//! modules, metrics, patterns, risks, recommendations) and whose edges are
//! directed causal relationships annotated with a strength score and a
//! free-text evidence description.
//!
//! # Design goals
//!
//! - **Acyclicity guaranteed at insertion time.** [`CausalDag::add_link`]
//!   rejects any edge that would introduce a cycle, so a `CausalDag` value
//!   is always a valid DAG.
//! - **Zero panics.** All fallible paths return [`nexcore_error::NexError`].
//! - **Fully serialisable.** Every public type derives [`serde::Serialize`]
//!   and [`serde::Deserialize`].
//!
//! # Example
//!
//! ```
//! use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
//!
//! let mut dag = CausalDag::new();
//!
//! let metric = CausalNode {
//!     id: NodeId::new("latency"),
//!     label: "P99 latency".to_string(),
//!     node_type: NodeType::Metric,
//! };
//! let risk = CausalNode {
//!     id: NodeId::new("slo-breach"),
//!     label: "SLO breach risk".to_string(),
//!     node_type: NodeType::Risk,
//! };
//!
//! dag.add_node(metric);
//! dag.add_node(risk);
//!
//! dag.add_link(CausalLink {
//!     from: NodeId::new("latency"),
//!     to: NodeId::new("slo-breach"),
//!     strength: 0.85,
//!     evidence: "historical correlation r=0.91".to_string(),
//! }).expect("link should not create a cycle");
//!
//! assert_eq!(dag.roots(), vec![&NodeId::new("latency")]);
//! assert_eq!(dag.leaves(), vec![&NodeId::new("slo-breach")]);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Stable identifier for a node in the [`CausalDag`].
///
/// Wraps a plain `String` so that the type system distinguishes node
/// identifiers from other string values.
///
/// # Example
///
/// ```
/// use nexcore_reason::dag::NodeId;
///
/// let a = NodeId::new("module-a");
/// let b = NodeId::new("module-a");
/// assert_eq!(a, b);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Creates a new [`NodeId`] from any value that converts into a `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::NodeId;
    ///
    /// let id = NodeId::new("latency");
    /// assert_eq!(id.as_str(), "latency");
    /// ```
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns a reference to the underlying identifier string.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::NodeId;
    ///
    /// let id = NodeId::new("risk-score");
    /// assert_eq!(id.as_str(), "risk-score");
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// NodeType
// ---------------------------------------------------------------------------

/// Semantic category of a [`CausalNode`].
///
/// The category does not affect graph semantics; it is metadata for
/// downstream consumers such as renderers and report generators.
///
/// # Example
///
/// ```
/// use nexcore_reason::dag::NodeType;
///
/// let ty = NodeType::Metric;
/// let json = serde_json::to_string(&ty).expect("infallible");
/// assert_eq!(json, r#""Metric""#);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// A quantitative measurement (e.g. P99 latency, error rate).
    Metric,
    /// A structural or behavioural pattern identified by A2.
    Pattern,
    /// A software component or crate.
    Module,
    /// A potential negative outcome or vulnerability.
    Risk,
    /// An actionable conclusion produced by A3.
    Recommendation,
}

// ---------------------------------------------------------------------------
// CausalNode
// ---------------------------------------------------------------------------

/// A vertex in the [`CausalDag`].
///
/// A node represents a named entity — a metric, pattern, module, risk, or
/// recommendation — that participates in causal relationships.
///
/// # Example
///
/// ```
/// use nexcore_reason::dag::{CausalNode, NodeId, NodeType};
///
/// let node = CausalNode {
///     id: NodeId::new("error-rate"),
///     label: "HTTP 5xx error rate".to_string(),
///     node_type: NodeType::Metric,
/// };
/// assert_eq!(node.id.as_str(), "error-rate");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalNode {
    /// Stable identifier — must be unique within a [`CausalDag`].
    pub id: NodeId,
    /// Human-readable display name.
    pub label: String,
    /// Semantic category of this node.
    pub node_type: NodeType,
}

// ---------------------------------------------------------------------------
// CausalLink
// ---------------------------------------------------------------------------

/// A directed edge in the [`CausalDag`] representing a causal relationship.
///
/// The edge goes from `from` (cause) to `to` (effect). The `strength` score
/// is a normalised value in `[0.0, 1.0]` where `1.0` represents perfect
/// causal confidence and `0.0` represents a purely speculative relationship.
///
/// # Example
///
/// ```
/// use nexcore_reason::dag::{CausalLink, NodeId};
///
/// let link = CausalLink {
///     from: NodeId::new("high-memory"),
///     to: NodeId::new("oom-risk"),
///     strength: 0.9,
///     evidence: "heap profiler shows 95% headroom consumed".to_string(),
/// };
/// assert_eq!(link.strength, 0.9);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalLink {
    /// Identifier of the cause node.
    pub from: NodeId,
    /// Identifier of the effect node.
    pub to: NodeId,
    /// Causal confidence in `[0.0, 1.0]`.
    pub strength: f64,
    /// Free-text description of the evidence supporting this link.
    pub evidence: String,
}

// ---------------------------------------------------------------------------
// CausalDag
// ---------------------------------------------------------------------------

/// A directed acyclic graph of causal relationships.
///
/// Nodes represent entities; edges represent directed causal links with
/// associated strength scores and evidence descriptions.
///
/// Acyclicity is maintained as an invariant: [`add_link`] rejects any edge
/// that would create a cycle and returns an [`nexcore_error::NexError`] instead.
///
/// [`add_link`]: CausalDag::add_link
///
/// # Example
///
/// ```
/// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
///
/// let mut dag = CausalDag::new();
///
/// dag.add_node(CausalNode {
///     id: NodeId::new("a"),
///     label: "Module A".to_string(),
///     node_type: NodeType::Module,
/// });
/// dag.add_node(CausalNode {
///     id: NodeId::new("b"),
///     label: "Risk B".to_string(),
///     node_type: NodeType::Risk,
/// });
///
/// let result = dag.add_link(CausalLink {
///     from: NodeId::new("a"),
///     to: NodeId::new("b"),
///     strength: 0.7,
///     evidence: "static analysis".to_string(),
/// });
/// assert!(result.is_ok());
/// assert!(!dag.has_cycle());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CausalDag {
    /// All nodes registered in the DAG.
    pub nodes: Vec<CausalNode>,
    /// All directed edges registered in the DAG.
    pub links: Vec<CausalLink>,
}

impl CausalDag {
    /// Creates a new, empty [`CausalDag`].
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::CausalDag;
    ///
    /// let dag = CausalDag::new();
    /// assert!(dag.nodes.is_empty());
    /// assert!(dag.links.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node to the DAG.
    ///
    /// Duplicate node identifiers are allowed at the data level, but callers
    /// should ensure identifiers are unique; graph traversal methods rely on
    /// identifier uniqueness for correct results.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode {
    ///     id: NodeId::new("metric-x"),
    ///     label: "Metric X".to_string(),
    ///     node_type: NodeType::Metric,
    /// });
    /// assert_eq!(dag.nodes.len(), 1);
    /// ```
    pub fn add_node(&mut self, node: CausalNode) {
        self.nodes.push(node);
    }

    /// Appends a directed causal link to the DAG.
    ///
    /// The link is validated before insertion: if adding it would create a
    /// cycle the method returns an [`nexcore_error::NexError`] and the DAG is left
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the new link would create a cycle in the graph.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("x"), label: "X".to_string(), node_type: NodeType::Pattern });
    /// dag.add_node(CausalNode { id: NodeId::new("y"), label: "Y".to_string(), node_type: NodeType::Risk });
    ///
    /// assert!(dag.add_link(CausalLink {
    ///     from: NodeId::new("x"),
    ///     to: NodeId::new("y"),
    ///     strength: 0.5,
    ///     evidence: "observed".to_string(),
    /// }).is_ok());
    ///
    /// // A back-edge would form a cycle and must be rejected.
    /// assert!(dag.add_link(CausalLink {
    ///     from: NodeId::new("y"),
    ///     to: NodeId::new("x"),
    ///     strength: 0.3,
    ///     evidence: "hypothetical".to_string(),
    /// }).is_err());
    /// ```
    pub fn add_link(&mut self, link: CausalLink) -> Result<(), nexcore_error::NexError> {
        // Temporarily push the link then check for a cycle.  If a cycle is
        // detected, remove the link so the DAG invariant is preserved.
        self.links.push(link);

        if self.has_cycle() {
            // The last element is the one we just pushed.
            self.links.pop();
            return Err(nexcore_error::nexerror!(
                "adding this link would introduce a cycle in the causal DAG"
            ));
        }

        Ok(())
    }

    /// Returns the identifiers of all **root** nodes — those with no incoming
    /// edges.
    ///
    /// In a causal DAG, roots are the initial causes: nodes that are not the
    /// effect of any other node currently in the graph.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("cause"), label: "Cause".to_string(), node_type: NodeType::Metric });
    /// dag.add_node(CausalNode { id: NodeId::new("effect"), label: "Effect".to_string(), node_type: NodeType::Risk });
    /// dag.add_link(CausalLink {
    ///     from: NodeId::new("cause"),
    ///     to: NodeId::new("effect"),
    ///     strength: 0.8,
    ///     evidence: "direct".to_string(),
    /// }).expect("acyclic");
    ///
    /// assert_eq!(dag.roots(), vec![&NodeId::new("cause")]);
    /// ```
    #[must_use]
    pub fn roots(&self) -> Vec<&NodeId> {
        let targets: HashSet<&NodeId> = self.links.iter().map(|l| &l.to).collect();
        self.nodes
            .iter()
            .filter(|n| !targets.contains(&n.id))
            .map(|n| &n.id)
            .collect()
    }

    /// Returns the identifiers of all **leaf** nodes — those with no outgoing
    /// edges.
    ///
    /// In a causal DAG, leaves are the terminal effects: nodes that do not
    /// cause any other node currently in the graph.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("cause"), label: "Cause".to_string(), node_type: NodeType::Metric });
    /// dag.add_node(CausalNode { id: NodeId::new("effect"), label: "Effect".to_string(), node_type: NodeType::Risk });
    /// dag.add_link(CausalLink {
    ///     from: NodeId::new("cause"),
    ///     to: NodeId::new("effect"),
    ///     strength: 0.8,
    ///     evidence: "direct".to_string(),
    /// }).expect("acyclic");
    ///
    /// assert_eq!(dag.leaves(), vec![&NodeId::new("effect")]);
    /// ```
    #[must_use]
    pub fn leaves(&self) -> Vec<&NodeId> {
        let sources: HashSet<&NodeId> = self.links.iter().map(|l| &l.from).collect();
        self.nodes
            .iter()
            .filter(|n| !sources.contains(&n.id))
            .map(|n| &n.id)
            .collect()
    }

    /// Returns all **transitive ancestors** of the given node.
    ///
    /// An ancestor of node `N` is any node `A` from which there exists a
    /// directed path `A -> ... -> N`. The result does not include `id` itself
    /// and has no guaranteed ordering.
    ///
    /// Returns an empty vector when `id` is not present in the DAG or has no
    /// ancestors.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// for name in ["a", "b", "c"] {
    ///     dag.add_node(CausalNode { id: NodeId::new(name), label: name.to_string(), node_type: NodeType::Module });
    /// }
    /// dag.add_link(CausalLink { from: NodeId::new("a"), to: NodeId::new("b"), strength: 1.0, evidence: String::new() }).unwrap();
    /// dag.add_link(CausalLink { from: NodeId::new("b"), to: NodeId::new("c"), strength: 1.0, evidence: String::new() }).unwrap();
    ///
    /// let mut ancs: Vec<_> = dag.ancestors(&NodeId::new("c")).into_iter().map(NodeId::as_str).collect();
    /// ancs.sort();
    /// assert_eq!(ancs, vec!["a", "b"]);
    /// ```
    #[must_use]
    pub fn ancestors(&self, id: &NodeId) -> Vec<&NodeId> {
        // Reverse-adjacency map: target -> [sources].
        let mut parents: HashMap<&NodeId, Vec<&NodeId>> = HashMap::new();
        for link in &self.links {
            parents.entry(&link.to).or_default().push(&link.from);
        }

        let mut visited: HashSet<&NodeId> = HashSet::new();
        let mut queue: VecDeque<&NodeId> = VecDeque::new();

        if let Some(direct_parents) = parents.get(id) {
            for p in direct_parents {
                if visited.insert(p) {
                    queue.push_back(p);
                }
            }
        }

        while let Some(current) = queue.pop_front() {
            if let Some(grandparents) = parents.get(current) {
                for gp in grandparents {
                    if visited.insert(gp) {
                        queue.push_back(gp);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Returns all **transitive descendants** of the given node.
    ///
    /// A descendant of node `N` is any node `D` reachable via a directed path
    /// `N -> ... -> D`. The result does not include `id` itself and has no
    /// guaranteed ordering.
    ///
    /// Returns an empty vector when `id` is not present in the DAG or has no
    /// descendants.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// for name in ["a", "b", "c"] {
    ///     dag.add_node(CausalNode { id: NodeId::new(name), label: name.to_string(), node_type: NodeType::Module });
    /// }
    /// dag.add_link(CausalLink { from: NodeId::new("a"), to: NodeId::new("b"), strength: 1.0, evidence: String::new() }).unwrap();
    /// dag.add_link(CausalLink { from: NodeId::new("b"), to: NodeId::new("c"), strength: 1.0, evidence: String::new() }).unwrap();
    ///
    /// let mut descs: Vec<_> = dag.descendants(&NodeId::new("a")).into_iter().map(NodeId::as_str).collect();
    /// descs.sort();
    /// assert_eq!(descs, vec!["b", "c"]);
    /// ```
    #[must_use]
    pub fn descendants(&self, id: &NodeId) -> Vec<&NodeId> {
        // Forward-adjacency map: source -> [targets].
        let mut children: HashMap<&NodeId, Vec<&NodeId>> = HashMap::new();
        for link in &self.links {
            children.entry(&link.from).or_default().push(&link.to);
        }

        let mut visited: HashSet<&NodeId> = HashSet::new();
        let mut queue: VecDeque<&NodeId> = VecDeque::new();

        if let Some(direct_children) = children.get(id) {
            for c in direct_children {
                if visited.insert(c) {
                    queue.push_back(c);
                }
            }
        }

        while let Some(current) = queue.pop_front() {
            if let Some(grandchildren) = children.get(current) {
                for gc in grandchildren {
                    if visited.insert(gc) {
                        queue.push_back(gc);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Returns `true` when the graph contains at least one cycle.
    ///
    /// Uses Kahn's algorithm: builds in-degree counts for every node that
    /// appears in any edge, processes nodes with zero in-degree, and
    /// concludes that a cycle exists when the processed count is less than
    /// the total number of distinct nodes referenced by edges.
    ///
    /// A graph with no edges is always acyclic.
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// dag.add_node(CausalNode { id: NodeId::new("p"), label: "P".to_string(), node_type: NodeType::Pattern });
    /// dag.add_node(CausalNode { id: NodeId::new("q"), label: "Q".to_string(), node_type: NodeType::Pattern });
    ///
    /// dag.add_link(CausalLink {
    ///     from: NodeId::new("p"),
    ///     to: NodeId::new("q"),
    ///     strength: 0.6,
    ///     evidence: String::new(),
    /// }).unwrap();
    ///
    /// assert!(!dag.has_cycle());
    /// ```
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        // Collect every node id referenced by at least one edge.
        let mut all_ids: HashSet<&NodeId> = HashSet::new();
        for link in &self.links {
            all_ids.insert(&link.from);
            all_ids.insert(&link.to);
        }

        if all_ids.is_empty() {
            return false;
        }

        // Build in-degree map initialised to zero.
        let mut in_degree: HashMap<&NodeId, usize> =
            all_ids.iter().map(|&id| (id, 0_usize)).collect();

        for link in &self.links {
            *in_degree.entry(&link.to).or_insert(0) += 1;
        }

        // Adjacency list: source -> [targets].
        let mut adj: HashMap<&NodeId, Vec<&NodeId>> = HashMap::new();
        for link in &self.links {
            adj.entry(&link.from).or_default().push(&link.to);
        }

        // Seed the queue with zero-in-degree nodes.
        let mut queue: VecDeque<&NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut processed: usize = 0;

        while let Some(current) = queue.pop_front() {
            processed += 1;
            if let Some(neighbours) = adj.get(current) {
                for nb in neighbours {
                    let deg = in_degree.entry(*nb).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(*nb);
                    }
                }
            }
        }

        processed < all_ids.len()
    }

    /// Returns a topological ordering of all nodes in the DAG.
    ///
    /// The result lists every node registered via [`add_node`] in an order
    /// where each node appears before all of its descendants. Nodes that are
    /// not referenced by any edge appear in their insertion order at the front
    /// of the result (they have no ordering constraints).
    ///
    /// Uses Kahn's algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph contains a cycle. Under normal operation
    /// this cannot happen because [`add_link`] prevents cycles at insertion
    /// time; the error path exists as a safety net for graphs that were
    /// deserialised from external sources without going through `add_link`.
    ///
    /// [`add_node`]: CausalDag::add_node
    /// [`add_link`]: CausalDag::add_link
    ///
    /// # Example
    ///
    /// ```
    /// use nexcore_reason::dag::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    ///
    /// let mut dag = CausalDag::new();
    /// for name in ["x", "y", "z"] {
    ///     dag.add_node(CausalNode { id: NodeId::new(name), label: name.to_string(), node_type: NodeType::Module });
    /// }
    /// dag.add_link(CausalLink { from: NodeId::new("x"), to: NodeId::new("y"), strength: 1.0, evidence: String::new() }).unwrap();
    /// dag.add_link(CausalLink { from: NodeId::new("y"), to: NodeId::new("z"), strength: 1.0, evidence: String::new() }).unwrap();
    ///
    /// let order = dag.topological_order().expect("acyclic");
    /// // x must appear before y, and y before z.
    /// let pos: std::collections::HashMap<_, _> =
    ///     order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    /// assert!(pos[&NodeId::new("x")] < pos[&NodeId::new("y")]);
    /// assert!(pos[&NodeId::new("y")] < pos[&NodeId::new("z")]);
    /// ```
    pub fn topological_order(&self) -> Result<Vec<&NodeId>, nexcore_error::NexError> {
        // Map each NodeId to its insertion index so we can produce a stable,
        // deterministic result for nodes with equal in-degree.
        let node_index: HashMap<&NodeId, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (&n.id, i))
            .collect();

        // Compute in-degree for every registered node. Nodes with no edges
        // at all start at degree 0.
        let mut in_degree: HashMap<&NodeId, usize> =
            node_index.keys().map(|&id| (id, 0_usize)).collect();

        for link in &self.links {
            *in_degree.entry(&link.to).or_insert(0) += 1;
        }

        // Adjacency list: source -> [targets].
        let mut adj: HashMap<&NodeId, Vec<&NodeId>> = HashMap::new();
        for link in &self.links {
            adj.entry(&link.from).or_default().push(&link.to);
        }

        // Seed with zero-in-degree nodes, sorted by insertion index for
        // deterministic output.
        let mut zero_deg: Vec<&NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        zero_deg.sort_by_key(|id| node_index.get(id).copied().unwrap_or(usize::MAX));

        let mut queue: VecDeque<&NodeId> = zero_deg.into();
        let mut result: Vec<&NodeId> = Vec::with_capacity(self.nodes.len());

        while let Some(current) = queue.pop_front() {
            result.push(current);
            if let Some(neighbours) = adj.get(current) {
                let mut sorted = neighbours.clone();
                sorted.sort_by_key(|id| node_index.get(id).copied().unwrap_or(usize::MAX));
                for nb in sorted {
                    let deg = in_degree.entry(nb).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        if result.len() < self.nodes.len() {
            return Err(nexcore_error::nexerror!(
                "topological sort failed: the DAG contains a cycle"
            ));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{CausalDag, CausalLink, CausalNode, NodeId, NodeType};
    use std::collections::HashMap;

    // -- Helpers -------------------------------------------------------------

    fn node(id: &str, ty: NodeType) -> CausalNode {
        CausalNode {
            id: NodeId::new(id),
            label: id.to_string(),
            node_type: ty,
        }
    }

    fn link(from: &str, to: &str, strength: f64) -> CausalLink {
        CausalLink {
            from: NodeId::new(from),
            to: NodeId::new(to),
            strength,
            evidence: format!("{from} causes {to}"),
        }
    }

    // -- Tests ---------------------------------------------------------------

    /// A freshly constructed DAG must be empty and free of cycles.
    #[test]
    fn new_dag_is_empty_and_acyclic() {
        let dag = CausalDag::new();
        assert!(dag.nodes.is_empty());
        assert!(dag.links.is_empty());
        assert!(!dag.has_cycle());
    }

    /// Nodes can be added and are accessible through the `nodes` field.
    #[test]
    fn add_node_appends_in_order() {
        let mut dag = CausalDag::new();
        dag.add_node(node("a", NodeType::Metric));
        dag.add_node(node("b", NodeType::Risk));

        assert_eq!(dag.nodes.len(), 2);
        assert_eq!(dag.nodes[0].id, NodeId::new("a"));
        assert_eq!(dag.nodes[1].id, NodeId::new("b"));
    }

    /// Adding a valid acyclic link must succeed and the link must be stored.
    #[test]
    fn add_link_acyclic_succeeds() {
        let mut dag = CausalDag::new();
        dag.add_node(node("a", NodeType::Module));
        dag.add_node(node("b", NodeType::Risk));

        assert!(dag.add_link(link("a", "b", 0.8)).is_ok());
        assert_eq!(dag.links.len(), 1);
        assert!(!dag.has_cycle());
    }

    /// Adding a direct back-edge (a->b then b->a) must fail and leave the DAG
    /// with only the original link.
    #[test]
    fn add_link_back_edge_rejected_and_dag_unchanged() {
        let mut dag = CausalDag::new();
        dag.add_node(node("a", NodeType::Pattern));
        dag.add_node(node("b", NodeType::Pattern));

        dag.add_link(link("a", "b", 0.6)).expect("acyclic");
        let result = dag.add_link(link("b", "a", 0.4));

        assert!(result.is_err());
        assert_eq!(dag.links.len(), 1);
        assert!(!dag.has_cycle());
    }

    /// A three-node transitive cycle (a->b, b->c, c->a) must be caught on
    /// the final edge.
    #[test]
    fn add_link_transitive_cycle_rejected() {
        let mut dag = CausalDag::new();
        for name in ["a", "b", "c"] {
            dag.add_node(node(name, NodeType::Module));
        }

        dag.add_link(link("a", "b", 1.0)).expect("acyclic");
        dag.add_link(link("b", "c", 1.0)).expect("acyclic");

        assert!(dag.add_link(link("c", "a", 0.5)).is_err());
        assert_eq!(dag.links.len(), 2);
        assert!(!dag.has_cycle());
    }

    /// `roots` must return only nodes with no incoming edges.
    #[test]
    fn roots_returns_nodes_with_no_incoming_edges() {
        let mut dag = CausalDag::new();
        for name in ["root", "middle", "leaf"] {
            dag.add_node(node(name, NodeType::Metric));
        }
        dag.add_link(link("root", "middle", 0.9)).expect("acyclic");
        dag.add_link(link("middle", "leaf", 0.7)).expect("acyclic");

        let roots = dag.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], &NodeId::new("root"));
    }

    /// `leaves` must return only nodes with no outgoing edges.
    #[test]
    fn leaves_returns_nodes_with_no_outgoing_edges() {
        let mut dag = CausalDag::new();
        for name in ["root", "middle", "leaf"] {
            dag.add_node(node(name, NodeType::Recommendation));
        }
        dag.add_link(link("root", "middle", 0.9)).expect("acyclic");
        dag.add_link(link("middle", "leaf", 0.7)).expect("acyclic");

        let leaves = dag.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0], &NodeId::new("leaf"));
    }

    /// A node with no edges must appear in both `roots` and `leaves`.
    #[test]
    fn isolated_node_is_both_root_and_leaf() {
        let mut dag = CausalDag::new();
        dag.add_node(node("alone", NodeType::Pattern));

        assert!(dag.roots().contains(&&NodeId::new("alone")));
        assert!(dag.leaves().contains(&&NodeId::new("alone")));
    }

    /// `ancestors` must return all transitive predecessors of a node.
    #[test]
    fn ancestors_returns_transitive_predecessors() {
        let mut dag = CausalDag::new();
        for name in ["a", "b", "c", "d"] {
            dag.add_node(node(name, NodeType::Module));
        }
        // Chain: a -> b -> c -> d
        dag.add_link(link("a", "b", 1.0)).expect("acyclic");
        dag.add_link(link("b", "c", 1.0)).expect("acyclic");
        dag.add_link(link("c", "d", 1.0)).expect("acyclic");

        let mut ancs: Vec<&str> = dag
            .ancestors(&NodeId::new("d"))
            .into_iter()
            .map(NodeId::as_str)
            .collect();
        ancs.sort_unstable();

        assert_eq!(ancs, vec!["a", "b", "c"]);
    }

    /// `descendants` must return all transitive successors of a node.
    #[test]
    fn descendants_returns_transitive_successors() {
        let mut dag = CausalDag::new();
        for name in ["a", "b", "c", "d"] {
            dag.add_node(node(name, NodeType::Module));
        }
        // Chain: a -> b -> c -> d
        dag.add_link(link("a", "b", 1.0)).expect("acyclic");
        dag.add_link(link("b", "c", 1.0)).expect("acyclic");
        dag.add_link(link("c", "d", 1.0)).expect("acyclic");

        let mut descs: Vec<&str> = dag
            .descendants(&NodeId::new("a"))
            .into_iter()
            .map(NodeId::as_str)
            .collect();
        descs.sort_unstable();

        assert_eq!(descs, vec!["b", "c", "d"]);
    }

    /// `topological_order` must place every cause before its effect in a
    /// diamond-shaped DAG (a->b, a->c, b->d, c->d).
    #[test]
    fn topological_order_respects_edge_direction_in_diamond() {
        let mut dag = CausalDag::new();
        for name in ["a", "b", "c", "d"] {
            dag.add_node(node(name, NodeType::Risk));
        }
        dag.add_link(link("a", "b", 1.0)).expect("acyclic");
        dag.add_link(link("a", "c", 1.0)).expect("acyclic");
        dag.add_link(link("b", "d", 1.0)).expect("acyclic");
        dag.add_link(link("c", "d", 1.0)).expect("acyclic");

        let order = dag.topological_order().expect("acyclic diamond");
        let pos: HashMap<&NodeId, usize> =
            order.iter().enumerate().map(|(i, id)| (*id, i)).collect();

        assert!(pos[&NodeId::new("a")] < pos[&NodeId::new("b")]);
        assert!(pos[&NodeId::new("a")] < pos[&NodeId::new("c")]);
        assert!(pos[&NodeId::new("b")] < pos[&NodeId::new("d")]);
        assert!(pos[&NodeId::new("c")] < pos[&NodeId::new("d")]);
    }

    /// `topological_order` must return an error for cyclic graphs.  This
    /// exercises the safety net for graphs deserialised from untrusted sources
    /// that bypass `add_link`.
    #[test]
    fn topological_order_errors_on_deserialized_cycle() {
        let mut dag = CausalDag::new();
        dag.add_node(node("x", NodeType::Metric));
        dag.add_node(node("y", NodeType::Metric));

        // Bypass `add_link` to inject a cycle directly.
        dag.links.push(link("x", "y", 0.5));
        dag.links.push(link("y", "x", 0.5));

        assert!(dag.has_cycle());
        assert!(dag.topological_order().is_err());
    }

    /// Serialisation round-trip: a DAG serialised to JSON and deserialised
    /// must be structurally identical to the original.
    #[test]
    fn serde_round_trip_preserves_dag() {
        let mut dag = CausalDag::new();
        dag.add_node(node("src", NodeType::Module));
        dag.add_node(node("dst", NodeType::Recommendation));
        dag.add_link(link("src", "dst", 0.42)).expect("acyclic");

        let json = serde_json::to_string(&dag).expect("serialisation must not fail");
        let restored: CausalDag =
            serde_json::from_str(&json).expect("deserialisation must not fail");

        assert_eq!(restored.nodes.len(), dag.nodes.len());
        assert_eq!(restored.links.len(), dag.links.len());
        assert_eq!(restored.links[0].strength, 0.42);
        assert_eq!(restored.links[0].from, NodeId::new("src"));
        assert_eq!(restored.links[0].to, NodeId::new("dst"));
    }
}
