use std::collections::HashMap;

/// Node identifier for multi-input graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// Multi-input computation graph with optimized performance.
/// Forward evaluation is pure; reuse an [`EvalTape`] to cache intermediates explicitly.
#[derive(Debug)]
pub struct MultiGraph {
    nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    next_id: usize,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub enum Node {
    Input(String),
    AfterOperation(Op, Box<[NodeId]>),
    Output(NodeId),
}

/// Operations that can be performed on nodes
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scale(f64),
    Sin,
    Cos,
    Pow(i32),
    Add,
    Mul,
}

/// Workspace that stores intermediate primals/tangents during evaluation.
/// Reuse it across calls to avoid repeated allocations when performance matters.
#[derive(Debug, Default)]
pub struct EvalTape {
    primals: Vec<f64>,
    tangents: Vec<f64>,
}

impl EvalTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            primals: Vec::with_capacity(cap),
            tangents: Vec::with_capacity(cap),
        }
    }

    fn reset(&mut self, needed_size: usize) {
        self.primals.clear();
        self.tangents.clear();
        self.primals.resize(needed_size, 0.0);
        self.tangents.resize(needed_size, 0.0);
    }
}

impl Op {
    fn compute(self, inputs: &[f64]) -> f64 {
        match self {
            Op::Scale(factor) => inputs[0] * factor,
            Op::Sin => inputs[0].sin(),
            Op::Cos => inputs[0].cos(),
            Op::Pow(exp) => inputs[0].powi(exp),
            Op::Add => inputs.iter().sum(),
            Op::Mul => inputs.iter().product(),
        }
    }

    fn compute_derivative(self, inputs: &[f64], input_idx: usize) -> f64 {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => inputs[0].cos(),
            Op::Cos => -inputs[0].sin(),
            Op::Pow(exp) => exp as f64 * inputs[0].powi(exp - 1),
            Op::Add => 1.0,
            Op::Mul => inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != input_idx)
                .map(|(_, &x)| x)
                .product(),
        }
    }
}

impl MultiGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn input(&mut self, name: String) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Input(name.clone()));
        self.node_map.insert(name, id);
        id
    }

    pub fn operation<I>(&mut self, op: Op, inputs: I) -> NodeId
    where
        I: AsRef<[NodeId]>,
    {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes
            .push(Node::AfterOperation(op, Box::from(inputs.as_ref())));
        id
    }

    pub fn output(&mut self, node: NodeId) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Output(node));
        id
    }

    /// Allocate a tape sized for this graph. Reuse it to avoid allocations between runs.
    pub fn tape(&self) -> EvalTape {
        EvalTape::with_capacity(self.nodes.len())
    }

    /// Pure forward evaluation that allocates its own tape. Suitable for single-shot calls.
    pub fn compute(&self, inputs: &[f64]) -> Vec<(f64, f64)> {
        let mut tape = self.tape();
        self.compute_with_tape(inputs, &mut tape)
    }

    /// Forward evaluation that reuses the provided tape to cache intermediates.
    pub fn compute_with_tape(&self, inputs: &[f64], tape: &mut EvalTape) -> Vec<(f64, f64)> {
        tape.reset(self.nodes.len());

        // Create a mapping from input names to their indices in the inputs array
        let mut input_indices = HashMap::new();
        let mut input_count = 0;
        for node in &self.nodes {
            if let Node::Input(name) = node {
                input_indices.insert(name.clone(), input_count);
                input_count += 1;
            }
        }

        // First pass: handle inputs
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Input(name) = node {
                if let Some(&input_idx) = input_indices.get(name) {
                    if input_idx < inputs.len() {
                        tape.primals[i] = inputs[input_idx];
                        tape.tangents[i] = 1.0;
                    } else {
                        // Handle case where input index is out of bounds
                        tape.primals[i] = 0.0;
                        tape.tangents[i] = 0.0;
                    }
                } else {
                    // Handle case where input name is not found
                    tape.primals[i] = 0.0;
                    tape.tangents[i] = 0.0;
                }
            }
        }

        // Second pass: handle operations (topological order)
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::AfterOperation(op, inputs) = node {
                // Pre-allocate input_primals to avoid repeated allocations
                let mut input_primals = Vec::with_capacity(inputs.len());
                for &id in inputs {
                    if id.0 < tape.primals.len() {
                        input_primals.push(tape.primals[id.0]);
                    } else {
                        input_primals.push(0.0);
                    }
                }

                tape.primals[i] = op.compute(&input_primals);

                // Compute derivatives using chain rule
                let mut total_derivative = 0.0;
                for (j, &input_id) in inputs.iter().enumerate() {
                    if input_id.0 < tape.tangents.len() {
                        let partial = op.compute_derivative(&input_primals, j);
                        total_derivative += tape.tangents[input_id.0] * partial;
                    }
                }
                tape.tangents[i] = total_derivative;
            }
        }

        // Third pass: handle outputs
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Output(input_id) = node {
                if input_id.0 < tape.primals.len() {
                    tape.primals[i] = tape.primals[input_id.0];
                    tape.tangents[i] = tape.tangents[input_id.0];
                } else {
                    tape.primals[i] = 0.0;
                    tape.tangents[i] = 0.0;
                }
            }
        }

        // Collect outputs
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                if matches!(node, Node::Output(_)) {
                    Some((tape.primals[i], tape.tangents[i]))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Legacy single-input computation graph (kept for backward compatibility)
#[derive(Clone, Debug)]
pub struct CompGraph {
    ops: Vec<Op>,
}

impl CompGraph {
    pub fn new(ops: Vec<Op>) -> Self {
        Self { ops }
    }

    pub fn tape(&self) -> EvalTape {
        EvalTape::with_capacity(self.ops.len() + 1)
    }

    pub fn compute(&self, input: f64) -> (f64, f64) {
        let mut tape = self.tape();
        self.compute_with_tape(input, &mut tape)
    }

    pub fn compute_with_tape(&self, input: f64, tape: &mut EvalTape) -> (f64, f64) {
        tape.reset(self.ops.len() + 1);

        tape.primals[0] = input;
        tape.tangents[0] = 1.0;

        for (i, op) in self.ops.iter().enumerate() {
            let primal = op.compute(&[tape.primals[i]]);
            let tangent = tape.tangents[i] * op.compute_derivative(&[tape.primals[i]], 0);

            tape.primals[i + 1] = primal;
            tape.tangents[i + 1] = tangent;
        }

        (
            *tape
                .primals
                .last()
                .expect("tape primals should have at least one element"),
            *tape
                .tangents
                .last()
                .expect("tape tangents should have at least one element"),
        )
    }
}

/// Macro for building computation graphs
///
/// # Examples
///
/// Single input graph:
/// ```rust,ignore
/// let graph = graph! {
///     input -> Sin -> Cos -> output
/// };
/// ```
///
/// Multi-input graph:
/// ```rust,ignore
/// let graph = graph! {
///     inputs: [x, y]
///     x -> Pow(2) -> @x_sq
///     y -> Sin -> @y_sin
///     (@x_sq, @y_sin) -> Add -> @result
///     output @result
/// };
/// ```
///
/// Mixed graph (operations without intermediate names):
/// ```rust,ignore
/// let graph = graph! {
///     inputs: [x, y]
///     x -> Pow(2) -> @temp1
///     y -> Cos -> @temp2
///     (@temp1, @temp2) -> Mul -> @res
///     output @res
/// };
/// ```
///
/// # Performance Notes
///
/// The default `compute` path allocates a fresh [`EvalTape`] each call for purity.
/// When you need to reuse buffers, create a tape with `graph.tape()` and call
/// `compute_with_tape` to keep allocations off the hot path. Operations use
/// type-level arity for compile-time safety.
#[macro_export]
macro_rules! graph {
    // Single input graph (backward compatibility)
    (input -> $($rest:tt)*) => {
        {
            use $crate::autodiff::{Op, CompGraph};
            $crate::graph! {
                @build_linear
                [],
                $($rest)*
            }
        }
    };

    // Multi-input graph
    (inputs: [$($input:ident),*] $($rest:tt)*) => {
        {
            use $crate::autodiff::{MultiGraph, Op, NodeId};
            let mut graph = MultiGraph::new();
            $(let $input = graph.input(stringify!($input).to_string());)*
            $crate::graph! {
                @build_multi
                graph,
                $($rest)*
            }
        }
    };

    // Linear building (single input)
    (@build_linear [$($ops:expr,)*], $op:ident -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::$op,],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], $op:ident ( $($op_args:tt)* ) -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::$op($($op_args)*),],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], output) => {
        CompGraph::new(Vec::from([$($ops,)*]))
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // Generic N-ary op without extra args: (@a, @b, @c) -> add -> @result
    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$($node),+]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // Generic N-ary op with extra args: (@a, @b, @c) -> scale(2.0) -> @res
    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$($node),+]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, output @ $node:ident) => {
        $graph.output($node);
        $graph
    };

    (@build_multi $graph:ident, output) => {
        $graph
    };
}
