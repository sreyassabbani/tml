use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::RefCell, rc::Rc};

use crate::Float;

/// Node identifier for multi-input graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: usize,
    graph_id: u64,
}

impl NodeId {
    fn new(index: usize, graph_id: u64) -> Self {
        Self { index, graph_id }
    }
}

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Multi-input computation graph with optimized performance.
/// Forward evaluation is pure; reuse an [`EvalTape`] to cache intermediates explicitly.
#[derive(Debug)]
pub struct MultiGraph {
    graph_id: u64,
    nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    inputs: Vec<NodeId>,
    input_names: Vec<String>,
    outputs: Vec<NodeId>,
    max_arity: usize,
    next_id: usize,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub enum Node {
    Input(String),
    Const(Float),
    AfterOperation(Op, Box<[NodeId]>),
    Output(NodeId),
}

/// Operations that can be performed on nodes
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scale(Float),
    Sin,
    Cos,
    Pow(i32),
    Add,
    Mul,
}

/// Workspace that stores intermediate primals and gradient vectors during evaluation.
/// Reuse it across calls to avoid repeated allocations when performance matters.
#[derive(Debug, Default)]
pub struct EvalTape {
    primals: Vec<Float>,
    tangents: Vec<Float>,
    input_count: usize,
    scratch_primals: Vec<Float>,
    scratch_partials: Vec<Float>,
}

impl EvalTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(nodes: usize, input_count: usize, max_arity: usize) -> Self {
        Self {
            primals: Vec::with_capacity(nodes),
            tangents: Vec::with_capacity(nodes * input_count),
            input_count,
            scratch_primals: Vec::with_capacity(max_arity),
            scratch_partials: Vec::with_capacity(max_arity),
        }
    }

    fn reset(&mut self, nodes: usize, input_count: usize, max_arity: usize) {
        self.input_count = input_count;
        self.primals.clear();
        self.tangents.clear();
        self.primals.resize(nodes, 0.0);
        self.tangents.resize(nodes * input_count, 0.0);
        self.scratch_primals.clear();
        self.scratch_partials.clear();
        self.scratch_primals.resize(max_arity, 0.0);
        self.scratch_partials.resize(max_arity, 0.0);
    }

    fn tangent_index(&self, node_idx: usize, input_idx: usize) -> usize {
        node_idx * self.input_count + input_idx
    }
}

/// Workspace that stores intermediate primals and adjoints during reverse-mode evaluation.
#[derive(Debug, Default)]
pub struct ReverseTape {
    primals: Vec<Float>,
    adjoints: Vec<Float>,
    scratch_primals: Vec<Float>,
    scratch_partials: Vec<Float>,
}

impl ReverseTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(nodes: usize, max_arity: usize) -> Self {
        Self {
            primals: Vec::with_capacity(nodes),
            adjoints: Vec::with_capacity(nodes),
            scratch_primals: Vec::with_capacity(max_arity),
            scratch_partials: Vec::with_capacity(max_arity),
        }
    }

    fn reset(&mut self, nodes: usize, max_arity: usize) {
        self.primals.clear();
        self.adjoints.clear();
        self.primals.resize(nodes, 0.0);
        self.adjoints.resize(nodes, 0.0);
        self.scratch_primals.clear();
        self.scratch_partials.clear();
        self.scratch_primals.resize(max_arity, 0.0);
        self.scratch_partials.resize(max_arity, 0.0);
    }
}

impl Op {
    fn validate_arity(self, inputs_len: usize) {
        let ok = match self {
            Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => inputs_len == 1,
            Op::Add | Op::Mul => inputs_len >= 2,
        };

        assert!(
            ok,
            "invalid arity for {:?}: expected {}, got {}",
            self,
            match self {
                Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => "1",
                Op::Add | Op::Mul => ">= 2",
            },
            inputs_len
        );
    }

    fn compute(self, inputs: &[Float]) -> Float {
        match self {
            Op::Scale(factor) => inputs[0] * factor,
            Op::Sin => inputs[0].sin(),
            Op::Cos => inputs[0].cos(),
            Op::Pow(exp) => inputs[0].powi(exp),
            Op::Add => inputs.iter().sum(),
            Op::Mul => inputs.iter().product(),
        }
    }

    fn compute_derivative(self, inputs: &[Float], input_idx: usize) -> Float {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => inputs[0].cos(),
            Op::Cos => -inputs[0].sin(),
            Op::Pow(exp) => exp as Float * inputs[0].powi(exp - 1),
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
            graph_id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
            node_map: HashMap::new(),
            inputs: Vec::new(),
            input_names: Vec::new(),
            outputs: Vec::new(),
            max_arity: 0,
            next_id: 0,
        }
    }

    fn make_node_id(&self, index: usize) -> NodeId {
        NodeId::new(index, self.graph_id)
    }

    fn is_valid_node(&self, id: NodeId) -> bool {
        id.graph_id == self.graph_id && id.index < self.next_id
    }

    fn assert_valid_node(&self, id: NodeId, context: &str) {
        assert!(
            self.is_valid_node(id),
            "{context} does not belong to this graph or is out of bounds"
        );
    }

    pub fn input(&mut self, name: String) -> NodeId {
        assert!(
            !self.node_map.contains_key(&name),
            "input name already exists: {name}"
        );

        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Input(name.clone()));
        self.node_map.insert(name.clone(), id);
        self.inputs.push(id);
        self.input_names.push(name);
        id
    }

    pub fn constant(&mut self, value: Float) -> NodeId {
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Const(value));
        id
    }

    pub fn operation<I>(&mut self, op: Op, inputs: I) -> NodeId
    where
        I: AsRef<[NodeId]>,
    {
        let inputs_ref = inputs.as_ref();
        op.validate_arity(inputs_ref.len());
        assert!(
            inputs_ref.iter().all(|id| self.is_valid_node(*id)),
            "operation inputs must reference earlier nodes in the same graph"
        );
        self.max_arity = self.max_arity.max(inputs_ref.len());
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes
            .push(Node::AfterOperation(op, Box::from(inputs_ref)));
        id
    }

    pub fn output(&mut self, node: NodeId) -> NodeId {
        self.assert_valid_node(node, "output node");
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Output(node));
        self.outputs.push(id);
        id
    }

    /// Allocate a tape sized for this graph. Reuse it to avoid allocations between runs.
    pub fn tape(&self) -> EvalTape {
        EvalTape::with_capacity(self.nodes.len(), self.inputs.len(), self.max_arity)
    }

    pub fn reverse_tape(&self) -> ReverseTape {
        ReverseTape::with_capacity(self.nodes.len(), self.max_arity)
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Pure forward evaluation that allocates its own tape. Suitable for single-shot calls.
    /// Returns a value and per-input gradient vector for each output.
    pub fn compute(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.tape();
        self.compute_with_tape(inputs, &mut tape)
    }

    /// Forward evaluation that reuses the provided tape to cache intermediates.
    /// Returns a value and per-input gradient vector for each output.
    pub fn compute_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );

        tape.reset(self.nodes.len(), self.inputs.len(), self.max_arity);

        // First pass: handle inputs (ordered by definition)
        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            let node_idx = node_id.index;
            tape.primals[node_idx] = inputs[input_idx];
            let tangent_idx = tape.tangent_index(node_idx, input_idx);
            tape.tangents[tangent_idx] = 1.0;
        }

        // Second pass: handle operations (topological order)
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }

                    tape.primals[i] = op.compute(input_primals);

                    // Compute derivatives using chain rule for each input dimension
                    let partials = &mut tape.scratch_partials[..arity];
                    for (j, partial) in partials.iter_mut().enumerate() {
                        *partial = op.compute_derivative(input_primals, j);
                    }

                    let input_count = tape.input_count;
                    let tangents = &mut tape.tangents;
                    for input_dim in 0..input_count {
                        let mut total = 0.0;
                        for (j, &input_id) in inputs.iter().enumerate() {
                            let idx = input_id.index * input_count + input_dim;
                            total += tangents[idx] * partials[j];
                        }
                        let out_idx = i * input_count + input_dim;
                        tangents[out_idx] = total;
                    }
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                _ => {}
            }
        }

        // Third pass: handle outputs
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Output(input_id) = node {
                tape.primals[i] = tape.primals[input_id.index];
                let src_start = tape.tangent_index(input_id.index, 0);
                let dst_start = tape.tangent_index(i, 0);
                let len = tape.input_count;
                tape.tangents
                    .copy_within(src_start..(src_start + len), dst_start);
            }
        }

        self.outputs
            .iter()
            .map(|id| {
                let idx = id.index;
                let start = tape.tangent_index(idx, 0);
                let end = start + tape.input_count;
                (tape.primals[idx], tape.tangents[start..end].to_vec())
            })
            .collect()
    }

    pub fn compute_single(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.tape();
        self.compute_single_with_tape(inputs, &mut tape)
    }

    pub fn compute_single_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.compute_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    pub fn compute_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.tape();
        self.compute_with_tape_named(inputs, &mut tape)
    }

    pub fn compute_with_tape_named(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.compute_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    /// Reverse-mode evaluation that allocates its own tape. Suitable for single-shot calls.
    /// Returns a value and per-input gradient vector for each output.
    pub fn compute_reverse(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.compute_reverse_with_tape(inputs, &mut tape)
    }

    /// Reverse-mode evaluation that reuses the provided tape to cache intermediates.
    /// Returns a value and per-input gradient vector for each output.
    pub fn compute_reverse_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        self.compute_reverse_with_tape_for(inputs, &self.outputs, tape)
    }

    /// Reverse-mode evaluation for a selected set of outputs.
    pub fn compute_reverse_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
    ) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.compute_reverse_with_tape_for(inputs, outputs, &mut tape)
    }

    /// Reverse-mode evaluation for a selected set of outputs with a reusable tape.
    pub fn compute_reverse_with_tape_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        for &output in outputs {
            self.assert_valid_node(output, "requested output");
        }

        tape.reset(self.nodes.len(), self.max_arity);

        // Forward primals
        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            tape.primals[node_id.index] = inputs[input_idx];
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }
                    tape.primals[i] = op.compute(input_primals);
                }
                Node::Output(input_id) => {
                    tape.primals[i] = tape.primals[input_id.index];
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                Node::Input(_) => {}
            }
        }

        let mut results = Vec::with_capacity(outputs.len());

        for output_id in outputs {
            tape.adjoints.fill(0.0);
            tape.adjoints[output_id.index] = 1.0;

            for (i, node) in self.nodes.iter().enumerate().rev() {
                match node {
                    Node::Output(input_id) => {
                        tape.adjoints[input_id.index] += tape.adjoints[i];
                    }
                    Node::AfterOperation(op, inputs) => {
                        let arity = inputs.len();
                        let input_primals = &mut tape.scratch_primals[..arity];
                        for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                            *slot = tape.primals[id.index];
                        }

                        let partials = &mut tape.scratch_partials[..arity];
                        for (j, partial) in partials.iter_mut().enumerate() {
                            *partial = op.compute_derivative(input_primals, j);
                        }

                        let adj = tape.adjoints[i];
                        if adj != 0.0 {
                            for (j, &input_id) in inputs.iter().enumerate() {
                                tape.adjoints[input_id.index] += adj * partials[j];
                            }
                        }
                    }
                    Node::Const(_) | Node::Input(_) => {}
                }
            }

            let grads = self
                .inputs
                .iter()
                .map(|id| tape.adjoints[id.index])
                .collect::<Vec<_>>();
            results.push((tape.primals[output_id.index], grads));
        }

        results
    }

    pub fn compute_reverse_single(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.reverse_tape();
        self.compute_reverse_single_with_tape(inputs, &mut tape)
    }

    pub fn compute_reverse_single_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.compute_reverse_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    pub fn compute_reverse_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.compute_reverse_with_tape_named(inputs, &mut tape)
    }

    pub fn compute_reverse_with_tape_named(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.compute_reverse_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    pub fn compute_reverse_named_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.compute_reverse_with_tape_named_for(inputs, outputs, &mut tape)
    }

    pub fn compute_reverse_with_tape_named_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.compute_reverse_with_tape_for(inputs, outputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }
}

impl Default for MultiGraph {
    fn default() -> Self {
        Self::new()
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
        EvalTape::with_capacity(self.ops.len() + 1, 1, 1)
    }

    pub fn compute(&self, input: Float) -> (Float, Float) {
        let mut tape = self.tape();
        self.compute_with_tape(input, &mut tape)
    }

    pub fn compute_with_tape(&self, input: Float, tape: &mut EvalTape) -> (Float, Float) {
        tape.reset(self.ops.len() + 1, 1, 1);

        tape.primals[0] = input;
        let idx = tape.tangent_index(0, 0);
        tape.tangents[idx] = 1.0;

        for (i, op) in self.ops.iter().enumerate() {
            let primal = op.compute(&[tape.primals[i]]);
            let tangent = tape.tangents[tape.tangent_index(i, 0)]
                * op.compute_derivative(&[tape.primals[i]], 0);

            tape.primals[i + 1] = primal;
            let idx = tape.tangent_index(i + 1, 0);
            tape.tangents[idx] = tangent;
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

    pub fn reverse_tape(&self) -> ReverseTape {
        ReverseTape::with_capacity(self.ops.len() + 1, 1)
    }

    pub fn compute_reverse(&self, input: Float) -> (Float, Float) {
        let mut tape = self.reverse_tape();
        self.compute_reverse_with_tape(input, &mut tape)
    }

    pub fn compute_reverse_with_tape(
        &self,
        input: Float,
        tape: &mut ReverseTape,
    ) -> (Float, Float) {
        tape.reset(self.ops.len() + 1, 1);

        tape.primals[0] = input;
        for (i, op) in self.ops.iter().enumerate() {
            tape.primals[i + 1] = op.compute(&[tape.primals[i]]);
        }

        tape.adjoints.fill(0.0);
        tape.adjoints[self.ops.len()] = 1.0;

        for (i, op) in self.ops.iter().enumerate().rev() {
            let partial = op.compute_derivative(&[tape.primals[i]], 0);
            tape.adjoints[i] += tape.adjoints[i + 1] * partial;
        }

        (
            *tape
                .primals
                .last()
                .expect("tape primals should have at least one element"),
            tape.adjoints[0],
        )
    }
}

#[derive(Debug, Clone)]
pub struct Gradients {
    pub value: Float,
    pub grads: Vec<(String, Float)>,
}

impl Gradients {
    pub fn get(&self, name: &str) -> Option<Float> {
        self.grads
            .iter()
            .find_map(|(key, value)| (key == name).then_some(*value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapeError {
    InputLengthMismatch { expected: usize, got: usize },
    UnknownInput(String),
}

impl std::fmt::Display for TapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputLengthMismatch { expected, got } => {
                write!(f, "expected {expected} inputs, got {got}")
            }
            Self::UnknownInput(name) => write!(f, "unknown input name: {name}"),
        }
    }
}

impl std::error::Error for TapeError {}

/// Rust-like autodiff tape with operator overloading.
#[derive(Debug, Clone)]
pub struct Tape {
    inner: Rc<RefCell<TapeInner>>,
}

#[derive(Debug)]
struct TapeInner {
    graph: MultiGraph,
    values: Vec<Float>,
}

/// A node handle tied to a [`Tape`].
#[derive(Debug, Clone)]
pub struct Var {
    id: NodeId,
    inner: Rc<RefCell<TapeInner>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TapeInner {
                graph: MultiGraph::new(),
                values: Vec::new(),
            })),
        }
    }

    pub fn input(&mut self, name: impl Into<String>, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.input(name.into());
        inner.values.push(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    pub fn input_unnamed(&mut self, value: Float) -> Var {
        let idx = self.inner.borrow().values.len();
        self.input(format!("_{}", idx), value)
    }

    pub fn constant(&mut self, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.constant(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    pub fn set_inputs(&mut self, values: &[Float]) {
        self.try_set_inputs(values)
            .expect("input length mismatch for Tape::set_inputs");
    }

    pub fn try_set_inputs(&mut self, values: &[Float]) -> Result<(), TapeError> {
        let mut inner = self.inner.borrow_mut();
        let expected = inner.values.len();
        if values.len() != expected {
            return Err(TapeError::InputLengthMismatch {
                expected,
                got: values.len(),
            });
        }
        inner.values.copy_from_slice(values);
        Ok(())
    }

    pub fn set(&mut self, name: &str, value: Float) {
        self.try_set(name, value)
            .expect("unknown input name for Tape::set");
    }

    pub fn try_set(&mut self, name: &str, value: Float) -> Result<(), TapeError> {
        let mut inner = self.inner.borrow_mut();
        let Some(idx) = inner.graph.input_names.iter().position(|n| n == name) else {
            return Err(TapeError::UnknownInput(name.to_string()));
        };
        inner.values[idx] = value;
        Ok(())
    }

    pub fn input_names(&self) -> Vec<String> {
        self.inner.borrow().graph.input_names.clone()
    }

    pub fn gradients(&self, output: &Var) -> Gradients {
        output.assert_same_tape(self);
        let inner = self.inner.borrow();
        let results = inner
            .graph
            .compute_reverse_named_for(&inner.values, &[output.id]);
        let (value, grads) = results.into_iter().next().expect("missing output");
        Gradients { value, grads }
    }

    pub fn gradients_for(&self, outputs: &[Var]) -> Vec<Gradients> {
        if outputs.is_empty() {
            return Vec::new();
        }
        outputs[0].assert_same_tape(self);
        for var in outputs.iter().skip(1) {
            var.assert_same_tape(self);
        }

        let inner = self.inner.borrow();
        let ids = outputs.iter().map(|var| var.id).collect::<Vec<_>>();
        inner
            .graph
            .compute_reverse_named_for(&inner.values, &ids)
            .into_iter()
            .map(|(value, grads)| Gradients { value, grads })
            .collect()
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Var {
    fn assert_same_tape(&self, tape: &Tape) {
        assert!(
            Rc::ptr_eq(&self.inner, &tape.inner),
            "cannot mix Vars from different tapes"
        );
    }

    fn assert_same_var_tape(&self, other: &Var) {
        assert!(
            Rc::ptr_eq(&self.inner, &other.inner),
            "cannot mix Vars from different tapes"
        );
    }

    fn unary_op(&self, op: Op) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.operation(op, vec![self.id]);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    fn binary_op(&self, rhs: &Var, op: Op) -> Var {
        self.assert_same_var_tape(rhs);
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.operation(op, vec![self.id, rhs.id]);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    fn konst(&self, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.constant(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    pub fn sin(&self) -> Var {
        self.unary_op(Op::Sin)
    }

    pub fn cos(&self) -> Var {
        self.unary_op(Op::Cos)
    }

    pub fn powi(&self, exp: i32) -> Var {
        self.unary_op(Op::Pow(exp))
    }

    pub fn scale(&self, factor: Float) -> Var {
        self.unary_op(Op::Scale(factor))
    }
}

impl std::ops::Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Self::Output {
        self.binary_op(&rhs, Op::Add)
    }
}

impl std::ops::Add<Float> for Var {
    type Output = Var;
    fn add(self, rhs: Float) -> Self::Output {
        let rhs = self.konst(rhs);
        self.binary_op(&rhs, Op::Add)
    }
}

impl std::ops::Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<Float> for Var {
    type Output = Var;
    fn sub(self, rhs: Float) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Self::Output {
        self.binary_op(&rhs, Op::Mul)
    }
}

impl std::ops::Mul<Float> for Var {
    type Output = Var;
    fn mul(self, rhs: Float) -> Self::Output {
        self.scale(rhs)
    }
}

impl std::ops::Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Self::Output {
        self * rhs.powi(-1)
    }
}

impl std::ops::Div<Float> for Var {
    type Output = Var;
    fn div(self, rhs: Float) -> Self::Output {
        self.scale(1.0 / rhs)
    }
}

impl std::ops::Neg for Var {
    type Output = Var;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
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
/// `compute_with_tape` to keep allocations off the hot path. Operation arity is
/// validated at runtime.
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
    (@build_linear [$($ops:expr,)*], Add -> $($rest:tt)*) => {
        compile_error!("Add is n-ary; use the multi-input graph form");
    };

    (@build_linear [$($ops:expr,)*], Mul -> $($rest:tt)*) => {
        compile_error!("Mul is n-ary; use the multi-input graph form");
    };

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

    (@build_multi $graph:ident, $node:ident -> Add -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add is n-ary; use (@a, @b, ...) -> Add");
    };

    (@build_multi $graph:ident, $node:ident -> Mul -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul is n-ary; use (@a, @b, ...) -> Mul");
    };

    (@build_multi $graph:ident, $node:ident -> Add ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add takes no arguments and is n-ary; use (@a, @b, ...) -> Add");
    };

    (@build_multi $graph:ident, $node:ident -> Mul ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul takes no arguments and is n-ary; use (@a, @b, ...) -> Mul");
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // Reject unary ops in n-ary position
    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Sin -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Sin is unary; use x -> Sin");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Cos -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Cos is unary; use x -> Cos");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Scale ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Scale is unary; use x -> Scale(factor)");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Pow ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Pow is unary; use x -> Pow(exp)");
    };

    // Generic N-ary op without extra args: (@a, @b, @c) -> Add -> @result
    (@build_multi $graph:ident, ( @ $node:ident ) -> Add -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add requires at least 2 inputs");
    };

    (@build_multi $graph:ident, ( @ $node:ident ) -> Mul -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul requires at least 2 inputs");
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Float, b: Float, eps: Float) {
        let diff = (a - b).abs();
        assert!(diff <= eps, "expected {a} ~= {b} (diff={diff}, eps={eps})");
    }

    #[test]
    fn reverse_matches_forward_and_finite_difference() {
        let mut g = MultiGraph::new();
        let x = g.input("x".to_string());
        let z = g.input("z".to_string());
        let x_sq = g.operation(Op::Pow(2), [x]);
        let z_cos = g.operation(Op::Cos, [z]);
        let sum = g.operation(Op::Add, [x_sq, z_cos]);
        let out = g.operation(Op::Sin, [sum]);
        g.output(out);

        let base = [1.3, -0.7];
        let (fwd_val, fwd_grad) = g.compute_single(&base);
        let (rev_val, rev_grad) = g.compute_reverse_single(&base);

        approx_eq(fwd_val, rev_val, 1e-12);
        approx_eq(fwd_grad[0], rev_grad[0], 1e-10);
        approx_eq(fwd_grad[1], rev_grad[1], 1e-10);

        let eps = 1e-7;
        for i in 0..base.len() {
            let mut plus = base;
            let mut minus = base;
            plus[i] += eps;
            minus[i] -= eps;
            let f_plus = g.compute_single(&plus).0;
            let f_minus = g.compute_single(&minus).0;
            let numeric = (f_plus - f_minus) / (2.0 * eps);
            approx_eq(rev_grad[i], numeric, 1e-6);
        }
    }

    #[test]
    fn output_rejects_foreign_node_id() {
        let mut g1 = MultiGraph::new();
        let foreign = g1.input("x".to_string());

        let mut g2 = MultiGraph::new();
        let _ = g2.input("y".to_string());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g2.output(foreign);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn tape_try_set_variants() {
        let mut tape = Tape::new();
        let x = tape.input("x", 1.0);
        let y = tape.input("y", 2.0);
        let out = x + y;

        tape.try_set_inputs(&[3.0, 4.0])
            .expect("valid input update");
        let grads = tape.gradients(&out);
        approx_eq(grads.value, 7.0, 1e-12);

        let err = tape
            .try_set_inputs(&[1.0])
            .expect_err("length mismatch should fail");
        assert!(matches!(
            err,
            TapeError::InputLengthMismatch {
                expected: 2,
                got: 1
            }
        ));

        tape.try_set("x", 5.0).expect("known input should be set");
        let err = tape
            .try_set("missing", 0.0)
            .expect_err("unknown input should fail");
        assert!(matches!(err, TapeError::UnknownInput(_)));
    }
}
