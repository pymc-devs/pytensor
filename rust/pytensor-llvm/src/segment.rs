//! Segments: runs of statically small nodes compiled as one unit so LLVM can inline the
//! member kernels into each other.
//!
//! A node is *small* if every array it reads data from and every output has a static
//! shape with at most `max_elems` elements (`Shape_i` only reads its input's shape, so its
//! input may be anything). The pre-pass
//!
//! 1. reorders the schedule (still topological) so small nodes cluster,
//! 2. merges runs of >= 2 small nodes into one `Op::Segment` node,
//!
//! Inside a segment, intermediate values live on the stack and are promoted to registers
//! after inlining; values used outside become outputs of the segment node and get normal
//! planner storage. Views (DimShuffle, Assert) are free inside a segment. A view used only
//! outside the segment is moved after it; a view used both inside and outside is
//! materialized into a (small) segment output by a copy.

use crate::dtype::DType;
use crate::ir::{FusedOut, Graph, IndexEntry, Node, Op, SRef, ScalarGraph, ScalarNode};
use crate::spec::{KernelSpec, Layout, Operand};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Clone, Debug)]
pub enum SegLoc {
    /// external input i of the segment (args[i])
    ExtIn(usize),
    /// external output k of the segment (args[n_in + k]), provided by the driver
    ExtOut(usize),
    /// stack temporary with a static shape
    Temp,
    /// view of another segment value
    View { base: usize, order: Vec<Option<usize>> },
    /// basic indexing of another value with static indices, one entry per base dim
    Slice { base: usize, dims: Vec<SliceDim> },
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub enum SliceDim {
    /// integer index: drops the dim
    Index(i64),
    /// start and step of a slice; the length is the value's static shape
    Range { start: i64, step: i64 },
}

/// Scalar integer constants of a graph (var -> value), for static indexing.
pub type ConstInts = HashMap<usize, i64>;

/// Static resolution of basic indexing (Python semantics): one `SliceDim` per base
/// dim, or None if an index isn't constant, a shape isn't static or it's out of bounds.
fn static_slice(g: &Graph, n: &Node, consts: &ConstInts) -> Option<Vec<SliceDim>> {
    match &n.op {
        Op::Subtensor { entries } => resolve_static(g, n.inputs[0], entries, &n.inputs, static_shape(g, n.outputs[0])?, consts),
        Op::IncSubtensor { entries, .. } => {
            // the region y is written to: y's shape after dropping leading broadcast dims
            let region = region_shape(g, n.inputs[0], entries, &n.inputs, consts)?;
            resolve_static(g, n.inputs[0], entries, &n.inputs, region, consts)
        }
        _ => None,
    }
}

/// Shape of x[entries] (static), for IncSubtensor's target region.
fn region_shape(g: &Graph, x: usize, entries: &[IndexEntry], inputs: &[usize], consts: &ConstInts) -> Option<Vec<i64>> {
    let shape = static_shape(g, x)?;
    let mut out = Vec::new();
    for (d, &len) in shape.iter().enumerate() {
        match entries.get(d) {
            None => out.push(len),
            Some(IndexEntry::Int(_)) => {}
            Some(e @ IndexEntry::Slice { .. }) => out.push(slice_len(len, e, inputs, consts)?.2),
        }
    }
    Some(out)
}

/// (start, step, count) of a slice entry on a dim of length `len` (Python semantics).
fn slice_len(len: i64, e: &IndexEntry, inputs: &[usize], consts: &ConstInts) -> Option<(i64, i64, i64)> {
    let IndexEntry::Slice { start, stop, step } = e else { return None };
    let val = |pos: usize| consts.get(&inputs[pos]).copied();
    let step = match step { Some(p) => val(*p)?, None => 1 };
    if step == 0 {
        return None;
    }
    let norm = |v: i64, lo: i64, hi: i64| (if v < 0 { v + len } else { v }).clamp(lo, hi);
    let opt = |p: &Option<usize>| -> Option<Option<i64>> { match p { Some(p) => Some(Some(val(*p)?)), None => Some(None) } };
    let (a, b) = (opt(start)?, opt(stop)?);
    let (s, en) = if step > 0 {
        (a.map_or(0, |v| norm(v, 0, len)), b.map_or(len, |v| norm(v, 0, len)))
    } else {
        (a.map_or(len - 1, |v| norm(v, -1, len - 1)), b.map_or(-1, |v| norm(v, -1, len - 1)))
    };
    let count = if step > 0 { (en - s + step - 1).div_euclid(step) } else { (s - en - step - 1).div_euclid(-step) }.max(0);
    Some((if count > 0 { s } else { 0 }, step, count))
}

/// x[entries] with constant indices as one `SliceDim` per dim of x, if its shape is
/// `expect`.
fn resolve_static(g: &Graph, x: usize, entries: &[IndexEntry], inputs: &[usize], expect: Vec<i64>, consts: &ConstInts) -> Option<Vec<SliceDim>> {
    let shape = static_shape(g, x)?;
    let out_shape = expect;
    let val = |pos: usize| consts.get(&inputs[pos]).copied();
    let mut dims = Vec::new();
    let mut out_lens = Vec::new();
    for (d, &len) in shape.iter().enumerate() {
        match entries.get(d) {
            None => {
                dims.push(SliceDim::Range { start: 0, step: 1 });
                out_lens.push(len);
            }
            Some(IndexEntry::Int(p)) => {
                let i = val(*p)?;
                let i = if i < 0 { i + len } else { i };
                if i < 0 || i >= len {
                    return None;
                }
                dims.push(SliceDim::Index(i));
            }
            Some(e @ IndexEntry::Slice { .. }) => {
                let (start, step, count) = slice_len(len, e, inputs, consts)?;
                dims.push(SliceDim::Range { start, step });
                out_lens.push(count);
            }
        }
    }
    (out_lens == out_shape).then_some(dims)
}

/// Segment location of a node's output if the node is a view (no member needed).
fn view_loc(g: &Graph, n: &Node, ins: &[usize], o: usize, consts: &ConstInts) -> Option<SegLoc> {
    match &n.op {
        Op::DimShuffle { order } => Some(SegLoc::View { base: ins[0], order: order.clone() }),
        Op::Assert { .. } => Some(SegLoc::View { base: ins[0], order: (0..g.vars[o].ndim()).map(Some).collect() }),
        Op::Subtensor { .. } => Some(SegLoc::Slice { base: ins[0], dims: static_slice(g, n, consts)? }),
        _ => None,
    }
}

/// Views without any member code (Assert is a view with a check).
fn is_pure_view(op: &Op) -> bool {
    matches!(op, Op::DimShuffle { .. } | Op::Subtensor { .. })
}

fn is_view_op(op: &Op) -> bool {
    matches!(op, Op::DimShuffle { .. } | Op::Assert { .. } | Op::Subtensor { .. })
}

#[derive(Serialize, Clone, Debug)]
pub struct SegValue {
    pub dtype: DType,
    pub ndim: usize,
    /// Static shape; `None` only for external inputs that are just shape-queried.
    pub shape: Option<Vec<i64>>,
    pub loc: SegLoc,
}

#[derive(Serialize, Clone, Debug)]
pub enum SegMember {
    /// A generated Elemwise or Reduce kernel; args = inputs ++ outputs (value ids).
    Kernel { spec: Box<KernelSpec>, args: Vec<usize> },
    ShapeI { i: usize, input: usize, output: usize },
    MakeVector { inputs: Vec<usize>, output: usize },
    /// Fails (segment returns member index + 1) if any 0-d condition is false.
    Assert { conds: Vec<usize> },
}

#[derive(Serialize, Clone, Debug)]
pub struct SegmentSpec {
    pub values: Vec<SegValue>,
    pub members: Vec<SegMember>,
    pub n_in: usize,
    pub n_out: usize,
    pub fastmath: bool,
}

impl SegmentSpec {
    pub fn out_shape(&self, k: usize) -> &[i64] {
        self.values
            .iter()
            .find(|v| matches!(v.loc, SegLoc::ExtOut(j) if j == k))
            .and_then(|v| v.shape.as_deref())
            .expect("segment outputs have static shapes")
    }
}

fn static_shape(g: &Graph, v: usize) -> Option<Vec<i64>> {
    let info = &g.vars[v];
    (0..info.ndim())
        .map(|d| if info.bcast[d] { Some(1) } else { info.static_shape.get(d).copied().flatten() })
        .collect()
}

fn is_small(g: &Graph, v: usize, max_elems: i64) -> bool {
    static_shape(g, v).is_some_and(|s| s.iter().product::<i64>() <= max_elems)
}

/// Can this node be a segment member?
fn eligible(g: &Graph, n: &Node, max_elems: i64, consts: &ConstInts) -> bool {
    let small = |v: usize| is_small(g, v, max_elems);
    let outs_small = n.outputs.iter().all(|&o| small(o));
    match &n.op {
        // a view: the base can be large (e.g. the joined parameter vector)
        Op::Subtensor { .. } => outs_small && static_slice(g, n, consts).is_some(),
        Op::Join { .. } => outs_small && n.inputs.iter().all(|&i| small(i)),
        Op::IncSubtensor { .. } => {
            outs_small && small(n.inputs[1]) && static_slice(g, n, consts).is_some_and(|_| {
                // y must broadcast into the region
                let region = region_shape(g, n.inputs[0], match &n.op { Op::IncSubtensor { entries, .. } => entries, _ => unreachable!() }, &n.inputs, consts).unwrap();
                let y = static_shape(g, n.inputs[1]).unwrap();
                y.len() <= region.len() && y.iter().rev().zip(region.iter().rev()).all(|(&a, &b)| a == b || a == 1)
            })
        }
        // indexed writes update their target in place: not as a member
        Op::FusedElemwise { outs, .. } => {
            outs.iter().all(|o| !matches!(o, FusedOut::Write { .. })) && outs_small && n.inputs.iter().all(|&i| small(i))
        }
        Op::Alloc => outs_small && static_shape(g, n.inputs[0]).is_some(),
        Op::Elemwise { .. } | Op::Reduce { .. } | Op::DimShuffle { .. } | Op::MakeVector => {
            outs_small && n.inputs.iter().all(|&i| small(i))
        }
        Op::ShapeI { .. } => true,
        Op::Assert { .. } => outs_small && n.inputs.iter().all(|&i| small(i)),
        _ => false,
    }
}

/// Topological order that prefers continuing the current run (small or not), so small
/// nodes end up adjacent. Ties keep the original order.
fn cluster_order(g: &Graph, elig: &[bool]) -> Vec<usize> {
    let nn = g.nodes.len();
    let mut producer: HashMap<usize, usize> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &o in &n.outputs {
            producer.insert(o, i);
        }
    }
    let mut indeg = vec![0usize; nn];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); nn];
    for (i, n) in g.nodes.iter().enumerate() {
        let deps: HashSet<usize> = n.inputs.iter().filter_map(|v| producer.get(v).copied()).collect();
        indeg[i] = deps.len();
        for d in deps {
            users[d].push(i);
        }
    }
    let mut ready: Vec<usize> = (0..nn).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(nn);
    let mut want_small = true;
    while !ready.is_empty() {
        ready.sort_unstable();
        let pos = ready.iter().position(|&i| elig[i] == want_small).unwrap_or(0);
        let i = ready.remove(pos);
        want_small = elig[i];
        order.push(i);
        for &u in &users[i] {
            indeg[u] -= 1;
            if indeg[u] == 0 {
                ready.push(u);
            }
        }
    }
    order
}

fn identity_copy(dt: DType) -> ScalarGraph {
    ScalarGraph {
        inputs: vec![dt],
        nodes: vec![ScalarNode { op: "identity".into(), inputs: vec![SRef::In(0)], dtype: dt }],
        outputs: vec![SRef::Node(0)],
    }
}

/// Rewrite `g` so runs of small nodes become `Op::Segment` nodes.
pub fn form_segments(g: Graph, max_elems: i64, fastmath: bool, consts: &ConstInts) -> Graph {
    let elig: Vec<bool> = g.nodes.iter().map(|n| eligible(&g, n, max_elems, consts)).collect();
    let order = cluster_order(&g, &elig);
    let graph_outputs: HashSet<usize> = g.outputs.iter().copied().collect();

    // runs of eligible nodes in the new order
    let mut runs: Vec<(bool, Vec<usize>)> = Vec::new();
    for &i in &order {
        match runs.last_mut() {
            Some((e, r)) if *e == elig[i] => r.push(i),
            _ => runs.push((elig[i], vec![i])),
        }
    }

    // users of every var, by node index
    let mut users: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &v in &n.inputs {
            users.entry(v).or_default().push(i);
        }
    }

    let mut new_nodes: Vec<Node> = Vec::new();
    let mut vars = g.vars.clone();
    for (e, run) in runs {
        let worth_it = run.iter().filter(|&&i| !is_pure_view(&g.nodes[i].op)).count() >= 2;
        if !e || !worth_it {
            new_nodes.extend(run.iter().map(|&i| g.nodes[i].clone()));
            continue;
        }
        let members: HashSet<usize> = run.iter().copied().collect();
        let used_inside = |v: usize| users.get(&v).is_some_and(|u| u.iter().any(|x| members.contains(x)));
        let used_outside =
            |v: usize| graph_outputs.contains(&v) || users.get(&v).is_some_and(|u| u.iter().any(|x| !members.contains(x)));

        // views used only outside move after the segment
        let mut moved_out: Vec<usize> = Vec::new();
        let mut run_in: Vec<usize> = Vec::new();
        for &i in &run {
            let n = &g.nodes[i];
            if is_pure_view(&n.op) && !used_inside(n.outputs[0]) {
                moved_out.push(i);
            } else {
                run_in.push(i);
            }
        }
        let used_outside = |v: usize| {
            used_outside(v) || moved_out.iter().any(|&m| g.nodes[m].inputs.contains(&v))
        };
        if run_in.iter().filter(|&&i| !is_pure_view(&g.nodes[i].op)).count() < 2 {
            new_nodes.extend(run.iter().map(|&i| g.nodes[i].clone()));
            continue;
        }

        let mut produced: HashSet<usize> = HashSet::new();
        for &i in &run_in {
            produced.extend(&g.nodes[i].outputs);
        }
        let mut val_of: HashMap<usize, usize> = HashMap::new();
        let mut values: Vec<SegValue> = Vec::new();
        let mut ext_in: Vec<usize> = Vec::new();
        let mut ext_out: Vec<usize> = Vec::new(); // graph vars that are segment outputs
        let mut seg_members: Vec<SegMember> = Vec::new();
        let mut copies: Vec<(usize, usize)> = Vec::new(); // (view value, graph var to materialize)
        let mut descs: Vec<String> = Vec::new();

        let get_in = |v: usize, values: &mut Vec<SegValue>, val_of: &mut HashMap<usize, usize>, ext_in: &mut Vec<usize>| -> usize {
            if let Some(&x) = val_of.get(&v) {
                return x;
            }
            ext_in.push(v);
            values.push(SegValue {
                dtype: vars[v].dtype,
                ndim: vars[v].ndim(),
                shape: static_shape(&g, v),
                loc: SegLoc::ExtIn(ext_in.len() - 1),
            });
            val_of.insert(v, values.len() - 1);
            values.len() - 1
        };

        for &i in &run_in {
            let n = &g.nodes[i];
            let ins: Vec<usize> = n.inputs.iter().map(|&v| get_in(v, &mut values, &mut val_of, &mut ext_in)).collect();
            // define outputs
            let is_view = is_view_op(&n.op);
            let mut outs = Vec::new();
            for &o in &n.outputs {
                let loc = if is_view {
                    view_loc(&g, n, &ins, o, consts).expect("eligible view")
                } else if used_outside(o) {
                    ext_out.push(o);
                    SegLoc::ExtOut(ext_out.len() - 1)
                } else {
                    SegLoc::Temp
                };
                values.push(SegValue { dtype: vars[o].dtype, ndim: vars[o].ndim(), shape: static_shape(&g, o), loc });
                val_of.insert(o, values.len() - 1);
                if is_view && used_outside(o) {
                    copies.push((values.len() - 1, o));
                }
                outs.push(values.len() - 1);
            }
            let before = seg_members.len();
            push_member(&g, n, &ins, &outs, &mut values, fastmath, consts, &mut seg_members);
            descs.extend(std::iter::repeat_n(node_desc(n), seg_members.len() - before));
        }
        // views needed outside: copy into a segment output
        for (x, o) in copies {
            ext_out.push(o);
            let k = ext_out.len() - 1;
            let shape = static_shape(&g, o);
            values.push(SegValue { dtype: vars[o].dtype, ndim: vars[o].ndim(), shape, loc: SegLoc::ExtOut(k) });
            let y = values.len() - 1;
            let spec = KernelSpec::Elemwise {
                ndim: vars[o].ndim(),
                inputs: vec![Operand { dtype: vars[o].dtype, bcast: vars[o].bcast.clone(), layout: Layout::Any }],
                outputs: vec![Operand { dtype: vars[o].dtype, bcast: vars[o].bcast.clone(), layout: Layout::C.normalize(vars[o].ndim()) }],
                inplace: vec![None],
                scalar: identity_copy(vars[o].dtype),
                fastmath,
            };
            seg_members.push(SegMember::Kernel { spec: Box::new(spec), args: vec![x, y] });
        }
        descs.resize(seg_members.len(), "copy".to_string());
        let spec = SegmentSpec { n_in: ext_in.len(), n_out: ext_out.len(), values, members: seg_members, fastmath };
        new_nodes.push(Node { op: Op::Segment { spec: Box::new(spec), descs }, inputs: ext_in, outputs: ext_out });
        new_nodes.extend(moved_out.iter().map(|&i| g.nodes[i].clone()));
    }
    let _ = &mut vars;
    Graph { nodes: new_nodes, ..g }
}


/// The segment member for an eligible node (none for pure views).
fn push_member(g: &Graph, n: &Node, ins: &[usize], outs: &[usize], values: &mut Vec<SegValue>, fastmath: bool, consts: &ConstInts, seg_members: &mut Vec<SegMember>) {
    let vars = &g.vars;
    let layout_of = |values: &[SegValue], x: usize| match values[x].loc {
        SegLoc::Temp | SegLoc::ExtOut(_) => Layout::C.normalize(values[x].ndim),
        _ => Layout::Any,
    };
    let operand = |x: usize, v: usize| Operand { dtype: vars[v].dtype, bcast: vars[v].bcast.clone(), layout: layout_of(values, x) };
    let copy_member = |x: usize, y: usize, in_bcast: Vec<bool>, values: &[SegValue], info: &crate::ir::VarInfo| SegMember::Kernel {
        spec: Box::new(KernelSpec::Elemwise {
            ndim: info.ndim(),
            inputs: vec![Operand { dtype: info.dtype, bcast: in_bcast, layout: Layout::Any }],
            outputs: vec![Operand { dtype: info.dtype, bcast: info.bcast.clone(), layout: layout_of(values, y) }],
            inplace: vec![None],
            scalar: identity_copy(info.dtype),
            fastmath,
        }),
        args: vec![x, y],
    };
    match &n.op {
        Op::Subtensor { .. } => {}
        Op::FusedElemwise { scalar, ndim, n_elem, reads, outs: fouts } => {
            let spec = KernelSpec::Fused {
                ndim: *ndim,
                inputs: n.inputs.iter().zip(ins).map(|(&v, &x)| operand(x, v)).collect(),
                n_elem: *n_elem,
                reads: reads.clone(),
                outputs: n.outputs.iter().zip(outs).map(|(&v, &x)| operand(x, v)).collect(),
                outs: fouts.clone(),
                dim_src: (0..*ndim).map(|d| crate::runtime::fused_dim_candidates(g, n, d).first().copied()).collect(),
                scalar: scalar.clone(),
                fastmath,
            };
            let mut args = ins.to_vec();
            args.extend(outs);
            seg_members.push(SegMember::Kernel { spec: Box::new(spec), args });
        }
        Op::IncSubtensor { set, .. } => {
            // out = copy of x; then out[region] (+)= y, broadcasting y
            let info = &vars[n.outputs[0]];
            let out = outs[0];
            seg_members.push(copy_member(ins[0], out, vars[n.inputs[0]].bcast.clone(), values, info));
            let dims = static_slice(g, n, consts).expect("eligible");
            let Op::IncSubtensor { entries, .. } = &n.op else { unreachable!() };
            let region = region_shape(g, n.inputs[0], entries, &n.inputs, consts).expect("eligible");
            let rd = region.len();
            values.push(SegValue { dtype: info.dtype, ndim: rd, shape: Some(region.clone()), loc: SegLoc::Slice { base: out, dims } });
            let target = values.len() - 1;
            let y = ins[1];
            let yshape = values[y].shape.clone().expect("static");
            let lead = rd - yshape.len();
            let mut yfull = vec![1; lead];
            yfull.extend(&yshape);
            let order = (0..rd).map(|d| (d >= lead).then(|| d - lead)).collect();
            values.push(SegValue { dtype: info.dtype, ndim: rd, shape: Some(yfull.clone()), loc: SegLoc::View { base: y, order } });
            let yv = values.len() - 1;
            let ybc: Vec<bool> = yfull.iter().zip(&region).map(|(&a, &b)| a == 1 && b != 1).collect();
            let tbc: Vec<bool> = region.iter().map(|&b| b == 1).collect();
            let ybc_op: Vec<bool> = ybc.iter().zip(&tbc).map(|(&a, &t)| a || t).collect();
            let rinfo = crate::ir::VarInfo { dtype: info.dtype, bcast: tbc.clone(), static_shape: region.iter().map(|&x| Some(x)).collect() };
            if *set {
                seg_members.push(copy_member(yv, target, ybc_op, values, &rinfo));
            } else {
                // target = target + y, elementwise on exactly the same memory
                let dt = info.dtype;
                let add = ScalarGraph {
                    inputs: vec![dt, dt],
                    nodes: vec![ScalarNode { op: "add".into(), inputs: vec![SRef::In(0), SRef::In(1)], dtype: dt }],
                    outputs: vec![SRef::Node(0)],
                };
                let spec = KernelSpec::Elemwise {
                    ndim: rd,
                    inputs: vec![
                        Operand { dtype: dt, bcast: tbc.clone(), layout: Layout::Any },
                        Operand { dtype: dt, bcast: ybc_op, layout: Layout::Any },
                    ],
                    outputs: vec![Operand { dtype: dt, bcast: tbc, layout: Layout::Any }],
                    inplace: vec![None],
                    scalar: add,
                    fastmath,
                };
                seg_members.push(SegMember::Kernel { spec: Box::new(spec), args: vec![target, yv, target] });
            }
        }
        Op::Join { axis } => {
            // copy every input into its slice of the output
            let info = &vars[n.outputs[0]];
            let out = outs[0];
            let mut off = 0;
            for (&x, &v) in ins.iter().zip(&n.inputs) {
                let shape = values[x].shape.clone().expect("join inputs have static shapes");
                let len = shape[*axis];
                if len > 0 {
                    let dims = (0..info.ndim()).map(|d| SliceDim::Range { start: if d == *axis { off } else { 0 }, step: 1 }).collect();
                    values.push(SegValue { dtype: info.dtype, ndim: info.ndim(), shape: Some(shape), loc: SegLoc::Slice { base: out, dims } });
                    let y = values.len() - 1;
                    seg_members.push(copy_member(x, y, vars[v].bcast.clone(), values, &vars[v]));
                }
                off += len;
            }
        }
        Op::Alloc => {
            // broadcast copy of the value; leading new dims through a view
            let info = &vars[n.outputs[0]];
            let (x, nd) = (ins[0], info.ndim());
            let vshape = values[x].shape.clone().expect("alloc value has a static shape");
            let lead = nd - vshape.len();
            let mut shape = vec![1; lead];
            shape.extend(&vshape);
            let order = (0..nd).map(|d| (d >= lead).then(|| d - lead)).collect();
            values.push(SegValue { dtype: info.dtype, ndim: nd, shape: Some(shape.clone()), loc: SegLoc::View { base: x, order } });
            let xv = values.len() - 1;
            let out_shape = values[outs[0]].shape.clone().expect("alloc output has a static shape");
            let bcast = shape.iter().zip(&out_shape).map(|(&a, &b)| a == 1 && b != 1).collect();
            seg_members.push(copy_member(xv, outs[0], bcast, values, info));
        }
        Op::Elemwise { scalar } => {
            let spec = KernelSpec::Elemwise {
                ndim: vars[n.outputs[0]].ndim(),
                inputs: n.inputs.iter().zip(ins).map(|(&v, &x)| operand(x, v)).collect(),
                outputs: n.outputs.iter().zip(outs).map(|(&v, &x)| operand(x, v)).collect(),
                inplace: vec![None; n.outputs.len()],
                scalar: scalar.clone(),
                fastmath,
            };
            let mut args = ins.to_vec();
            args.extend(outs);
            seg_members.push(SegMember::Kernel { spec: Box::new(spec), args });
        }
        Op::Reduce { op, axes } => {
            let spec = KernelSpec::Reduce {
                op: *op,
                axes: axes.clone(),
                input: operand(ins[0], n.inputs[0]),
                output: operand(outs[0], n.outputs[0]),
                fastmath,
            };
            seg_members.push(SegMember::Kernel { spec: Box::new(spec), args: vec![ins[0], outs[0]] });
        }
        Op::ShapeI { i } => seg_members.push(SegMember::ShapeI { i: *i, input: ins[0], output: outs[0] }),
        Op::MakeVector => seg_members.push(SegMember::MakeVector { inputs: ins.to_vec(), output: outs[0] }),
        Op::Assert { .. } => seg_members.push(SegMember::Assert { conds: ins[1..].to_vec() }),
        Op::DimShuffle { .. } => {}
        _ => unreachable!("not eligible"),
    }
}

pub(crate) fn identity_member(x: usize, y: usize, info: &crate::ir::VarInfo, fastmath: bool) -> SegMember {
    let spec = KernelSpec::Elemwise {
        ndim: info.ndim(),
        inputs: vec![Operand { dtype: info.dtype, bcast: info.bcast.clone(), layout: Layout::Any }],
        outputs: vec![Operand { dtype: info.dtype, bcast: info.bcast.clone(), layout: Layout::C.normalize(info.ndim()) }],
        inplace: vec![None],
        scalar: identity_copy(info.dtype),
        fastmath,
    };
    SegMember::Kernel { spec: Box::new(spec), args: vec![x, y] }
}

/// A whole graph as one segment, if every node is small (see `eligible`). External
/// inputs are the graph inputs followed by its constants; external outputs are the graph
/// outputs, in order. Used for Scan step functions.
pub fn whole_graph_segment(g: &Graph, max_elems: i64, fastmath: bool, consts: &ConstInts) -> Option<(SegmentSpec, Vec<String>)> {
    if !g.nodes.iter().all(|n| eligible(g, n, max_elems, consts)) {
        return None;
    }
    let vars = &g.vars;
    let mut values: Vec<SegValue> = Vec::new();
    let mut val_of: HashMap<usize, usize> = HashMap::new();
    for (k, &v) in g.inputs.iter().chain(&g.constants).enumerate() {
        values.push(SegValue { dtype: vars[v].dtype, ndim: vars[v].ndim(), shape: static_shape(g, v), loc: SegLoc::ExtIn(k) });
        val_of.insert(v, values.len() - 1);
    }
    // outputs written directly by their producing (non-view) node: first occurrence only
    let mut direct: HashMap<usize, usize> = HashMap::new();
    for (k, &o) in g.outputs.iter().enumerate() {
        let produced = g.nodes.iter().any(|n| n.outputs.contains(&o) && !is_view_op(&n.op));
        if produced && !direct.contains_key(&o) {
            direct.insert(o, k);
        }
    }
    let mut members = Vec::new();
    let mut descs = Vec::new();
    for n in &g.nodes {
        let ins: Vec<usize> = n.inputs.iter().map(|v| val_of.get(v).copied()).collect::<Option<_>>()?;
        let is_view = is_view_op(&n.op);
        let mut outs = Vec::new();
        for &o in &n.outputs {
            let loc = if is_view {
                view_loc(g, n, &ins, o, consts)?
            } else if let Some(&k) = direct.get(&o) {
                SegLoc::ExtOut(k)
            } else {
                SegLoc::Temp
            };
            values.push(SegValue { dtype: vars[o].dtype, ndim: vars[o].ndim(), shape: static_shape(g, o), loc });
            val_of.insert(o, values.len() - 1);
            outs.push(values.len() - 1);
        }
        let before = members.len();
        push_member(g, n, &ins, &outs, &mut values, fastmath, consts, &mut members);
        descs.extend(std::iter::repeat_n(node_desc(n), members.len() - before));
    }
    // outputs that are inputs, constants, views or repeated: copy into the output
    for (k, &o) in g.outputs.iter().enumerate() {
        if direct.get(&o) == Some(&k) {
            continue;
        }
        let x = *val_of.get(&o)?;
        let shape = static_shape(g, o)?;
        values.push(SegValue { dtype: vars[o].dtype, ndim: vars[o].ndim(), shape: Some(shape), loc: SegLoc::ExtOut(k) });
        members.push(identity_member(x, values.len() - 1, &vars[o], fastmath));
        descs.push("copy".into());
    }
    // every output must have a static shape (the caller provides stack-sized buffers)
    for &o in &g.outputs {
        static_shape(g, o)?;
    }
    let spec = SegmentSpec { n_in: g.inputs.len() + g.constants.len(), n_out: g.outputs.len(), values, members, fastmath };
    Some((spec, descs))
}

fn node_desc(n: &Node) -> String {
    match &n.op {
        Op::Elemwise { scalar } => {
            let ops: Vec<&str> = scalar.nodes.iter().map(|x| x.op.as_str()).collect();
            format!("Elemwise{{{}}}", ops.join(","))
        }
        Op::Reduce { op, .. } => format!("Reduce{{{op:?}}}"),
        Op::FusedElemwise { scalar, .. } => {
            let ops: Vec<&str> = scalar.nodes.iter().map(|x| x.op.as_str()).collect();
            format!("FusedElemwise{{{}}}", ops.join(","))
        }
        Op::Join { axis } => format!("Join{{{axis}}} (copy)"),
        Op::Alloc => "Alloc (broadcast copy)".into(),
        Op::ShapeI { i } => format!("Shape_i{{{i}}}"),
        Op::MakeVector => "MakeVector".into(),
        Op::Assert { msg } => format!("Assert: {msg}"),
        other => format!("{other:?}").chars().take(40).collect(),
    }
}
