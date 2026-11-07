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
use crate::ir::{Graph, Node, Op, SRef, ScalarGraph, ScalarNode};
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
fn eligible(g: &Graph, n: &Node, max_elems: i64) -> bool {
    let small = |v: usize| is_small(g, v, max_elems);
    let outs_small = n.outputs.iter().all(|&o| small(o));
    match &n.op {
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
pub fn form_segments(g: Graph, max_elems: i64, fastmath: bool) -> Graph {
    let elig: Vec<bool> = g.nodes.iter().map(|n| eligible(&g, n, max_elems)).collect();
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
        let worth_it = run.iter().filter(|&&i| !matches!(g.nodes[i].op, Op::DimShuffle { .. })).count() >= 2;
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
            if matches!(n.op, Op::DimShuffle { .. }) && !used_inside(n.outputs[0]) {
                moved_out.push(i);
            } else {
                run_in.push(i);
            }
        }
        let used_outside = |v: usize| {
            used_outside(v) || moved_out.iter().any(|&m| g.nodes[m].inputs.contains(&v))
        };
        if run_in.iter().filter(|&&i| !matches!(g.nodes[i].op, Op::DimShuffle { .. })).count() < 2 {
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
            let is_view = matches!(n.op, Op::DimShuffle { .. } | Op::Assert { .. });
            let mut outs = Vec::new();
            for &o in &n.outputs {
                let loc = if is_view {
                    let order = match &n.op {
                        Op::DimShuffle { order } => order.clone(),
                        _ => (0..vars[o].ndim()).map(Some).collect(),
                    };
                    SegLoc::View { base: ins[0], order }
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
            let layout_of = |x: usize, values: &Vec<SegValue>| match values[x].loc {
                SegLoc::Temp | SegLoc::ExtOut(_) => Layout::C.normalize(values[x].ndim),
                _ => Layout::Any,
            };
            let operand = |x: usize, v: usize, values: &Vec<SegValue>| Operand {
                dtype: vars[v].dtype,
                bcast: vars[v].bcast.clone(),
                layout: layout_of(x, values),
            };
            match &n.op {
                Op::Elemwise { scalar } => {
                    let spec = KernelSpec::Elemwise {
                        ndim: vars[n.outputs[0]].ndim(),
                        inputs: n.inputs.iter().zip(&ins).map(|(&v, &x)| operand(x, v, &values)).collect(),
                        outputs: n.outputs.iter().zip(&outs).map(|(&v, &x)| operand(x, v, &values)).collect(),
                        inplace: vec![None; n.outputs.len()],
                        scalar: scalar.clone(),
                        fastmath,
                    };
                    let mut args = ins.clone();
                    args.extend(&outs);
                    seg_members.push(SegMember::Kernel { spec: Box::new(spec), args });
                }
                Op::Reduce { op, axes } => {
                    let spec = KernelSpec::Reduce {
                        op: *op,
                        axes: axes.clone(),
                        input: operand(ins[0], n.inputs[0], &values),
                        output: operand(outs[0], n.outputs[0], &values),
                        fastmath,
                    };
                    seg_members.push(SegMember::Kernel { spec: Box::new(spec), args: vec![ins[0], outs[0]] });
                }
                Op::ShapeI { i } => seg_members.push(SegMember::ShapeI { i: *i, input: ins[0], output: outs[0] }),
                Op::MakeVector => seg_members.push(SegMember::MakeVector { inputs: ins.clone(), output: outs[0] }),
                Op::Assert { .. } => seg_members.push(SegMember::Assert { conds: ins[1..].to_vec() }),
                Op::DimShuffle { .. } => {}
                _ => unreachable!("not eligible"),
            }
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
        let descs: Vec<String> = run_in
            .iter()
            .filter(|&&i| !matches!(g.nodes[i].op, Op::DimShuffle { .. }))
            .map(|&i| node_desc(&g.nodes[i]))
            .chain(std::iter::repeat_n("copy".to_string(), seg_members.len()))
            .take(seg_members.len())
            .collect();
        let spec = SegmentSpec { n_in: ext_in.len(), n_out: ext_out.len(), values, members: seg_members, fastmath };
        new_nodes.push(Node { op: Op::Segment { spec: Box::new(spec), descs }, inputs: ext_in, outputs: ext_out });
        new_nodes.extend(moved_out.iter().map(|&i| g.nodes[i].clone()));
    }
    let _ = &mut vars;
    Graph { nodes: new_nodes, ..g }
}

fn node_desc(n: &Node) -> String {
    match &n.op {
        Op::Elemwise { scalar } => {
            let ops: Vec<&str> = scalar.nodes.iter().map(|x| x.op.as_str()).collect();
            format!("Elemwise{{{}}}", ops.join(","))
        }
        Op::Reduce { op, .. } => format!("Reduce{{{op:?}}}"),
        Op::ShapeI { i } => format!("Shape_i{{{i}}}"),
        Op::MakeVector => "MakeVector".into(),
        Op::Assert { msg } => format!("Assert: {msg}"),
        other => format!("{other:?}").chars().take(40).collect(),
    }
}
