//! OpFromGraph: a graph used as an op, compiled as one unit or interpreted.

use crate::v2::graph::Graph;
use crate::v2::index::IndexVec;
use crate::v2::interp::{Options, Program};
use crate::v2::op::{Generate, Kernel, Key, TensorOp, R};
use crate::v2::types::{InputIdx, OutputIdx, Signature};
use crate::v2::unit::{graph_key, ArrVal, EnvBuilder, UnitCx};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitKind {
    Compiled,
    Interpreted,
}

pub struct OpFromGraph {
    pub graph: Arc<Graph>,
    pub kind: UnitKind,
    pub opts: Options,
}

impl std::fmt::Debug for OpFromGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ops: Vec<String> = self.graph.nodes.iter().map(|n| format!("{:?}", n.op)).collect();
        write!(f, "OpFromGraph[{:?}]({})", self.kind, ops.join(", "))
    }
}

impl TensorOp for OpFromGraph {
    fn check(&self, sig: &Signature) -> R<()> {
        let g = &self.graph;
        if sig.inputs.len() != g.inputs.len() || sig.outputs.len() != g.outputs.len() {
            return Err("number of inputs or outputs differs from the inner graph".into());
        }
        for (i, is) in sig.inputs.iter_enumerated() {
            if &is.ty != g.ty(g.inputs[i]) {
                return Err(format!("input {} has type {:?}, the inner graph {:?}", i.0, is.ty, g.ty(g.inputs[i])));
            }
        }
        for (o, os) in sig.outputs.iter_enumerated() {
            if &os.ty != g.ty(g.outputs[o]) {
                return Err(format!("output {} has type {:?}, the inner graph {:?}", o.0, os.ty, g.ty(g.outputs[o])));
            }
        }
        Ok(())
    }

    fn kernel(&self, _sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        match self.kind {
            UnitKind::Interpreted => Ok(Some(Box::new(Program::build(self.graph.clone(), &self.opts)?))),
            UnitKind::Compiled => Ok(None),
        }
    }

    fn generate(&self) -> Option<&dyn Generate> {
        match self.kind {
            UnitKind::Compiled => Some(self),
            UnitKind::Interpreted => None,
        }
    }
}

impl Generate for OpFromGraph {
    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
        // graph_key only fails for graphs that couldn't be built
        let _ = graph_key(&self.graph, k);
    }

    fn collect_env(&self, env: &mut EnvBuilder, _sig: &Signature) -> R<()> {
        env.collect_graph(&self.graph)
    }

    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        _sig: &Signature,
        ins: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        cx.emit_nested_graph(&self.graph, ins, provided)
    }
}
