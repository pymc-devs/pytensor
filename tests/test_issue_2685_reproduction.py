import pytest
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.rewriting.basic import NodeProcessingGraphRewriter
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import NodeRewriter
from pytensor.tensor import scalar

class DummyRewriter(NodeRewriter):
    def transform(self, fgraph, node, enforce_tracks=True):
        return False
    def apply(self, fgraph, node):
        return False

class DummyOp(Op):
    def make_node(self, *inputs):
        return Apply(self, inputs, [scalar("out").type()])
    def perform(self, node, inputs, outputs):
        pass

def test_node_processing_graph_rewriter_none_rewriter():
    # Test that calling process_node with no rewriter doesn't crash
    
    class ConcreteRewriter(NodeProcessingGraphRewriter):
        def apply(self, fgraph):
            return False

    rewriter = ConcreteRewriter(node_rewriter=None, ignore_newtrees=False)
    op = DummyOp()
    node = op.make_node(scalar("in"))
    
    # This should return False, not crash
    result = rewriter.process_node(None, node, node_rewriter=None)
    assert result is False
