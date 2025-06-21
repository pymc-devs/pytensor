from textwrap import dedent

import pytest

from pytensor import function
from pytensor import tensor as pt
from pytensor.mermaid import function_to_mermaid


@pytest.fixture
def sample_function():
    x = pt.dmatrix("x")
    y = pt.dvector("y")
    z = pt.dot(x, y)
    z.name = "z"
    return function([x, y], z)


def test_function_to_mermaid(sample_function):
    diagram = function_to_mermaid(sample_function)

    assert (
        diagram
        == dedent("""
        graph TD
        %% Nodes:
        n1["Shape_i"]
        n1@{ shape: rounded }
        style n1 fill:#00FFFF
        n2["x"]
        n2@{ shape: rect }
        style n2 fill:#32CD32
        n2["x"]
        n2@{ shape: rect }
        style n2 fill:#32CD32
        n4["AllocEmpty"]
        n4@{ shape: rounded }
        n6["CGemv"]
        n6@{ shape: rounded }
        n7["1.0"]
        n7@{ shape: rect }
        style n7 fill:#00FF7F
        n8["y"]
        n8@{ shape: rect }
        style n8 fill:#32CD32
        n9["0.0"]
        n9@{ shape: rect }
        style n9 fill:#00FF7F
        n10["z"]
        n10@{ shape: rect }
        style n10 fill:#1E90FF

        %% Edges:
        n2 --> n1
        n1 --> n4
        n4 --> n6
        n7 --> n6
        n2 --> n6
        n8 --> n6
        n9 --> n6
        n6 --> n10
                             """).strip()
    )
