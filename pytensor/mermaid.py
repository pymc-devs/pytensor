from pytensor.d3viz.formatting import PyDotFormatter


def function_to_mermaid(fn):
    formatter = PyDotFormatter()
    dot = formatter(fn)

    nodes = dot.get_nodes()
    edges = dot.get_edges()

    mermaid_lines = ["graph TD"]
    mermaid_lines.append("%% Nodes:")
    for node in nodes:
        name = node.get_name()
        label = node.get_label()
        shape = node.get_shape()

        if label.endswith("."):
            label = f"{label}0"

        if shape == "box":
            shape = "rect"
        else:
            shape = "rounded"

        mermaid_lines.extend(
            [
                f'{name}["{label}"]',
                f"{name}@{{ shape: {shape} }}",
            ]
        )

        fillcolor = node.get_fillcolor()
        if fillcolor is not None and not fillcolor.startswith("#"):
            fillcolor = _color_to_hex(fillcolor)
            mermaid_lines.append(f"style {name} fill:{fillcolor}")

    mermaid_lines.append("\n%% Edges:")
    for edge in edges:
        source = edge.get_source()
        target = edge.get_destination()

        mermaid_lines.append(f"{source} --> {target}")

    return "\n".join(mermaid_lines)


def _color_to_hex(color_name):
    """Based on the colors in d3viz module."""
    return {
        "limegreen": "#32CD32",
        "SpringGreen": "#00FF7F",
        "YellowGreen": "#9ACD32",
        "dodgerblue": "#1E90FF",
        "lightgrey": "#D3D3D3",
        "yellow": "#FFFF00",
        "cyan": "#00FFFF",
        "magenta": "#FF00FF",
        "red": "#FF0000",
        "blue": "#0000FF",
        "green": "#008000",
        "grey": "#808080",
    }.get(color_name)
