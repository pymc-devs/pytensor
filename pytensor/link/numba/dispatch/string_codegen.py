from collections.abc import Sequence
from enum import Enum, auto


def create_tuple_string(x):
    if len(x) == 1:
        return f"({x[0]},)"
    else:
        return f"({', '.join(x)})"


class CODE_TOKEN(Enum):
    INDENT = auto()
    DEDENT = auto()
    EMPTY_LINE = auto()


def build_source_code(code: Sequence[str | CODE_TOKEN]) -> str:
    lines = []
    indentation_level = 0
    for line in code:
        if line is CODE_TOKEN.INDENT:
            indentation_level += 1
        elif line is CODE_TOKEN.DEDENT:
            indentation_level -= 1
            assert indentation_level >= 0
        elif line is CODE_TOKEN.EMPTY_LINE:
            lines.append("")
        else:
            lines.append(f"{'    ' * indentation_level}{line}")
    return "\n".join(lines)
