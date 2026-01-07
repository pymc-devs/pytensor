def create_tuple_string(x):
    if len(x) == 1:
        return f"({x[0]},)"
    else:
        return f"({', '.join(x)})"
