import pytensor


def test_root_module_not_polluted():
    import types

    # Filter out submodules since other tests may have imported them
    module_items = sorted(
        i
        for i in dir(pytensor)
        if not i.startswith("__")
        and not isinstance(getattr(pytensor, i), types.ModuleType)
    )
    assert module_items == [
        "In",
        "Lop",
        "Mode",
        "OpFromGraph",
        "Out",
        "Rop",
        "config",
        "dprint",
        "foldl",
        "foldr",
        "function",
        "grad",
        "ifelse",
        "map",
        "reduce",
        "scan",
        "shared",
        "wrap_jax",
        "wrap_py",
    ]
