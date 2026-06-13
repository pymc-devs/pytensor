import warnings
from collections.abc import Collection, Hashable
from functools import singledispatch
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import ComputeMapType, Op, StorageMapType, ThunkType
from pytensor.graph.type import HasDataType
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.interface import CLinkerOp
from pytensor.link.c.op import (
    COp,
    CThunkWrapperType,
    get_io_macros,
    get_sub_macros,
    is_cthunk_wrapper_type,
)
from pytensor.link.c.section_loader import load_c_code_sections
from pytensor.utils import hash_from_code


@singledispatch
def c_funcify(op: Op, node: Apply | None = None, **kwargs) -> CLinkerOp:
    """Return the C implementation of `op` at `node`.

    By default an op implementing `CLinkerOp` (every `COp`) is its own
    implementation; otherwise raise `NotImplementedError` and let the caller fall
    back to the Python thunk.
    """
    if isinstance(op, CLinkerOp):
        return op
    raise NotImplementedError(f"No C implementation registered for {type(op).__name__}")


def _hashable_aliasing_map(aliasing_map: dict[int, list[int]]) -> tuple:
    return tuple(sorted((idx, tuple(vals)) for idx, vals in aliasing_map.items()))


class CImpl(CLinkerOp):
    """A C implementation of an `Op`, detached from the op.

    Returned by `c_funcify`; never a graph op. Subclasses that add configuration
    must extend `_impl_props`, which backs equality, hashing, and the cache key.
    """

    # `Apply.clone_with_new_inputs` reads this off the node's op when the
    # single-node graph is cloned for compilation; impl outputs never depend on
    # input values (the graph op already fixed the output types).
    _output_type_depends_on_input_value = False

    def __init__(
        self,
        op: Op,
        *,
        destroy_map: dict[int, list[int]] | None = None,
        view_map: dict[int, list[int]] | None = None,
    ):
        self.op = op
        if destroy_map is None:
            destroy_map = getattr(op, "destroy_map", {})
        if view_map is None:
            view_map = getattr(op, "view_map", {})
        self.destroy_map = destroy_map
        self.view_map = view_map

    def _impl_props(self) -> tuple:
        return (
            self.op,
            _hashable_aliasing_map(self.destroy_map),
            _hashable_aliasing_map(self.view_map),
        )

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._impl_props() == other._impl_props()

    def __hash__(self) -> int:
        return hash((type(self), *self._impl_props()))

    def __str__(self) -> str:
        return f"{type(self).__name__}{{{self.op}}}"

    def make_node(self, *inputs) -> NoReturn:
        raise RuntimeError(
            f"{type(self).__name__} is a C implementation, not a graph op."
        )

    def prepare_node(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType | None,
        impl: str | None,
    ) -> None:
        """No-op: C preparation happens when `c_funcify` constructs the impl."""


def get_impl_c_macros(
    node: Apply, name: str, macros: dict[str, Any] | None = None
) -> tuple[str, str]:
    """Construct paired C ``#define``/``#undef`` blocks for an impl's code sections.

    Defines ``DTYPE_``/``TYPENUM_``/``ITEMSIZE_`` macros for every input and
    output with a dtype, the ``APPLY_SPECIFIC(str)`` name-mangling macro, and one
    macro per ``macros`` entry.
    """
    define_macros = []
    undef_macros = []

    variables = node.inputs + node.outputs
    variable_names = [f"INPUT_{i}" for i in range(len(node.inputs))] + [
        f"OUTPUT_{i}" for i in range(len(node.outputs))
    ]

    for vname, v in zip(variable_names, variables, strict=True):
        if not isinstance(v.type, HasDataType):
            continue

        dtype = np.dtype(v.type.dtype)
        define_macros.append(f"#define DTYPE_{vname} npy_{v.type.dtype}")
        define_macros.append(f"#define TYPENUM_{vname} {dtype.num}")
        define_macros.append(f"#define ITEMSIZE_{vname} {dtype.itemsize}")
        undef_macros.append(f"#undef DTYPE_{vname}")
        undef_macros.append(f"#undef TYPENUM_{vname}")
        undef_macros.append(f"#undef ITEMSIZE_{vname}")

    define_macros.append(f"#define APPLY_SPECIFIC(str) str##_{name}")
    undef_macros.append("#undef APPLY_SPECIFIC")

    if macros:
        for macro_name, macro_value in macros.items():
            define_macros.append(f"#define {macro_name} {macro_value}")
            undef_macros.append(f"#undef {macro_name}")

    return "\n".join(define_macros), "\n".join(undef_macros)


class CFileImpl(CImpl):
    """A C implementation loaded from ``#section``-marked ``.c`` files.

    Built by `c_impl_from_files`; see its docstring for the file conventions.
    """

    def __init__(
        self,
        op: Op,
        *,
        c_files: list[Path],
        func_name: str | None = None,
        headers: list[str] | None = None,
        header_files: list[Path] | None = None,
        libraries: list[str] | None = None,
        lib_dirs: list[str] | None = None,
        compile_args: list[str] | None = None,
        macros: dict[str, Any] | None = None,
        cache_version: tuple[Hashable, ...] = (),
        destroy_map: dict[int, list[int]] | None = None,
        view_map: dict[int, list[int]] | None = None,
    ):
        super().__init__(op, destroy_map=destroy_map, view_map=view_map)

        self.func_files = tuple(Path(f) for f in c_files)
        self.header_files = tuple(Path(h) for h in header_files) if header_files else ()
        for path in (*self.func_files, *self.header_files):
            if not path.is_absolute():
                raise ValueError(
                    f"c_impl_from_files requires absolute paths, got {path}. "
                    "Resolve against the dispatch module, e.g. "
                    'Path(__file__).parent / "c_code" / ...'
                )

        self.func_name = func_name
        self.headers = tuple(headers) if headers else ()
        self.libraries = tuple(libraries) if libraries else ()
        self.lib_dirs = tuple(lib_dirs) if lib_dirs else ()
        self.compile_args = tuple(compile_args) if compile_args else ()
        self.macros = dict(macros) if macros else {}
        self.cache_version = tuple(cache_version)

        self.func_codes, self.code_sections = load_c_code_sections(self.func_files)
        self.header_codes = [h.read_text(encoding="utf-8") for h in self.header_files]

        if func_name is not None and "code" in self.code_sections:
            raise ValueError("Cannot have a `code` section and specify `func_name`")

    def _impl_props(self) -> tuple:
        return (
            *super()._impl_props(),
            self.func_files,
            self.func_name,
            self.headers,
            self.header_files,
            self.libraries,
            self.lib_dirs,
            self.compile_args,
            tuple(sorted(self.macros.items())),
            self.cache_version,
        )

    def _wrap_in_macros(self, code: str, node: Apply, name: str) -> str:
        define_macros, undef_macros = get_impl_c_macros(node, name, self.macros)
        return f"\n{define_macros}\n{code}\n{undef_macros}"

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        source_hash = hash_from_code("\n".join([*self.func_codes, *self.header_codes]))
        return (
            *self.cache_version,
            source_hash,
            tuple(sorted(self.macros.items())),
        )

    def c_headers(self, **kwargs) -> list[str]:
        return [*self.headers, *(f'"{h.name}"' for h in self.header_files)]

    def c_header_dirs(self, **kwargs) -> list[str]:
        return [str(h.parent) for h in self.header_files]

    def c_libraries(self, **kwargs) -> list[str]:
        return list(self.libraries)

    def c_lib_dirs(self, **kwargs) -> list[str]:
        return list(self.lib_dirs)

    def c_compile_args(self, **kwargs) -> list[str]:
        return list(self.compile_args)

    def c_init_code(self, **kwargs) -> list[str]:
        if "init_code" in self.code_sections:
            return [self.code_sections["init_code"]]
        return super().c_init_code(**kwargs)

    def c_support_code(self, **kwargs) -> str:
        if "support_code" in self.code_sections:
            return self.code_sections["support_code"]
        return super().c_support_code(**kwargs)

    def c_init_code_apply(self, node: Apply, name: str) -> str:
        if "init_code_apply" in self.code_sections:
            return self._wrap_in_macros(
                self.code_sections["init_code_apply"], node, name
            )
        return super().c_init_code_apply(node, name)

    def c_support_code_apply(self, node: Apply, name: str) -> str:
        if "support_code_apply" in self.code_sections:
            return self._wrap_in_macros(
                self.code_sections["support_code_apply"], node, name
            )
        return super().c_support_code_apply(node, name)

    def c_support_code_struct(self, node: Apply, name: str) -> str:
        if "support_code_struct" in self.code_sections:
            return self._wrap_in_macros(
                self.code_sections["support_code_struct"], node, name
            )
        return super().c_support_code_struct(node, name)

    def c_cleanup_code_struct(self, node: Apply, name: str) -> str:
        if "cleanup_code_struct" in self.code_sections:
            return self._wrap_in_macros(
                self.code_sections["cleanup_code_struct"], node, name
            )
        return super().c_cleanup_code_struct(node, name)

    def c_init_code_struct(self, node: Apply, name: str, sub: dict[str, str]) -> str:
        if "init_code_struct" in self.code_sections:
            define_macros, undef_macros = get_impl_c_macros(node, name, self.macros)
            define_sub, undef_sub = get_sub_macros(sub)
            code = self.code_sections["init_code_struct"]
            return (
                f"\n{define_macros}\n{define_sub}\n{code}\n{undef_sub}\n{undef_macros}"
            )
        return super().c_init_code_struct(node, name, sub)

    def c_code(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        if self.func_name is not None:
            define_macros, undef_macros = get_impl_c_macros(node, name, self.macros)
            args = ", ".join([*inputs, *(f"&{o}" for o in outputs)])
            return f"""
                {define_macros}
                {{
                  if ({self.func_name}({args}) != 0) {{
                    {sub["fail"]}
                  }}
                }}
                {undef_macros}
                """

        if "code" in self.code_sections:
            define_macros, undef_macros = get_impl_c_macros(node, name, self.macros)
            define_sub, undef_sub = get_sub_macros(sub)
            define_io, undef_io = get_io_macros(inputs, outputs)
            code = self.code_sections["code"]
            return (
                f"{define_macros}\n{define_sub}\n{define_io}\n{code}"
                f"\n{undef_io}\n{undef_sub}\n{undef_macros}"
            )

        raise NotImplementedError(
            f"{self} has neither a `code` section nor a `func_name`"
        )

    def c_code_cleanup(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        if "code_cleanup" in self.code_sections:
            define_macros, undef_macros = get_impl_c_macros(node, name, self.macros)
            define_sub, undef_sub = get_sub_macros(sub)
            define_io, undef_io = get_io_macros(inputs, outputs)
            code = self.code_sections["code_cleanup"]
            return (
                f"{define_macros}\n{define_sub}\n{define_io}\n{code}"
                f"\n{undef_io}\n{undef_sub}\n{undef_macros}"
            )
        return super().c_code_cleanup(node, name, inputs, outputs, sub)


def c_impl_from_files(
    *,
    op: Op,
    c_files: list[Path],
    func_name: str | None = None,
    headers: list[str] | None = None,
    header_files: list[Path] | None = None,
    libraries: list[str] | None = None,
    lib_dirs: list[str] | None = None,
    compile_args: list[str] | None = None,
    macros: dict[str, Any] | None = None,
    cache_version: tuple[Hashable, ...] = (),
    destroy_map: dict[int, list[int]] | None = None,
    view_map: dict[int, list[int]] | None = None,
) -> CFileImpl:
    """Build a C implementation of `op` from ``.c`` (and ``.h``) files.

    The ``.c`` files use the ``#section`` markers of `ExternalCOp`
    (`pytensor.link.c.section_loader.C_CODE_SECTIONS`); sections in later files
    append to earlier ones. The per-node sections (``*_apply``, ``*_struct``,
    ``code``, ``code_cleanup``) are wrapped in ``#define``/``#undef`` pairs
    providing ``DTYPE_INPUT_i``-style dtype macros, ``APPLY_SPECIFIC`` name
    mangling, and one macro per `macros` entry, so per-op-instance flags are baked
    in at compile time. The compilation-cache version combines `cache_version`
    with a hash of all loaded sources and the macro values, so editing a file or
    changing a flag recompiles automatically.

    Parameters
    ----------
    op : Op
        The graph op being implemented. Registrations receive it from `c_funcify`
        and may bake its props into `macros`.
    c_files : list of Path
        Absolute paths to ``#section``-marked C files.
    func_name : str, optional
        Name of a C function defined in the support sections. When given, the
        generated ``c_code`` is a call site passing every input by value and a
        pointer to every output; the function returns nonzero on error with a
        Python exception set. Mutually exclusive with a ``code`` section.
        Default None.
    headers : list of str, optional
        Literal header names for ``c_headers``, e.g. ``"<math.h>"``. Default None.
    header_files : list of Path, optional
        Absolute paths to ``.h`` files shipped next to the dispatch module. Each
        contributes its name to ``c_headers``, its directory to
        ``c_header_dirs``, and its content to the cache version. Default None.
    libraries, lib_dirs, compile_args : list of str, optional
        Passed through to the corresponding `CLinkerOp` methods. Default None.
    macros : dict mapping str to C literal, optional
        Per-instance ``#define``s wrapped around the per-node sections and
        included in the cache version.
    cache_version : tuple, optional
        Explicit version prefix; bump when the calling registration changes in a
        way the source hash cannot see. Default ().
    destroy_map, view_map : dict, optional
        Override the maps mirrored from `op`.
    """
    return CFileImpl(
        op,
        c_files=c_files,
        func_name=func_name,
        headers=headers,
        header_files=header_files,
        libraries=libraries,
        lib_dirs=lib_dirs,
        compile_args=compile_args,
        macros=macros,
        cache_version=cache_version,
        destroy_map=destroy_map,
        view_map=view_map,
    )


def c_thunk_from_dispatch(
    node: Apply,
    storage_map: StorageMapType,
    compute_map: ComputeMapType | None,
    no_recycling: Collection[Variable],
) -> CThunkWrapperType:
    """Compile a C thunk for `node`, taking its implementation from `c_funcify`.

    Raises
    ------
    NotImplementedError
        If `node.op` has no C implementation, or has float16 inputs/outputs.
    MethodNotDefined
        If the implementation declines this node (e.g. an unsupported dtype).

    Callers fall back to a Python thunk on either.
    """
    # Imported here to avoid an import cycle.
    import pytensor.link.c.basic

    # Resolve eagerly so an unimplemented op raises before prepare_node runs and
    # before any compilation work; CLinker re-resolves (memoized) during codegen.
    c_funcify(node.op, node=node)

    node.op.prepare_node(
        node, storage_map=storage_map, compute_map=compute_map, impl="c"
    )

    node_input_storage = [storage_map[r] for r in node.inputs]
    node_output_storage = [storage_map[r] for r in node.outputs]

    fgraph = FunctionGraph(node.inputs, node.outputs)
    fgraph_no_recycling = [
        new_o
        for (new_o, old_o) in zip(fgraph.outputs, node.outputs, strict=True)
        if old_o in no_recycling
    ]
    cl = pytensor.link.c.basic.CLinker().accept(
        fgraph, no_recycling=fgraph_no_recycling
    )

    # float16 gets special treatment since running unprepared C code will get bad
    # results.
    if not getattr(node.op, "_f16_ok", False):

        def is_f16(t):
            return getattr(t, "dtype", "") == "float16"

        if any(is_f16(i.type) for i in node.inputs) or any(
            is_f16(o.type) for o in node.outputs
        ):
            # get_dynamic_module just tries to build the C code; it raises for
            # impls without C code, in which case we don't want to warn.
            cl.get_dynamic_module()
            warnings.warn(f"Disabling C code for {node.op} due to unsupported float16")
            raise NotImplementedError("float16")

    outputs = cl.make_thunk(
        input_storage=node_input_storage, output_storage=node_output_storage
    )
    thunk, _node_input_filters, _node_output_filters = outputs

    if compute_map is None:
        rval = is_cthunk_wrapper_type(thunk)
    else:
        cm_entries = [compute_map[o] for o in node.outputs]

        @is_cthunk_wrapper_type
        def rval(thunk=thunk, cm_entries=cm_entries):
            thunk()
            for entry in cm_entries:
                entry[0] = True

    rval.thunk = thunk
    rval.cthunk = thunk.cthunk
    rval.inputs = node_input_storage
    rval.outputs = node_output_storage
    rval.lazy = False
    return rval


# Ops whose `make_thunk` is one of these run through the dispatch; anything else
# overrode `make_thunk` (e.g. `IfElse`, `Scan`) and keeps its custom path.
_DEFAULT_MAKE_THUNKS = (Op.make_thunk, COp.make_thunk)


def make_node_thunk_with_c_dispatch(
    node: Apply,
    storage_map: StorageMapType,
    compute_map: ComputeMapType | None,
    no_recycling: Collection[Variable],
    *,
    try_c: bool,
    fallback_impl: str | None = None,
) -> ThunkType:
    """Make a thunk for `node`, trying the C dispatch first when `try_c`.

    When the C attempt fails (no implementation, or the implementation declines
    the node) the fallback passes ``impl="py"`` so `COp.make_thunk` does not
    retry the C path.
    """
    if try_c and type(node.op).make_thunk in _DEFAULT_MAKE_THUNKS:
        try:
            return c_thunk_from_dispatch(node, storage_map, compute_map, no_recycling)
        except (NotImplementedError, MethodNotDefined):
            fallback_impl = "py"
    # Op.make_thunk is untyped upstream; pin the result to its real type.
    thunk: ThunkType = node.op.make_thunk(
        node, storage_map, compute_map, no_recycling, impl=fallback_impl
    )
    return thunk
