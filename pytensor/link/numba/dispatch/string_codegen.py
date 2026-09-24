"""Compatibility import for the shared source-code builder."""

from pytensor.link.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)


__all__ = ["CODE_TOKEN", "build_source_code", "create_tuple_string"]
