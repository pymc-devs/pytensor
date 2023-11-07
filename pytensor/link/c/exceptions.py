try:
    from setuptools.errors import CompileError as BaseCompileError
except ImportError:
    import warnings
    from distutils.errors import CompileError as BaseCompileError  # type: ignore
    from importlib.metadata import version

    # These exception classes were made available in setuptools
    # since v59.0.0 via <https://github.com/pypa/setuptools/pull/2858>
    # in preparation for distutils deprecation. Complain loudly if they
    # are not available.
    setuptools_version = version("setuptools")
    warnings.warn(
        f"You appear to be using an ancient version of setuptools: "
        f"v{setuptools_version}. Please upgrade to at least v59.0.0. "
        f"Support for this version of setuptools is provisionary and "
        f"may be removed without warning in the future."
    )


class MissingGXX(Exception):
    """This error is raised when we try to generate c code, but g++ is not available."""


class CompileError(BaseCompileError):  # pyright: ignore
    """Custom `Exception` prints compilation errors with their original formatting."""

    def __str__(self):
        return self.args[0]
