import logging
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from configparser import (
    ConfigParser,
    InterpolationError,
    NoOptionError,
    NoSectionError,
    RawConfigParser,
)
from functools import wraps
from io import StringIO
from pathlib import Path
from shlex import shlex

from pytensor.utils import hash_from_code


_logger = logging.getLogger("pytensor.configparser")


class PyTensorConfigWarning(Warning):
    @classmethod
    def warn(cls, message: str, stacklevel: int = 0):
        warnings.warn(message, cls, stacklevel=stacklevel + 3)


class ConfigAccessViolation(AttributeError):
    """Raised when a config setting is accessed through the wrong config instance."""


class _ChangeFlagsDecorator:
    def __init__(self, _root=None, **kwargs):
        self.confs = {k: _root._config_var_dict[k] for k in kwargs}
        self.new_vals = kwargs
        self._root = _root

    def __call__(self, f):
        @wraps(f)
        def res(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return res

    def __enter__(self):
        self.old_vals = {}
        for k, v in self.confs.items():
            self.old_vals[k] = v.__get__(self._root, self._root.__class__)
        try:
            for k, v in self.confs.items():
                v.__set__(self._root, self.new_vals[k])
        except Exception:
            _logger.error(f"Failed to change flags for {self.confs}.")
            self.__exit__()
            raise

    def __exit__(self, *args):
        for k, v in self.confs.items():
            v.__set__(self._root, self.old_vals[k])


class PyTensorConfigParser:
    """Object that holds configuration settings."""

    # add_basic_configvars
    floatX: str
    warn_float64: str
    pickle_test_value: bool
    cast_policy: str
    device: str
    print_global_stats: bool
    unpickle_function: bool
    # add_compile_configvars
    mode: str
    cxx: str
    linker: str
    allow_gc: bool
    optimizer: str
    optimizer_verbose: bool
    optimizer_verbose_ignore: str
    compiler_verbose: bool
    on_opt_error: str
    nocleanup: bool
    on_unused_input: str
    gcc__cxxflags: str
    cmodule__warn_no_version: bool
    cmodule__remove_gxx_opt: bool
    cmodule__compilation_warning: bool
    cmodule__preload_cache: bool
    cmodule__age_thresh_use: int
    cmodule__debug: bool
    compile__wait: int
    compile__timeout: int
    # add_tensor_configvars
    tensor__cmp_sloppy: int
    lib__amdlibm: bool
    tensor__insert_inplace_optimizer_validate_nb: int
    # add_traceback_configvars
    traceback__limit: int
    traceback__compile_limit: int
    # add_error_and_warning_configvars
    warn__ignore_bug_before: int
    exception_verbosity: str
    # add_testvalue_and_checking_configvars
    print_test_value: bool
    compute_test_value: str
    compute_test_value_opt: str
    check_input: bool
    NanGuardMode__nan_is_error: bool
    NanGuardMode__inf_is_error: bool
    NanGuardMode__big_is_error: bool
    NanGuardMode__action: str
    DebugMode__patience: int
    DebugMode__check_c: bool
    DebugMode__check_py: bool
    DebugMode__check_finite: bool
    DebugMode__check_strides: int
    DebugMode__warn_input_not_reused: bool
    DebugMode__check_preallocated_output: str
    DebugMode__check_preallocated_output_ndim: int
    profiling__time_thunks: bool
    profiling__n_apply: int
    profiling__n_ops: int
    profiling__output_line_width: int
    profiling__min_memory_size: int
    profiling__min_peak_memory: bool
    profiling__destination: str
    profiling__debugprint: bool
    profiling__ignore_first_call: bool
    on_shape_error: str
    # add_multiprocessing_configvars
    openmp: bool
    openmp_elemwise_minsize: int
    # add_optimizer_configvars
    optimizer_excluding: str
    optimizer_including: str
    optimizer_requiring: str
    optdb__position_cutoff: float
    optdb__max_use_ratio: float
    cycle_detection: str
    check_stack_trace: str
    # add_metaopt_configvars
    metaopt__verbose: int
    # add_vm_configvars
    profile: bool
    profile_optimizer: bool
    profile_memory: bool
    vm__lazy: bool | None
    # add_deprecated_configvars
    unittests__rseed: str
    warn__round: bool
    # add_scan_configvars
    scan__allow_gc: bool
    scan__allow_output_prealloc: bool
    # add_numba_configvars
    numba__fastmath: bool
    numba__cache: bool
    # add_caching_dir_configvars
    compiledir_format: str
    base_compiledir: Path
    compiledir: Path
    # add_blas_configvars
    blas__ldflags: str
    blas__check_openmp: bool

    def __init__(
        self,
        flags_dict: dict,
        pytensor_cfg: ConfigParser,
        pytensor_raw_cfg: RawConfigParser,
    ):
        self._flags_dict = flags_dict
        self._pytensor_cfg = pytensor_cfg
        self._pytensor_raw_cfg = pytensor_raw_cfg
        self._config_var_dict: dict = {}

    def __str__(self, print_doc=True):
        sio = StringIO()
        self.config_print(buf=sio, print_doc=print_doc)
        return sio.getvalue()

    def config_print(self, buf, print_doc: bool = True):
        for cv in self._config_var_dict.values():
            print(cv, file=buf)
            if print_doc:
                print("    Doc: ", cv.doc, file=buf)
            print("    Value: ", cv.__get__(self, self.__class__), file=buf)
            print("", file=buf)

    def get_config_hash(self):
        """
        Return a string sha256 of the current config options. In the past,
        it was md5.

        The string should be such that we can safely assume that two different
        config setups will lead to two different strings.

        We only take into account config options for which `in_c_key` is True.
        """
        all_opts = sorted(
            [c for c in self._config_var_dict.values() if c.in_c_key],
            key=lambda cv: cv.name,
        )
        return hash_from_code(
            "\n".join(
                f"{cv.name} = {cv.__get__(self, self.__class__)}" for cv in all_opts
            )
        )

    def add(self, name: str, doc: str, configparam: "ConfigParam", in_c_key: bool):
        """Add a new variable to PyTensorConfigParser.

        This method performs some of the work of initializing `ConfigParam` instances.

        Parameters
        ----------
        name: string
            The full name for this configuration variable. Takes the form
            ``"[section0__[section1__[etc]]]_option"``.
        doc: string
            A string that provides documentation for the config variable.
        configparam: ConfigParam
            An object for getting and setting this configuration parameter
        in_c_key: boolean
            If ``True``, then whenever this config option changes, the key
            associated to compiled C modules also changes, i.e. it may trigger a
            compilation of these modules (this compilation will only be partial if it
            turns out that the generated C code is unchanged). Set this option to False
            only if you are confident this option should not affect C code compilation.

        """
        if "." in name:
            raise ValueError(
                f"Dot-based sections were removed. Use double underscores! ({name})"
            )

        configparam.doc = doc
        configparam.name = name
        configparam.in_c_key = in_c_key

        # Register it on this instance before the code below already starts accessing it
        self._config_var_dict[name] = configparam

        # Trigger a read of the value from config files and env vars
        # This allow to filter wrong value from the user.
        if not callable(configparam.default):
            configparam.__get__(self, type(self), delete_key=True)
        else:
            # We do not want to evaluate now the default value
            # when it is a callable.
            try:
                self.fetch_val_for_key(name)
                # The user provided a value, filter it now.
                configparam.__get__(self, type(self), delete_key=True)
            except KeyError:
                # This is raised because the underlying `ConfigParser` in
                # `self._pytensor_cfg` does not contain an entry for the given
                # section and/or value.
                _logger.info(
                    f"Suppressed KeyError in PyTensorConfigParser.add for parameter '{name}'!"
                )

        # the ConfigParam implements __get__/__set__, enabling us to create a property:
        setattr(self.__class__, name, configparam)

    def fetch_val_for_key(self, key, delete_key: bool = False):
        """Return the overriding config value for a key.
        A successful search returns a string value.
        An unsuccessful search raises a KeyError

        The (decreasing) priority order is:
        - PYTENSOR_FLAGS
        - ~/.pytensorrc

        """

        # first try to find it in the FLAGS
        if key in self._flags_dict:
            if delete_key:
                return self._flags_dict.pop(key)
            return self._flags_dict[key]

        # next try to find it in the config file

        # config file keys can be of form option, or section__option
        key_tokens = key.rsplit("__", 1)
        if len(key_tokens) > 2:
            raise KeyError(key)

        if len(key_tokens) == 2:
            section, option = key_tokens
        else:
            section, option = "global", key
        try:
            try:
                return self._pytensor_cfg.get(section, option)
            except InterpolationError:
                return self._pytensor_raw_cfg.get(section, option)
        except (NoOptionError, NoSectionError):
            raise KeyError(key)

    def change_flags(self, **kwargs) -> _ChangeFlagsDecorator:
        """
        Use this as a decorator or context manager to change the value of
        PyTensor config variables.

        Useful during tests.
        """
        return _ChangeFlagsDecorator(_root=self, **kwargs)

    def warn_unused_flags(self):
        for key in self._flags_dict:
            warnings.warn(f"PyTensor does not recognise this flag: {key}")


class ConfigParam:
    """Base class of all kinds of configuration parameters.

    A ConfigParam has not only default values and configurable mutability, but
    also documentation text, as well as filtering and validation routines
    that can be context-dependent.

    This class implements __get__ and __set__ methods to eventually become
    a property on an instance of PyTensorConfigParser.
    """

    def __init__(
        self,
        default: object | Callable[[object], object],
        *,
        apply: Callable[[object], object] | None = None,
        validate: Callable[[object], bool] | None = None,
        mutable: bool = True,
    ):
        """
        Represents a configuration parameter and its associated casting and validation logic.

        Parameters
        ----------
        default : object or callable
            A default value, or function that returns a default value for this parameter.
        apply : callable, optional
            Callable that applies a modification to an input value during assignment.
            Typical use cases: type casting or expansion of '~' to user home directory.
        validate : callable, optional
            A callable that validates the parameter value during assignment.
            It may raise an (informative!) exception itself, or simply return True/False.
            For example to check the availability of a path, device or to restrict a float into a range.
        mutable : bool
            If mutable is False, the value of this config settings can not be changed at runtime.
        """
        self._default = default
        self._apply = apply
        self._validate = validate
        self._mutable = mutable
        self.is_default = True
        # set by PyTensorConfigParser.add:
        self.name: str = "unnamed"
        self.doc: str = "undocumented"
        self.in_c_key: bool

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in PyTensorConfigParser.add, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.

    @property
    def default(self):
        return self._default

    @property
    def mutable(self) -> bool:
        return self._mutable

    def apply(self, value):
        """Applies modifications to a parameter value during assignment.

        Typical use cases are casting or the substitution of '~' with the user home directory.
        """
        if callable(self._apply):
            return self._apply(value)
        return value

    def validate(self, value) -> bool:
        """Validates that a parameter values falls into a supported set or range.

        Raises
        ------
        ValueError
            when the validation turns out negative
        """
        if not callable(self._validate):
            return True
        if self._validate(value) is False:
            raise ValueError(
                f"Invalid value ({value}) for configuration variable '{self.name}'."
            )
        return True

    def __get__(self, cls, type_, delete_key=False):
        if cls is None:
            return self
        if self.name not in cls._config_var_dict:
            raise ConfigAccessViolation(
                f"The config parameter '{self.name}' was registered on a different instance of the PyTensorConfigParser."
                f" It is not accessible through the instance with id '{id(cls)}' because of safeguarding."
            )
        try:
            return self.val
        except AttributeError:
            try:
                val_str = cls.fetch_val_for_key(self.name, delete_key=delete_key)
                self.is_default = False
            except KeyError:
                if callable(self.default):
                    val_str = self.default()
                else:
                    val_str = self.default
            self.__set__(cls, val_str)
        return self.val

    def __set__(self, cls, val):
        if not self.mutable and hasattr(self, "val"):
            raise Exception(
                f"Can't change the value of {self.name} config parameter after initialization!"
            )
        applied = self.apply(val)
        self.validate(applied)
        self.val = applied


class EnumStr(ConfigParam):
    def __init__(
        self, default: str, options: Sequence[str], validate=None, mutable: bool = True
    ):
        """Creates a str-based parameter that takes a predefined set of options.

        Parameters
        ----------
        default : str
            The default setting.
        options : sequence
            Further str values that the parameter may take.
            May, but does not need to include the default.
        validate : callable
            See `ConfigParam`.
        mutable : callable
            See `ConfigParam`.
        """
        self.all = {default, *options}

        # All options should be strings
        for val in self.all:
            if not isinstance(val, str):
                raise ValueError(f"Non-str value '{val}' for an EnumStr parameter.")
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, val):
        if val in self.all:
            return val
        else:
            raise ValueError(
                f"Invalid value ('{val}') for configuration variable '{self.name}'. "
                f"Valid options are {self.all}"
            )

    def __str__(self):
        return f"{self.name} ({self.all}) "


class TypedParam(ConfigParam):
    def __str__(self):
        # The "_apply" callable is the type itself.
        return f"{self.name} ({self._apply}) "


class StrParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=str, validate=validate, mutable=mutable)


class IntParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=int, validate=validate, mutable=mutable)


class FloatParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=float, validate=validate, mutable=mutable)


class BoolParam(TypedParam):
    """A boolean parameter that may be initialized from any of the following:
    False, 0, "false", "False", "0"
    True, 1, "true", "True", "1"
    """

    def __init__(self, default, validate=None, mutable: bool = True):
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, value):
        if value in {False, 0, "false", "False", "0"}:
            return False
        elif value in {True, 1, "true", "True", "1"}:
            return True
        raise ValueError(
            f"Invalid value ({value}) for configuration variable '{self.name}'."
        )


class DeviceParam(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        super().__init__(
            default, apply=self._apply, mutable=kwargs.get("mutable", True)
        )

    def _apply(self, val):
        if val.startswith("opencl") or val.startswith("cuda") or val.startswith("gpu"):
            raise ValueError(
                "You are trying to use the old GPU back-end. "
                "It was removed from PyTensor."
            )
        elif val == self.default:
            return val
        raise ValueError(
            f'Invalid value ("{val}") for configuration '
            f'variable "{self.name}". Valid options start with '
            'one of "cpu".'
        )

    def __str__(self):
        return f"{self.name} ({self.default})"


def parse_config_string(
    config_string: str, issue_warnings: bool = True
) -> dict[str, str]:
    """
    Parses a config string (comma-separated key=value components) into a dict.
    """
    config_dict = {}
    my_splitter = shlex(config_string, posix=True)
    my_splitter.whitespace = ","
    my_splitter.whitespace_split = True
    for kv_pair in my_splitter:
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split("=", 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                PyTensorConfigWarning.warn(
                    f"Config key '{kv_tuple[0]}' has no value, ignoring it",
                    stacklevel=1,
                )
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict


def config_files_from_pytensorrc() -> list[Path]:
    """
    PYTENSORRC can contain a colon-delimited list of config files, like

        PYTENSORRC=~/.pytensorrc:/etc/.pytensorrc

    In that case, definitions in files on the right (here, ``~/.pytensorrc``)
    have precedence over those in files on the left.
    """
    paths = [
        Path(s).expanduser()
        for s in os.getenv("PYTENSORRC", "~/.pytensorrc").split(os.pathsep)
    ]
    if os.getenv("PYTENSORRC") is None and sys.platform == "win32":
        # to don't need to change the filename and make it open easily
        paths.append(Path("~/.pytensorrc.txt").expanduser())
    return paths


def _create_default_config() -> PyTensorConfigParser:
    # The PYTENSOR_FLAGS environment variable should be a list of comma-separated
    # [section__]option=value entries. If the section part is omitted, there should
    # be only one section that contains the given option.
    PYTENSOR_FLAGS = os.getenv("PYTENSOR_FLAGS", "")
    PYTENSOR_FLAGS_DICT = parse_config_string(PYTENSOR_FLAGS, issue_warnings=True)

    config_files = config_files_from_pytensorrc()
    pytensor_cfg = ConfigParser(
        {
            "USER": os.getenv("USER", Path("~").expanduser().name),
            "LSCRATCH": os.getenv("LSCRATCH", ""),
            "TMPDIR": os.getenv("TMPDIR", ""),
            "TEMP": os.getenv("TEMP", ""),
            "TMP": os.getenv("TMP", ""),
            "PID": str(os.getpid()),
        }
    )
    pytensor_cfg.read(config_files)
    # Having a raw version of the config around as well enables us to pass
    # through config values that contain format strings.
    # The time required to parse the config twice is negligible.
    pytensor_raw_cfg = RawConfigParser()
    pytensor_raw_cfg.read(config_files)

    # Instances of PyTensorConfigParser can have independent current values!
    # But because the properties are assigned to the type, their existence is global.
    config = PyTensorConfigParser(
        flags_dict=PYTENSOR_FLAGS_DICT,
        pytensor_cfg=pytensor_cfg,
        pytensor_raw_cfg=pytensor_raw_cfg,
    )
    return config
