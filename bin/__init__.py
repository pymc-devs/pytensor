import warnings

warnings.warn(
    message= "Importing 'bin.pytensor_cache' is deprecated. Import from "
    "'pytensor.bin.pytensor_cache' instead.",
    category=DeprecationWarning,
    stacklevel=2,  # Raise the warning on the import line
)
