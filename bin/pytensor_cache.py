#!/usr/bin/env python

import warnings

from pytensor.bin.pytensor_cache import *
from pytensor.bin.pytensor_cache import _logger

if __name__ == "__main__":
    warnings.warn(
        message= "Running 'pytensor_cache.py' is deprecated. Use the pytensor-cache "
        "script instead.",
        category=DeprecationWarning,
    )
    main()
