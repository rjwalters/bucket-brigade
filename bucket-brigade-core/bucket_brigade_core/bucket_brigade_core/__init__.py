__all__ = ["lib", "ffi"]

import os
from .ffi import ffi

lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), 'libbucket_brigade_core.dylib'))
del os
