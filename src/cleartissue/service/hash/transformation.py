# ================================================================
# 0. Section: IMPORTS
# ================================================================
import inspect
import hashlib
from typing import Type

from ...domain_model.transformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
def hash_transformation_code(step_class: Type[AbstractTransformation]) -> str:
    try:
        source = inspect.getsource(step_class)
    except OSError:
        source = repr(step_class)

    return hashlib.sha256(source.encode("utf-8")).hexdigest()
