# ================================================================
# 0. Section: IMPORTS
# ================================================================
import hashlib

import numpy as np

from ...domain_model.data import SampleBatch, ClearData
from .stable_hash import stable_hash



# ================================================================
# 1. Section: Functions
# ================================================================
def hash_sample_batch(batch: SampleBatch) -> str:
    return stable_hash({
        "tissue": hash_clear_data(batch.tissue),
        "cells": hash_clear_data(batch.cells),
        "atlas": hash_clear_data(batch.atlas),
    })


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def hash_clear_data(data: ClearData) -> str:
    return stable_hash({
        "class": data.__class__.__name__,
        "data_hash": hash_array(data.data),
        "resolution": data.resolution,
        "unit": data.unit,
        "orientation": data.orientation,
        "tissue_type": data.tissue_type.name,
    })

def hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)

    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())

    return h.hexdigest()
