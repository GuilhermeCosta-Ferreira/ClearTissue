# ================================================================
# 0. Section: IMPORTS
# ================================================================
import hashlib
import json

import numpy as np

from typing import Any
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def stable_hash(obj: Any) -> str:
    payload = json.dumps(
        normalise_for_hashing(obj),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def normalise_for_hashing(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(key): normalise_for_hashing(value)
            for key, value in sorted(obj.items(), key=lambda item: str(item[0]))
        }

    if isinstance(obj, (list, tuple)):
        return [normalise_for_hashing(value) for value in obj]

    if isinstance(obj, np.generic):
        return obj.item()

    if hasattr(obj, "name") and hasattr(obj, "value"):
        # Enum-like object
        return obj.name

    return repr(obj)
