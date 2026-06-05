# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json
import pickle

import numpy as np

from typing import Any
from pathlib import Path


# ================================================================
# 1. Section: Functions
# ================================================================
def load_json(path: Path) -> dict | list:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload


def load_npy(path: Path) -> np.ndarray:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return np.load(path)


def load_pickle(path: Path) -> Any:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(str(path), "rb") as f:
        return pickle.load(f)
