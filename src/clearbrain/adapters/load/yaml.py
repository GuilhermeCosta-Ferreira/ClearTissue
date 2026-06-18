# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from ruamel.yaml import YAML



# ================================================================
# 1. Section: Functions
# ================================================================
def load_yaml(path: Path) -> dict:
    # 2. Starts the yaml parser
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 3. Loads the default yaml
    with open(path, "r") as f:
        config = yaml.load(f)

    return config


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
