"""Load config.yaml and resolve project-relative paths."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path=None, overrides=()) -> dict:
    """Load the config, then apply "dotted.key=value" overrides.

    Overrides change one run without editing the file, e.g.
    --set detector.conf=0.25 to try a different detection threshold.
    """
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)  # "3" -> 3, "true" -> True, text stays text
    return cfg


def resolve(path) -> Path:
    """Turn a path from the config into an absolute path under the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p
