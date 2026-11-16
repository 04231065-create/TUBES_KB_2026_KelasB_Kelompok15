"""Utilitas bersama untuk skrip eksperimen."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy.data import OUT_DIR  # noqa: E402

ART = OUT_DIR / "artifacts"
MODELS = OUT_DIR / "models"
LOGS = ART / "logs"
for d in (ART, MODELS, LOGS, ART / "figures"):
    d.mkdir(parents=True, exist_ok=True)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, name: str):
    path = ART / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path


def load_json(name: str):
    with open(ART / name, "r", encoding="utf-8") as f:
        return json.load(f)


class Logger:
    def __init__(self, name: str):
        self.f = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        self("started " + time.strftime("%Y-%m-%d %H:%M:%S"))

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
